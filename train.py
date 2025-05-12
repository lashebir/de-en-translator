import torch
from torch.optim import AdamW
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from model import Seq2SeqTransformer
from preprocess import load_tokenizer, get_dataloader_from_files, train_tokenizer, split_parallel_corpus_chunked, get_dataloaders3, train_tokenizer2, get_dataloaders4, make_loader
import torch.nn.functional as F
import os
import pandas as pd
import datetime
import shutil
import sys
import math
import gc
import time
from sklearn.model_selection import train_test_split

def my_check_disk_space(path=".", min_space_gb=1):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)  # Convert to GB
    if free_gb < min_space_gb:
        print(f"Warning: Only {free_gb}GB of free space available. Need at least {min_space_gb}GB.")
        return False
    return True

def save_model_locally(model, optimizer, epoch, train_loss, val_loss, path="checkpoints"):
    """Save model checkpoint locally and keep only the last 2 checkpoints."""
    os.makedirs(path, exist_ok=True)
    
    # Save new checkpoint
    checkpoint_path = f"{path}/model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    
    # Clean up old checkpoints, keeping only the last 2
    checkpoint_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
    if len(checkpoint_files) > 2:
        for old_file in checkpoint_files[:-2]:  # Keep last 2 files
            try:
                os.remove(os.path.join(path, old_file))
                print(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                print(f"Error removing old checkpoint {old_file}: {e}")
    
    return checkpoint_path

def save_training_results_to_df(results_dict, filename="training_results.csv"):
    """Save training results to a pandas DataFrame and CSV file."""
    # Create DataFrame
    df = pd.DataFrame([results_dict])
    
    # Add timestamp
    df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # If file exists, append to it
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nTraining results saved to {filename}")
    return df

def train_epoch(model, train_loader, optimizer, device, pad_id):
    model.train()
    total_loss = 0
    num_batches = 0
    total_tokens = 0
    start_time = time.time()

    try:
        # Create progress bar with total length
        pbar = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True, 
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                input_ids = batch['input_ids'].to(device)
                target_input = batch['target_input'].to(device)
                target_output = batch['labels'].to(device)


                # Check max values in input_ids, target_input, and target_output
                print("Max input_id:", input_ids.max().item())
                print("Max target_input:", target_input.max().item())
                print("Max label:", target_output.max().item())

                # Count non-padding tokens
                num_tokens = (target_output != -100).sum().item()
                total_tokens += num_tokens

                # Forward pass
                if input_ids.max() >= model.embedding.num_embeddings:
                    raise ValueError(f"input_id too large: {input_ids.max()}")

                if target_input.max() >= model.embedding.num_embeddings:
                    raise ValueError(f"target_input too large: {target_input.max()}")

                if target_output.max() >= model.embedding.num_embeddings:
                    raise ValueError(f"target_output too large: {target_output.max()}")


                assert input_ids.max() < model.embedding.num_embeddings
                assert target_input.max() < model.embedding.num_embeddings

                output = model(input_ids, target_input)

                # Reshape for loss computation
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    target_output.reshape(-1),
                    ignore_index=-100 #-100
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate metrics
                avg_loss = total_loss / num_batches
                elapsed_time = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
                
                # Update progress bar with more metrics
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'tokens/s': f'{tokens_per_sec:.0f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'batch': f'{batch_idx + 1}/{len(train_loader)}'
                })

            except Exception as e:
                print(f"\nError in batch {batch_idx + 1}: {str(e)}")
                print(f"Input shapes - input_ids: {input_ids.shape}, target_input: {target_input.shape}, target_output: {target_output.shape}")
                continue

        return total_loss / num_batches if num_batches > 0 else float('inf')

    except Exception as e:
        print(f"\nError in training epoch: {str(e)}")
        raise

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(device)
                target_input = batch['target_input'].to(device)
                target_output = batch['labels'].to(device)

                # Forward pass
                output = model(input_ids, target_input)
                
                # Reshape for loss computation
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    target_output.reshape(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"\nError in validation batch: {str(e)}")
                continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def translate(model, tokenizer, text, device, max_length=128):
    model.eval()
    with torch.no_grad():
        # Build source: [DE] ... <eos>
        de_token = tokenizer.piece_to_id("[DE]")
        print("[DE] ID:", tokenizer.piece_to_id("[DE]"))

        input_ids = [de_token] + tokenizer.encode(text) + [tokenizer.eos_id()]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Build initial decoder input: [EN] <bos>
        en_token = tokenizer.piece_to_id("[EN]")
        print("[EN] ID:", tokenizer.piece_to_id("[EN]"))
        
        bos_token = tokenizer.bos_id()
        target = torch.tensor([[en_token, bos_token]], dtype=torch.long).to(device)

        for _ in range(max_length):
            output = model(input_tensor, target)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            target = torch.cat([target, next_token], dim=1)

            if next_token.item() == tokenizer.eos_id():
                break

        decoded = tokenizer.decode(target[0].tolist()[2:])  # Skip [EN] and BOS
        return decoded


def main():
    # # Set up MLflow
    # mlflow.set_tracking_uri("file:./mlruns")
    # mlflow.set_experiment("german-english-translator")
    # Check disk space before starting
    if not my_check_disk_space():
        print("Error: Not enough disk space to proceed with training.")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train new tokenizer
    print("Training new tokenizer...")
    
    # Split into train/val/test which MUST match indices for the seq2seq too
    with open("de-en/europarl-v7.de-en.de", "r", encoding="utf-8") as f:
        de_lines = [line.strip() for line in f]
    with open("de-en/europarl-v7.de-en.en", "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]

    # Optional limit
    LIMIT = 10000  # or whatever number you want
    de_lines = de_lines[:LIMIT]
    en_lines = en_lines[:LIMIT]


    assert len(de_lines) == len(en_lines), "Mismatch between DE and EN file lengths"
    num_samples = len(de_lines)
    indices = list(range(num_samples))

    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    de_train = [de_lines[i] for i in train_idx]
    en_train = [en_lines[i] for i in train_idx]

    de_val = [de_lines[i] for i in val_idx]
    en_val = [en_lines[i] for i in val_idx]

    de_test = [de_lines[i] for i in test_idx]
    en_test = [en_lines[i] for i in test_idx]

    with open("tokenizer_input.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(de_train + en_train))


    # train_tokenizer(
    #     de_file="train_tokenizer.de",
    #     en_file="train_tokenizer.en",
    #     vocab_size=32000  # You can adjust this if needed
    # )
    train_tokenizer2(input="tokenizer_input.txt")

    # train_tokenizer(
    #     de_file='de-en/europarl-v7.de-en.de',
    #     en_file='de-en/europarl-v7.de-en.en',
    #     vocab_size=32000  # You can adjust this if needed
    # )
    print("Tokenizer training complete!")


    # print("Re-splitting dataset using new tokenizer...")
    # split_parallel_corpus_chunked(
    #     de_file='de-en/europarl-v7.de-en.de',
    #     en_file='de-en/europarl-v7.de-en.en',
    #     output_dir='de-en/split',
    #     train_ratio=0.7,
    #     val_ratio=0.2,
    #     test_ratio=0.1,
    #     lim=10000  # or however much you're loading
    # )

    # print("Dataset split complete!")

    # Get dataloaders and tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    
    # Print tokenizer info
    print("\nTokenizer Info:")
    print(f"Vocabulary size: {tokenizer.get_piece_size()}")
    print(f"Special tokens:")
    print(f"  [PAD] ID: {tokenizer.pad_id()}")
    print(f"  [UNK] ID: {tokenizer.unk_id()}")
    print(f"  [BOS] ID: {tokenizer.bos_id()}")
    print(f"  [EOS] ID: {tokenizer.eos_id()}")
    print(f"  [DE] ID: {tokenizer.piece_to_id('[DE]')}")
    print(f"  [EN] ID: {tokenizer.piece_to_id('[EN]')}")

    print("\nCreating dataloaders...")
    # train_loader = get_dataloader_from_files("de-en/split/train.de", "de-en/split/train.en", tokenizer, batch_size=32, lim=10000)
    # val_loader = get_dataloader_from_files("de-en/split/val.de", "de-en/split/val.en", tokenizer, batch_size=32, lim=10000)
    # test_loader = get_dataloader_from_files("de-en/split/test.de", "de-en/split/test.en", tokenizer, batch_size=32, lim=10000)

    # train_loader, val_loader, test_loader = get_dataloaders3("de-en/split/train.de", "de-en/split/train.en", tokenizer, batch_size=32)
    train_loader = make_loader(de_train, en_train, batch_size_=32, tokenizer=tokenizer)
    val_loader = make_loader(de_val, en_val, batch_size_=32, tokenizer=tokenizer)
    test_loader = make_loader(de_test, en_test, batch_size_=32, tokenizer=tokenizer)

    print("\nTraining set sample:")
    for batch in train_loader:
        print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
        # label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        label_ids = batch['labels'][0].tolist()
        label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        print("Target (EN):", tokenizer.decode(label_ids))
        break

    print("\nValidation set sample:")
    for batch in val_loader:
        print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
        # label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        label_ids = batch['labels'][0].tolist()
        label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        print("Target (EN):", tokenizer.decode(label_ids))
        break

    print("\nTest set sample:")
    for batch in test_loader:
        print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
        # label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        label_ids = batch['labels'][0].tolist()
        label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
        print("Target (EN):", tokenizer.decode(label_ids))
        break
    # print("pad_token_id:", tokenizer.pad_id() if hasattr(tokenizer, "pad_id") else "n/a")
    # print("unk_token_id:", tokenizer.unk_id() if hasattr(tokenizer, "unk_id") else "n/a")
    pad_id = tokenizer.pad_id()
    
    # Model parameters
    model_params = {
        "vocab_size": tokenizer.get_piece_size(),
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.0
    }
    
    # Training parameters
    training_params = {
        "learning_rate": 1e-4,
        "betas": (0.9, 0.98),
        "weight_decay": 0.01,
        "num_epochs": 10,
        "batch_size": 32,
        "warmup_steps": 4000,
        "gradient_clip_val": 1.0,
        "dropout": 0.0
    }

    try:
        # Initialize model and optimizer
        model = Seq2SeqTransformer(**model_params).to(device)
        optimizer = AdamW(model.parameters(), 
                        lr=training_params["learning_rate"],
                        betas=training_params["betas"],
                        weight_decay=training_params["weight_decay"])
        
        # Add learning rate scheduler
        # def get_lr_scheduler(optimizer, warmup_steps):
        #     def lr_lambda(step):
        #         if step < warmup_steps:
        #             return float(step) / float(max(1, warmup_steps))
        #         return 1.0
        #     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # scheduler = get_lr_scheduler(optimizer, training_params["warmup_steps"])
        # global_step = 0
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(training_params["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{training_params['num_epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, pad_id)
            
            # Evaluate
            val_loss = evaluate(model, val_loader, device)
            
            # Update learning rate
            # scheduler.step(global_step)
            # global_step += len(train_loader)  # Increment by number of batches in epoch
            
            # Save checkpoint locally (keeping only last 2)
            checkpoint_path = save_model_locally(model, optimizer, epoch + 1, train_loss, val_loss)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'checkpoint_path': checkpoint_path,
                'batch_size': training_params['batch_size'],
                'learning_rate': optimizer.param_groups[0]['lr'],  # Current learning rate
                'device': str(device)
            }
            training_history.append(epoch_results)
            
            # Save results to CSV
            df = pd.DataFrame(training_history)
            results_file = f"training_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(results_file, index=False)
            print(f"Saved training results to {results_file}")
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Test translation
            test_text = "Hallo, wie geht es dir?"
            translation = translate(model, tokenizer, test_text, device)
            print(f"\nSample translation:")
            print(f"German: {test_text}")
            print(f"English: {translation}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Check disk space before next epoch
            if not my_check_disk_space():
                print("Warning: Low disk space. Saving current state and stopping training.")
                break

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Save the model state even if training fails
        if 'model' in locals():
            save_model_locally(model, optimizer, epoch + 1, train_loss, val_loss, path="error_checkpoints")
        sys.exit(1)

if __name__ == "__main__":
    main()