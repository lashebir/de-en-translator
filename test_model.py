import torch
from model import Seq2SeqTransformer
from preprocess import load_tokenizer, get_dataloader_from_files
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import nltk
import os
import sentencepiece as spm
from preprocess import collate_fn
import pandas as pd
from datetime import datetime
import datetime

def load_mlflow_model(model_path, device):
    """Load a trained model from MLflow."""
    model = mlflow.pytorch.load_model(model_path)
    model.to(device)
    model.eval()
    return model

def load_tokenizer():
    """Load the trained tokenizer."""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('translation_tokenizer.model')
    return tokenizer

def translate_batch(model, tokenizer, texts, device, max_length=128):
    """Translate a batch of texts."""
    translations = []
    for text in texts:
        translation = translate(model, tokenizer, text, device, max_length)
        translations.append(translation)
    return translations

def translate(model, tokenizer, text, device, max_length=128):
    """Translate a single text."""
    model.eval()
    with torch.no_grad():
        # Tokenize input
        # input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
        input_ids = [tokenizer.bos_id()] + tokenizer.encode(text) + [tokenizer.eos_id()]
        input_ids = input_ids[:max_length]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # Initialize target sequence with BOS token
        target = torch.tensor([[tokenizer.bos_id()]]).to(device)
        # target = [tokenizer.bos_id()] +[tokenizer.encode(text) + [tokenizer.eos_id()]]
        # target = target[:max_length]
        # target = torch.tensor(target, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate translation
        for _ in range(max_length):
            output = model(input_tensor, target)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            target = torch.cat([target, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_id():
                break
        
        return tokenizer.decode(target[0].tolist())

def calculate_bleu(references, hypotheses):
    """Calculate BLEU score for translations."""
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Tokenize references and hypotheses
    references = [[ref.split()] for ref in references]
    hypotheses = [hyp.split() for hyp in hypotheses]
    
    # Calculate BLEU score
    return corpus_bleu(references, hypotheses)

def save_results_to_df(results_dict, filename="translation_results.csv"):
    """Save results to a pandas DataFrame and CSV file."""
    # Create DataFrame
    df = pd.DataFrame([results_dict])
    
    # Add timestamp
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # If file exists, append to it
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    return df

def test_model(model_path, device, batch_size=32, new_test_files=None):
    """
    Test the model on either the pre-split test set or new test files.
    
    Args:
        model_path: Path to the trained model
        device: Device to run the model on
        batch_size: Batch size for testing
        new_test_files: Optional tuple of (de_file, en_file) for new test data
    """
    # Load model
    model = load_mlflow_model(model_path, device)
    
    if new_test_files is None:
        # Use pre-split test set
        print("Using pre-split test set...")
        _, _, test_loader, tokenizer = get_dataloaders(
            'de-en/europarl-v7.de-en.de',
            'de-en/europarl-v7.de-en.en',
            batch_size=batch_size,
            test_indices='test_indices.npy'  # Use saved test indices
        )
    else:
        # Use new test files
        print("Using new test files...")
        de_file, en_file = new_test_files
        tokenizer = load_tokenizer()
        
        # Create dataset and dataloader for new test files
        test_dataset = TranslationDataset(de_file, en_file, tokenizer)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, pad_id=0)
        )
    
    # Test translations
    all_references = []
    all_hypotheses = []
    all_sources = []
    
    print(f"\nTesting model on {len(test_loader.dataset)} examples...")
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Get source texts
        sources = [tokenizer.decode(ids.tolist()) for ids in input_ids]
        all_sources.extend(sources)
        
        # Get reference translations
        references = [tokenizer.decode(label.tolist()) for label in labels]
        all_references.extend(references)
        
        # Get model translations
        hypotheses = translate_batch(model, tokenizer, sources, device)
        all_hypotheses.extend(hypotheses)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(all_references, all_hypotheses)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Number of test examples: {len(test_loader.dataset)}")
    print("\nExample translations:")
    for i in range(min(5, len(all_sources))):
        print(f"\nGerman: {all_sources[i]}")
        print(f"English (Reference): {all_references[i]}")
        print(f"English (Model): {all_hypotheses[i]}")
    print("="*50)
    
    # Save results to file
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/results.txt", "w") as f:
        f.write("TEST RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        f.write(f"Number of test examples: {len(test_loader.dataset)}\n")
        f.write("\nExample translations:\n")
        for i in range(min(10, len(all_sources))):
            f.write(f"\nGerman: {all_sources[i]}\n")
            f.write(f"English (Reference): {all_references[i]}\n")
            f.write(f"English (Model): {all_hypotheses[i]}\n")
    
    print(f"\nResults saved to test_results/results.txt")
    
    # Save results to DataFrame
    results_dict = {
        'bleu_score': bleu_score,
        'test_set_size': len(test_loader.dataset),
        'model_path': model_path,
        'batch_size': batch_size,
        'device': str(device),
        'test_files': str(new_test_files) if new_test_files else 'pre-split'
    }
    
    # Add example translations to results
    for i in range(min(5, len(all_sources))):
        results_dict[f'example_{i+1}_source'] = all_sources[i]
        results_dict[f'example_{i+1}_reference'] = all_references[i]
        results_dict[f'example_{i+1}_hypothesis'] = all_hypotheses[i]
    
    # Save to DataFrame
    df = save_results_to_df(results_dict)
    
    # Log results to MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("bleu_score", bleu_score)
        mlflow.log_param("test_set_size", len(test_loader.dataset))
        if new_test_files:
            mlflow.log_param("test_files", str(new_test_files))
        
        # Save example translations
        os.makedirs("test_results", exist_ok=True)
        with open("test_results/example_translations.txt", "w") as f:
            for i in range(min(10, len(all_sources))):
                f.write(f"German: {all_sources[i]}\n")
                f.write(f"English (Reference): {all_references[i]}\n")
                f.write(f"English (Model): {all_hypotheses[i]}\n\n")
        
        mlflow.log_artifact("test_results")
        
        # Log the DataFrame as an artifact
        df.to_csv(f"test_results/metrics_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        mlflow.log_artifact("test_results/metrics.csv")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test on pre-split test set
    print("\nTesting on pre-split test set:")
    test_model(
        model_path="mlruns/0/latest/artifacts/final_model",
        device=device
    )
    
    # Example of testing on new files
    # print("\nTesting on new test files:")
    # test_model(
    #     model_path="mlruns/0/latest/artifacts/final_model",
    #     device=device,
    #     new_test_files=("path/to/new_test.de", "path/to/new_test.en")
    # ) 