# import torch
# from torch.utils.data import Dataset, DataLoader, IterableDataset
# from transformers import AutoTokenizer
# import pandas as pd
# from tqdm import tqdm
# import sentencepiece as spm
# import os
# import numpy as np
# from torch.nn.utils.rnn import pad_sequence
# import re
# import random

# def get_dataloader_from_files(de_file, en_file, tokenizer, batch_size, pad_id=0, lim=10000):
#     # testing testing
#     dataset = TranslationIterableDataset(de_file, en_file, tokenizer, lim=lim, max_length=128)
#     return DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id,tokenizer=tokenizer))





# # de_file, en_file, tokenizer, lim, max_length=128)







# TAG_RE = re.compile(r'<[^>]+>')
# def remove_html(text):
#     return TAG_RE.sub('', text)

# # class TranslationDataset(Dataset):
# #     def __init__(self, de_file, en_file, tokenizer, lim, max_length=128):
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
# #         self.lim = lim
        
# #         with open(de_file, 'r', encoding='utf-8') as f:
# #             self.de_texts = []
# #             for i, line in enumerate(f):
# #                 if i >= self.lim:
# #                     break
# #                 self.de_texts.append(remove_html(line.strip()))

        
# #         with open(en_file, 'r', encoding='utf-8') as f:
# #             self.en_texts = []
# #             for i, line in enumerate(f):
# #                 if i >= self.lim:
# #                     break
# #                 self.en_texts.append(remove_html(line.strip()))

# #         print(f"Loaded {len(self.de_texts)} sentence pairs")

# def split_parallel_corpus_chunked(de_file, en_file, output_dir,
#                                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
#                                    seed=42, lim=50000):
#     assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     os.makedirs(output_dir, exist_ok=True)

#     # Step 1: Load only `lim` lines
#     pairs = []
#     with open(de_file, "r", encoding="utf-8") as de_f, open(en_file, "r", encoding="utf-8") as en_f:
#         for i, (de_line, en_line) in enumerate(zip(de_f, en_f)):
#             if lim is not None and i >= lim:
#                 break
#             pairs.append((de_line.strip(), en_line.strip()))

#     print(f"Loaded {len(pairs)} lines")

#     # Step 2: Shuffle once
#     random.seed(seed)
#     random.shuffle(pairs)

#     # Step 3: Compute split sizes
#     total = len(pairs)
#     train_end = int(train_ratio * total)
#     val_end = train_end + int(val_ratio * total)

#     splits = {
#         "train": pairs[:train_end],
#         "val": pairs[train_end:val_end],
#         "test": pairs[val_end:]
#     }

#     # Step 4: Write each split to disk
#     for split, data in splits.items():
#         with open(os.path.join(output_dir, f"{split}.de"), "w", encoding="utf-8") as de_out, \
#              open(os.path.join(output_dir, f"{split}.en"), "w", encoding="utf-8") as en_out:
#             for de_line, en_line in data:
#                 de_out.write(de_line + "\n")
#                 en_out.write(en_line + "\n")

#     print(f"Finished writing splits to '{output_dir}'")


# class TranslationIterableDataset(IterableDataset):
#     def __init__(self, de_file, en_file, tokenizer, lim=None, max_length=128):
#         self.de_file = de_file
#         self.en_file = en_file
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.lim = lim

#     def line_reader(self):
#         de_token_id = self.tokenizer.piece_to_id("[DE]")
#         en_token_id = self.tokenizer.piece_to_id("[EN]")
#         with open(self.de_file, 'r', encoding='utf-8') as de_f, open(self.en_file, 'r', encoding='utf-8') as en_f:
#             for i, (de_line, en_line) in enumerate(zip(de_f, en_f)):
#                 if self.lim is not None and i >= self.lim:
#                     break
#                 de_text = remove_html(de_line.strip())
#                 en_text = remove_html(en_line.strip())

#                 # Add [DE] to source, [EN] and BOS to target
#                 source = [de_token_id] + self.tokenizer.encode(de_text) + [self.tokenizer.eos_id()]
#                 target = [en_token_id, self.tokenizer.bos_id()] + self.tokenizer.encode(en_text) + [self.tokenizer.eos_id()]
#                 source = source[:self.max_length]
#                 target = target[:self.max_length]

#                 yield torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)

#     def __iter__(self):
#         return self.line_reader()
        
#     def __len__(self):
#         # If lim is set, return it; otherwise, count lines in file
#         if self.lim is not None:
#             return self.lim
#         with open(self.de_file, 'r', encoding='utf-8') as f:
#             return sum(1 for _ in f)

# def train_tokenizer(de_file, en_file, vocab_size=320000):
#     # Train SentencePiece tokenizer on both German and English data
#     spm.SentencePieceTrainer.train(
#         input=[de_file, en_file],
#         model_prefix='translation_tokenizer',
#         vocab_size=vocab_size,
#         character_coverage=1.0,
#         model_type='bpe',
#         pad_id=0,
#         unk_id=1,
#         bos_id=2,
#         eos_id=3,
#         pad_piece='[PAD]',
#         unk_piece='[UNK]',
#         bos_piece='[BOS]',
#         eos_piece='[EOS]',
#         user_defined_symbols='[EN],[DE]'
#     )
    
#     # Load the trained tokenizer
#     tokenizer = spm.SentencePieceProcessor()
#     tokenizer.load('translation_tokenizer.model')
#     return tokenizer

# # def collate_fn(batch, pad_id=0, bos_id=2):
# #     inputs = [x[0] for x in batch]
# #     targets = [x[1] for x in batch]

# #     # Pad inputs
# #     padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
# #     attention_mask = (padded_inputs != pad_id).long()

# #     # Build decoder_input and label_target
# #     decoder_inputs = []
# #     label_outputs = []

# #     for target in targets:
# #         # Shift right: <BOS> w1 w2 ... → w1 w2 ... <EOS>
# #         # Add [EN] token at the start (assume tokenizer.piece_to_id('[EN]') works)
# #         en_token = torch.tensor([tokenizer.piece_to_id('[EN]')])
# #         decoder_input = torch.cat([en_token, target[:-1]])
# #         label_output = target[1:]  # optionally skip [EN] here, or keep it for now

# #         # Guarantee same length
# #         min_len = min(len(decoder_input), len(label_output))
# #         decoder_inputs.append(decoder_input[:min_len])
# #         label_outputs.append(label_output[:min_len])

# #     # Now pad them
# #     padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)
# #     padded_labels = pad_sequence(label_outputs, batch_first=True, padding_value=-100)

# #     return {
# #         "input_ids": padded_inputs,
# #         "target_input": padded_decoder_inputs,
# #         "labels": padded_labels,
# #         "attention_mask": attention_mask
# #     }

# def collate_fn(batch, pad_id=0, tokenizer=None):
#     assert tokenizer is not None
#     de_token_id = tokenizer.piece_to_id("[DE]")
#     en_token_id = tokenizer.piece_to_id("[EN]")
#     bos_id = tokenizer.bos_id()
#     eos_id = tokenizer.eos_id()
#     vocab_size = tokenizer.get_piece_size()

#     inputs, decoder_inputs, label_outputs = [], [], []
#     for src, tgt in batch:
#         # Ensure source tokens are within vocabulary range
#         src = torch.clamp(src, 0, vocab_size - 1)
#         # Add [DE] token to source and EOS token
#         src_tensor = torch.cat([torch.tensor([de_token_id]), src, torch.tensor([eos_id])])
        
#         # Ensure target tokens are within vocabulary range
#         tgt = torch.clamp(tgt, 0, vocab_size - 1)
#         # Add [EN] token, BOS token, and EOS token to target
#         tgt_tensor = torch.cat([torch.tensor([en_token_id, bos_id]), tgt, torch.tensor([eos_id])])

#         # Create decoder input (shifted right) and label output
#         decoder_input = tgt_tensor[:-1]
#         label_output = tgt_tensor[1:]

#         # Ensure same length
#         min_len = min(len(decoder_input), len(label_output))
#         decoder_inputs.append(decoder_input[:min_len])
#         label_outputs.append(label_output[:min_len])
#         inputs.append(src_tensor)

#     # Pad sequences
#     padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
#     padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)
#     padded_labels = pad_sequence(label_outputs, batch_first=True, padding_value=-100)
#     attention_mask = (padded_inputs != pad_id).long()

#     return {
#         "input_ids": padded_inputs,
#         "target_input": padded_decoder_inputs,
#         "labels": padded_labels,
#         "attention_mask": attention_mask
#     }



# # def collate_fn(batch, pad_id=0):
# #     # Unzip the batch into inputs and targets
# #     inputs = [x[0] for x in batch]
# #     targets = [x[1] for x in batch]

# #     # Pad sequences
# #     padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
# #     padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)

# #     # Create attention mask for inputs
# #     attention_mask = (padded_inputs != pad_id).long()

# #     # For the decoder input, we use the target sequence but shift it right
# #     # by adding BOS token at the start and removing the last token
# #     decoder_inputs = []
# #     for target in targets:
# #         # Add BOS token at the start
# #         decoder_input = torch.cat([torch.tensor([2]), target[:-1]])  # 2 is BOS token ID
# #         decoder_inputs.append(decoder_input)
    
# #     padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)

# #     return {
# #         "input_ids": padded_inputs,
# #         "target_input": padded_decoder_inputs,
# #         "labels": padded_targets,
# #         "attention_mask": attention_mask
# #     }

# def get_dataloaders(de_file, en_file, batch_size=32, test_indices=None, lim=10000):
#     # Train and load tokenizer
#     tokenizer = train_tokenizer(de_file, en_file)
#     print("Tokenizer loaded!")

#     # Create dataset
#     dataset = TranslationIterableDataset(de_file, en_file, tokenizer,lim=lim)
#     print("\nDataset created!")
    
#     if test_indices is None:
#         # Calculate sizes for 70-20-10 split
#         total_size = len(dataset)
#         train_size = int(0.7 * total_size)  # 70% for training
#         val_size = int(0.1 * total_size)    # 10% for val
#         test_size = total_size - train_size - val_size  # Remaining 20% for test
        
#         # Create the splits
#         train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#             dataset, 
#             [train_size, val_size, test_size],
#             generator=torch.Generator().manual_seed(42)  # For reproducibility
#         )
#         print("\nRandom split completed")
#         # Save test indices
#         test_indices = test_dataset.indices
#         np.save('test_indices.npy', test_indices)
#     else:
#         # Use provided test indices
#         test_indices = np.load(test_indices)
#         total_size = len(dataset)
#         test_size = len(test_indices)
        
#         # Calculate remaining sizes for 70-20 split
#         remaining_size = total_size - test_size
#         train_size = int(0.7 * total_size)  # 70% of total for training
#         val_size = total_size - train_size - test_size  # Rest for validation
        
#         # Create splits using the saved test indices
#         train_val_indices = list(set(range(total_size)) - set(test_indices))
#         train_indices = train_val_indices[:train_size]
#         val_indices = train_val_indices[train_size:]
        
#         train_dataset = torch.utils.data.Subset(dataset, train_indices)
#         val_dataset = torch.utils.data.Subset(dataset, val_indices)
#         test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     print(f"Dataset sizes - Train: {len(train_dataset)} (70%), Val: {len(val_dataset)} (20%), Test: {len(test_dataset)} (10%)")
#     print(f"Total dataset size: {total_size}")

#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         collate_fn=lambda batch: collate_fn(batch, pad_id=0), num_workers=0
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         collate_fn=lambda batch: collate_fn(batch, pad_id=0, num_workers=0)
#     )
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         collate_fn=lambda batch: collate_fn(batch, pad_id=0, num_workers=0)
#     )
    
#     return train_loader, val_loader, test_loader, tokenizer

# def get_dataloaders2(
#     train_de, train_en,
#     val_de, val_en,
#     test_de, test_en,
#     batch_size=32,
#     vocab_train_src=None,
#     vocab_train_tgt=None,
#     vocab_size=32000,
#     max_length=128
# ):
#     # Train and load tokenizer using original full corpus (or vocab-specific subset)
#     tokenizer = train_tokenizer(
#         vocab_train_src or train_de,
#         vocab_train_tgt or train_en,
#         vocab_size=vocab_size
#     )
#     print("✅ Tokenizer trained and loaded")

#     # Build dataset objects
#     train_dataset = TranslationIterableDataset(train_de, train_en, tokenizer, lim=None, max_length=max_length)
#     val_dataset   = TranslationIterableDataset(val_de, val_en, tokenizer, lim=None, max_length=max_length)
#     test_dataset  = TranslationIterableDataset(test_de, test_en, tokenizer, lim=None, max_length=max_length)

#     print("✅ Datasets created")

#     # Build dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_id=0), num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, pad_id=0), num_workers=0)
#     test_loader  = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, pad_id=0, num_workers=0))

#     return train_loader, val_loader, test_loader, tokenizer


# if __name__ == "__main__":
#     # Test the dataloaders
#     train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
#         'de-en/europarl-v7.de-en.de',
#         'de-en/europarl-v7.de-en.en'
#     )
    
#     # Print a sample batch from each loader
#     print("\nTraining set sample:")
#     for batch in train_loader:
#         print("Input shape:", batch['input_ids'].shape)
#         print("Sample input:", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Sample target:", tokenizer.decode(batch['labels'][0].tolist()))
#         break
        
#     print("\nValidation set sample:")
#     for batch in val_loader:
#         print("Input shape:", batch['input_ids'].shape)
#         print("Sample input:", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Sample target:", tokenizer.decode(batch['labels'][0].tolist()))
#         break
        
#     print("\nTest set sample:")
#     for batch in test_loader:
#         print("Input shape:", batch['input_ids'].shape)
#         print("Sample input:", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Sample target:", tokenizer.decode(batch['labels'][0].tolist()))
#         break 


# preprocess.py
import os
import random
import re
import torch
import sentencepiece as spm
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from sklearn.model_selection import train_test_split

TAG_RE = re.compile(r'<[^>]+>')
def remove_html(text):
    return TAG_RE.sub('', text)

def train_tokenizer(de_file, en_file, vocab_size=32000):
    spm.SentencePieceTrainer.train(
        input=[de_file, en_file],
        model_prefix='translation_tokenizer',
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='[PAD]', unk_piece='[UNK]', bos_piece='[BOS]', eos_piece='[EOS]',
        user_defined_symbols=['[DE]', '[EN]']  # ✅ THE CRITICAL LINE
    )

def train_tokenizer2(input="tokenizer_input.txt", model_prefix="translation_tokenizer", vocab_size=32000):
    spm.SentencePieceTrainer.train(
        input=input,
        model_prefix=model_prefix,
        vocab_size=32000,
        user_defined_symbols=["[DE]", "[EN]"], # "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='[PAD]', unk_piece='[UNK]', bos_piece='[BOS]', eos_piece='[EOS]',
        character_coverage=1.0,
        model_type='bpe'
    )

def load_tokenizer(model_path='translation_tokenizer.model'):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    return tokenizer

def split_parallel_corpus_chunked(de_file, en_file, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, lim=None):
    os.makedirs(output_dir, exist_ok=True)
    pairs = []
    with open(de_file, "r", encoding="utf-8") as de_f, open(en_file, "r", encoding="utf-8") as en_f:
        for i, (de_line, en_line) in enumerate(zip(de_f, en_f)):
            if lim and i >= lim:
                break
            pairs.append((de_line.strip(), en_line.strip()))

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

    for split, data in splits.items():
        with open(os.path.join(output_dir, f"{split}.de"), "w", encoding="utf-8") as de_out, \
             open(os.path.join(output_dir, f"{split}.en"), "w", encoding="utf-8") as en_out:
            for de_line, en_line in data:
                de_out.write(de_line + "\n")
                en_out.write(en_line + "\n")

def collate_fn(batch, pad_id=0, tokenizer=None):
    assert tokenizer is not None
    # de_token_id = tokenizer.piece_to_id("[DE]")
    # en_token_id = tokenizer.piece_to_id("[EN]")
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    vocab_size = tokenizer.get_piece_size()
    print("Tokenizer vocab size:", vocab_size)  # should be ≤ 32000
    print("Special tokens:")
    for tok in ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[DE]', '[EN]']:
        print(f"  {tok} ID: {tokenizer.piece_to_id(tok)}")

    inputs, decoder_inputs, label_outputs = [], [], []
    for src, tgt in batch:
        # Ensure source tokens are within vocabulary range
        # Add [DE] token to source and EOS token
        # src_tensor = torch.cat([torch.tensor([de_token_id]), src, torch.tensor([eos_id])])
        # src = torch.clamp(src, 0, vocab_size - 1)
        src_tensor = torch.cat([src, torch.tensor([eos_id])])

        # Add [EN] token, BOS token, and EOS token to target
        # tgt_tensor = torch.cat([torch.tensor([en_token_id, bos_id]), tgt, torch.tensor([eos_id])])
        # tgt = torch.clamp(tgt, 0, vocab_size - 1)
        tgt_tensor = torch.cat([torch.tensor([bos_id]), tgt, torch.tensor([eos_id])])

        # Create decoder input (shifted right) and label output
        decoder_input = tgt_tensor[:-1]
        label_output = tgt_tensor[1:]
        label_output[0] = -100  # mask [EN] from loss

        src_tensor = torch.clamp(src_tensor, 0, vocab_size - 1)
        decoder_input = torch.clamp(decoder_input, 0, vocab_size - 1)
        label_output = torch.clamp(label_output, 0, vocab_size - 1)

        inputs.append(src_tensor)
        decoder_inputs.append(decoder_input)
        label_outputs.append(label_output)       

        # Ensure same length
        min_len = min(len(decoder_input), len(label_output))
        decoder_inputs.append(decoder_input[:min_len])
        label_outputs.append(label_output[:min_len])
        inputs.append(src_tensor)

    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)
    padded_labels = pad_sequence(label_outputs, batch_first=True, padding_value=-100)
    attention_mask = (padded_inputs != pad_id)

    return {
        "input_ids": padded_inputs,
        "target_input": padded_decoder_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

# class TranslationIterableDataset(IterableDataset):
#     def __init__(self, de_file, en_file, tokenizer, lim=None, max_length=128):
#         self.de_file = de_file
#         self.en_file = en_file
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.lim = lim

#     def __iter__(self):
#         with open(self.de_file, 'r', encoding='utf-8') as de_f, open(self.en_file, 'r', encoding='utf-8') as en_f:
#             for i, (de_line, en_line) in enumerate(zip(de_f, en_f)):
#                 if self.lim and i >= self.lim:
#                     break
#                 de_text = remove_html(de_line.strip())
#                 en_text = remove_html(en_line.strip())
#                 src = self.tokenizer.encode(de_text)[:self.max_length]
#                 tgt = self.tokenizer.encode(en_text)[:self.max_length]
#                 yield torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, de_list, en_list, tokenizer, max_length=128, lim=None): # de_file, en_file,
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # with open(de_file, 'r', encoding='utf-8') as f_de, open(en_file, 'r', encoding='utf-8') as f_en:
        #     for i, (de_line, en_line) in enumerate(zip(f_de, f_en)):
        #         if lim is not None and i >= lim:
        #             break
        de_text = [remove_html(word.strip()) for word in de_list]
        en_text = [remove_html(word.strip()) for word in en_list]
        self.samples = list(zip(de_text, en_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        de_text, en_text = self.samples[idx]
        source = f"[DE] {de_text}"
        target = f"[EN] {en_text}"
        src = [self.tokenizer.piece_to_id("[DE]")] + self.tokenizer.encode(de_text) + [self.tokenizer.eos_id()]
        tgt = [self.tokenizer.piece_to_id("[EN]"), self.tokenizer.bos_id()] + self.tokenizer.encode(en_text) + [self.tokenizer.eos_id()]
        src = self.tokenizer.encode(source)[:self.max_length]
        tgt = self.tokenizer.encode(target)[:self.max_length]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def get_dataloader_from_files(de_file, en_file, tokenizer, batch_size, pad_id=0, lim=None):
    dataset = TranslationDataset(de_file, en_file, tokenizer, lim=lim)
    collate = partial(collate_fn, pad_id=pad_id, tokenizer=tokenizer)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate,
        num_workers=2,  # Use 4 worker processes
        prefetch_factor=2,  # Prefetch 2 batches per worker
        pin_memory=True  # Use pinned memory for faster GPU transfer
    )

    # Wrap in datasets and dataloaders
def make_loader(de, en, batch_size_, tokenizer):
        ds = TranslationDataset(de, en, tokenizer)
        collate = partial(collate_fn, pad_id=tokenizer.pad_id(), tokenizer=tokenizer)
        return DataLoader(ds, batch_size=batch_size_, collate_fn=collate, num_workers=2)

def get_dataloaders3(de_file, en_file, tokenizer, batch_size):
    with open(de_file, 'r', encoding='utf-8') as f:
        de_lines = [line.strip() for line in f]
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]

    # Shuffle and split
    de_train, de_temp, en_train, en_temp = train_test_split(de_lines, en_lines, test_size=0.2, random_state=42)
    de_val, de_test, en_val, en_test = train_test_split(de_temp, en_temp, test_size=0.5, random_state=42)

    return make_loader(de_train, en_train, batch_size, tokenizer), make_loader(de_val, en_val, batch_size, tokenizer), make_loader(de_test, en_test, batch_size, tokenizer)

def get_dataloaders4(de, en, tokenizer, batch_size):
    return make_loader(de, en, batch_size, tokenizer)

# if __name__ == "__main__":
#     from preprocess import load_tokenizer, get_dataloader_from_files

#     print("Loading tokenizer...")
#     tokenizer = load_tokenizer()

#     print("Creating dataloaders...")
#     # train_loader = get_dataloader_from_files("de-en/split/train.de", "de-en/split/train.en", tokenizer, batch_size=32)
    

#     print("\nTraining set sample:")
#     for batch in train_loader:
#         print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Decoder Input (EN side):", tokenizer.decode(batch['target_input'][0].tolist()))
#         label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
#         print("Label Target (used for loss):", tokenizer.decode(label_ids))
#         break

#     print("\nValidation set sample:")
#     for batch in val_loader:
#         print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Decoder Input (EN side):", tokenizer.decode(batch['target_input'][0].tolist()))
#         label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
#         print("Label Target (used for loss):", tokenizer.decode(label_ids))
#         break

#     print("\nTest set sample:")
#     for batch in test_loader:
#         print("Input (DE):", tokenizer.decode(batch['input_ids'][0].tolist()))
#         print("Decoder Input (EN side):", tokenizer.decode(batch['target_input'][0].tolist()))
#         label_ids = [id for id in batch['labels'][0].tolist() if id != -100]
#         print("Label Target (used for loss):", tokenizer.decode(label_ids))
#         break
