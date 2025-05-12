import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        try:
            # Create target mask for decoder
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Create source padding mask
            src_padding_mask = (src == 0).to(src.device)  # True for padding tokens
            tgt_padding_mask = (tgt == 0).to(tgt.device)  # True for padding tokens
            
            # Embedding and positional encoding
            src = self.embedding(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            
            tgt = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            
            # Transformer forward pass
            output = self.transformer(
                src, 
                tgt,
                src_mask=None,  # No source mask needed
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            # Project to vocabulary size
            output = self.output_layer(output)
            
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - src: {src.shape}, tgt: {tgt.shape}")
            print(f"Device - src: {src.device}, tgt: {tgt.device}")
            raise
    
    def encode(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src)
    
    def decode(self, tgt, memory, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer.decoder(tgt, memory, tgt_mask)
        output = self.output_layer(output)
        return output 