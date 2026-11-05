import torch
import torch.nn as nn
from motor_matematico import TransformerBlock, DecoderBlock
from hiperparametros import DEVICE


class LabTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        
        self.encoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src, pad_token_id):
        return (src != pad_token_id).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # Máscara Causal (Look Ahead)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len).to(DEVICE)
        return trg_mask

    def forward(self, src, trg, pad_token_id):
        src_mask = self.make_src_mask(src, pad_token_id)
        trg_mask = self.make_trg_mask(trg)
        
        # Encoder
        enc_out = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
            
        # Decoder
        dec_out = self.embedding(trg) + self.pos_encoding[:, :trg.size(1), :]
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, trg_mask)
            
        return self.fc_out(dec_out)