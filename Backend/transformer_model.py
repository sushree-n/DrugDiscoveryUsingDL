import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))
        return x

class modelTransformer_smi(nn.Module):
    def __init__(self, args):
        super(modelTransformer_smi, self).__init__()
        vocab_size = len(args['vocab'])
        self.encoder = TransformerEncoder(
            vocab_size,
            args['hidden_dim'],
            args['num_heads'],
            args['num_layer'],
            args['dropout']
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(args['hidden_dim'], args['output_dim']),
            nn.ReLU(),
            nn.Linear(args['output_dim'], args['n_output'])
        )

    def forward(self, x, mask):
        encoded = self.encoder(x, mask).permute(0, 2, 1)
        pooled = self.pooling(encoded).squeeze(-1)
        return self.fc(pooled)

def getInput_mask(x):
    return (x != 0).long()
