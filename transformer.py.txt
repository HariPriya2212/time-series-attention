src/model_transformer.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=5000):
        super()._init_()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerForecast(nn.Module):
    def _init_(self, n_features, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super()._init_()
        self.n_features = n_features
        self.d_model = d_model
        self.enc_in = nn.Linear(n_features, d_model)
        self.dec_in = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, batch_first=True)
        self.out = nn.Linear(d_model, n_features)

    def forward(self, enc, dec_in):
        src = self.pos(self.enc_in(enc))
        tgt = self.pos(self.dec_in(dec_in))
        memory = self.transformer.encoder(src)
        out = self.transformer.decoder(tgt, memory)
        return self.out(out)

    def predict_autoregressive(self, enc, pred_len):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            enc = enc.to(device)
            memory = self.transformer.encoder(self.pos(self.enc_in(enc)))
            last = enc[:, -1:, :]
            dec_in = last
            outs = []
            for _ in range(pred_len):
                tgt = self.pos(self.dec_in(dec_in))
                out = self.transformer.decoder(tgt, memory)
                step = self.out(out[:, -1:, :])
                outs.append(step)
                dec_in = torch.cat([dec_in, step], dim=1)
            return torch.cat(outs, dim=1)