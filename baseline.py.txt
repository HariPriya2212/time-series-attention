# src/model_baseline.py
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def _init_(self, n_features, hidden_size=128, num_layers=2, dropout=0.1):
        super()._init_()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, enc, dec_in=None, pred_len=None):
        out, (hn, cn) = self.lstm(enc)
        h, c = hn, cn
        batch = enc.size(0)
        last = enc[:, -1:, :]
        preds = []
        decoder_input = last
        for _ in range(pred_len):
            out_dec, (h, c) = self.lstm(decoder_input, (h, c))
            step = self.fc(out_dec[:, -1, :]).unsqueeze(1)
            preds.append(step)
            decoder_input = step
        return torch.cat(preds, dim=1)