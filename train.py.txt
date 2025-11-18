# src/train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

from data_generation import generate_multivariate_series
from preprocessing import scale_split
from model_transformer import TransformerForecast
from model_baseline import LSTMForecast

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import joblib

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def to_loader(windows, batch_size=64, shuffle=True):
    X_enc, X_dec_in, Y = windows
    X = torch.tensor(X_enc)
    D = torch.tensor(X_dec_in)
    Y = torch.tensor(Y)
    ds = TensorDataset(X, D, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=5):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best = None
    best_loss = float('inf')
    no_imp = 0
    history = {'train':[], 'val':[]}
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X, D, Y in train_loader:
            X = X.to(DEVICE); D = D.to(DEVICE); Y = Y.to(DEVICE)
            opt.zero_grad()
            out = model(X, D)
            loss = criterion(out, Y)
            loss.backward()
            opt.step()
            train_loss += loss.item()*X.size(0)
        train_loss /= len(train_loader.dataset)
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, D, Y in val_loader:
                X = X.to(DEVICE); D = D.to(DEVICE); Y = Y.to(DEVICE)
                out = model(X, D)
                val_loss += nn.MSELoss()(out, Y).item()*X.size(0)
        val_loss /= len(val_loader.dataset)
        history['train'].append(train_loss); history['val'].append(val_loss)
        print(f'Epoch {ep} train_loss={train_loss:.6f} val_loss={val_loss:.6f}')
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best = {k:v.cpu() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp +=1
            if no_imp >= patience:
                print('Early stopping')
                break
    if best is not None:
        model.load_state_dict(best)
    return model, history

def evaluate_autoregressive(model, dataset, scaler):
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for i in range(len(dataset[0])):
            enc = dataset[0][i:i+1]
            enc_t = torch.tensor(enc).to(DEVICE)
            if hasattr(model, 'predict_autoregressive'):
                out = model.predict_autoregressive(enc_t, pred_len=dataset[2].shape[1])
            else:
                out = model(enc_t, torch.tensor(dataset[1][i:i+1]).to(DEVICE))
            preds.append(out.cpu().numpy()[0])
            targs.append(dataset[2][i])
    preds = np.array(preds); targs = np.array(targs)
    B,H,F = preds.shape
    preds_rs = scaler.inverse_transform(preds.reshape(-1,F)).reshape(B,H,F)
    targs_rs = scaler.inverse_transform(targs.reshape(-1,F)).reshape(B,H,F)
    return preds_rs, targs_rs

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)

    df = generate_multivariate_series(T=3000, n_features=5)
    df.to_csv('data/generated_series.csv')
    print('Data generated', df.shape)

    enc_len=120; dec_len=24
    train_w, val_w, test_w, scaler = scale_split(df.values, encoder_len=enc_len, decoder_len=dec_len)
    train_loader = to_loader(train_w, batch_size=64, shuffle=True)
    val_loader = to_loader(val_w, batch_size=64, shuffle=False)

    n_features = df.shape[1]

    # Transformer
    transformer = TransformerForecast(n_features=n_features, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer, hist_t = train_model(transformer, train_loader, val_loader, epochs=25, lr=1e-3, patience=6)
    torch.save(transformer.state_dict(), 'outputs/models/transformer.pt')
    preds_t, targs_t = evaluate_autoregressive(transformer, test_w, scaler)

    # LSTM baseline
    lstm = LSTMForecast(n_features=n_features, hidden_size=128, num_layers=2)
    lstm, hist_l = train_model(lstm, train_loader, val_loader, epochs=25, lr=1e-3, patience=6)
    torch.save(lstm.state_dict(), 'outputs/models/lstm.pt')
    preds_l, targs_l = evaluate_autoregressive(lstm, test_w, scaler)

    # metrics for primary feature f1
    def rmse_local(a,b): return math.sqrt(mean_squared_error(a,b))

    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    preds_t_r = preds_t.reshape(-1, n_features)[:,0]
    targs_t_r = targs_t.reshape(-1, n_features)[:,0]
    preds_l_r = preds_l.reshape(-1, n_features)[:,0]
    targs_l_r = targs_l.reshape(-1, n_features)[:,0]

    print('Transformer f1 RMSE:', math.sqrt(mean_squared_error(targs_t_r, preds_t_r)), 
          'MAE:', mean_absolute_error(targs_t_r, preds_t_r),
          'MAPE:', mape(targs_t_r, preds_t_r))
    print('LSTM f1 RMSE:', math.sqrt(mean_squared_error(targs_l_r, preds_l_r)), 
          'MAE:', mean_absolute_error(targs_l_r, preds_l_r),
          'MAPE:', mape(targs_l_r, preds_l_r))

    # simple plot example
    plt.figure(figsize=(10,5))
    plt.plot(range(-enc_len,0), scaler.inverse_transform(test_w[0][0])[:,0], label='history')
    plt.plot(range(0,dec_len), targs_t[0][:,0], label='target')
    plt.plot(range(0,dec_len), preds_t[0][:,0], label='pred_transformer')
    plt.plot(range(0,dec_len), preds_l[0][:,0], label='pred_lstm')
    plt.legend(); plt.title('Example predictions f1'); plt.savefig('outputs/plots/example_pred_f1.png')

    # save metrics
    with open('outputs/reports/metrics.txt','w') as fh:
        fh.write('Transformer f1 RMSE: {:.4f} MAE: {:.4f} MAPE: {:.2f}%\\n'.format(
            math.sqrt(mean_squared_error(targs_t_r, preds_t_r)), mean_absolute_error(targs_t_r, preds_t_r), mape(targs_t_r, preds_t_r)))
        fh.write('LSTM f1 RMSE: {:.4f} MAE: {:.4f} MAPE: {:.2f}%\\n'.format(
            math.sqrt(mean_squared_error(targs_l_r, preds_l_r)), mean_absolute_error(targs_l_r, preds_l_r), mape(targs_l_r, preds_l_r)))
    print('Saved outputs in outputs/')

if _name_ == '_main_':
    main()