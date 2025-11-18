 src/preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler

class WindowGenerator:
    def _init_(self, arr_values, encoder_len=120, decoder_len=24):
        self.arr = arr_values.astype(np.float32)
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len

    def split_series(self, train_frac=0.7, val_frac=0.15):
        T = self.arr.shape[0]
        train_end = int(T * train_frac)
        val_end = int(T * (train_frac + val_frac))
        train = self.arr[:train_end]
        val = self.arr[train_end:val_end]
        test = self.arr[val_end:]
        return train, val, test

    def create_windows(self, arr, step=1):
        X_enc, X_dec_in, Y = [], [], []
        max_start = arr.shape[0] - (self.encoder_len + self.decoder_len) + 1
        for s in range(0, max_start, step):
            enc = arr[s: s + self.encoder_len]
            dec_t = arr[s + self.encoder_len: s + self.encoder_len + self.decoder_len]
            last = enc[-1:]
            dec_in = np.vstack([last for _ in range(self.decoder_len)])
            X_enc.append(enc)
            X_dec_in.append(dec_in)
            Y.append(dec_t)
        return (np.array(X_enc), np.array(X_dec_in), np.array(Y))

def scale_split(df_values, encoder_len=120, decoder_len=24):
    wg = WindowGenerator(df_values, encoder_len, decoder_len)
    train, val, test = wg.split_series()
    scaler = StandardScaler()
    scaler.fit(train)
    train_s = scaler.transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    train_windows = wg.create_windows(train_s)
    val_windows = wg.create_windows(val_s)
    test_windows = wg.create_windows(test_s)
    return train_windows, val_windows, test_windows,scaler