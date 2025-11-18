
# src/data_generation.py
import numpy as np
import pandas as pd

def generate_multivariate_series(T=3000, n_features=5, seed=2025):
    """
    Generate synthetic multivariate series with trend, multiple seasonality,
    heteroskedastic noise and occasional spikes.
    """
    np.random.seed(seed)
    t = np.arange(T)
    trend = 0.0008 * (t ** 1.15)
    season1 = 2.0 * np.sin(2 * np.pi * t / 50.0)
    season2 = 0.8 * np.sin(2 * np.pi * t / 7.0 + 0.3)
    season3 = 0.4 * np.sin(2 * np.pi * t / 365.0 + 1.0)
    data = np.zeros((T, n_features))
    for i in range(n_features):
        amp = 1.0 + 0.3 * np.sin(0.2 * i + 0.1)
        coupling = 0.15 * i * np.sin(0.01 * t)
        noise_scale = 0.4 + 0.2 * np.abs(np.sin(2 * np.pi * t / (20 + 5 * i))) + 0.1 * trend
        feat = amp * (season1 + 0.5 * season2 + 0.2 * season3 + coupling) + trend * (1 + 0.05 * i)
        feat += noise_scale * np.random.normal(scale=1.0, size=T)
        spikes_idx = np.random.choice(T, size=max(1, T // 1200), replace=False)
        feat[spikes_idx] += np.random.choice([8, -6], size=spikes_idx.shape) * (0.5 + 0.5 * np.random.rand())
        data[:, i] = feat
    dates = pd.date_range(start='2000-01-01', periods=T, freq='D')
    df = pd.DataFrame(data, index=dates, columns=[f'f{i+1}' for i in range(n_features)])
    return df

if _name_ == '_main_':
    df = generate_multivariate_series()
    df.to_csv('data/generated_series.csv')
    print('Saved data/generated_series.csv', df.shape)

