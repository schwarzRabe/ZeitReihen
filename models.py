from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, List

import numpy as np
import pandas as pd


class ForecastModel(Protocol):
    name: str
    def fit(self, y: pd.Series) -> None: ...
    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series: ...


@dataclass
class ModelReport:
    model_name: str
    metric_name: str
    metric_value: float


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    a = y_true.to_numpy(dtype=float)
    b = y_pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    a = y_true.to_numpy(dtype=float)
    b = y_pred.to_numpy(dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((a - b) / (a + eps))) * 100.0)


def future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """30 календарных дней = 30 точек."""
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq="D")


def make_lag_xy(arr: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(n_lags, len(arr)):
        X.append(arr[i - n_lags:i])
        y.append(arr[i])
    return np.asarray(X, float), np.asarray(y, float)


def _rollout_predict(model, last_window: np.ndarray, horizon: int) -> np.ndarray:
    """Общий autoregressive rollout для моделей по лагам."""
    win = last_window.copy()
    preds: List[float] = []
    for _ in range(horizon):
        yhat = float(model.predict(win.reshape(1, -1))[0])
        preds.append(yhat)
        win = np.roll(win, -1)
        win[-1] = yhat
    return np.asarray(preds, float)


class NaiveModel:
    name = "Naive"
    def __init__(self) -> None:
        self._last: Optional[float] = None

    def fit(self, y: pd.Series) -> None:
        y = y.dropna()
        if len(y) < 5:
            raise ValueError("Недостаточно данных для Naive")
        self._last = float(y.iloc[-1])

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if self._last is None:
            raise RuntimeError("NaiveModel не обучена")
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(np.full(horizon, self._last, dtype=float), index=idx, name="yhat")


class LagRidgeModel:
    name = "LagRidge"
    def __init__(self, n_lags: int = 20) -> None:
        self.n_lags = n_lags
        self._model = None
        self._last_window: Optional[np.ndarray] = None

    def fit(self, y: pd.Series) -> None:
        from sklearn.linear_model import Ridge
        y = y.dropna()
        arr = y.to_numpy(float)
        if len(arr) <= self.n_lags + 20:
            raise ValueError("Недостаточно данных для LagRidge")
        X, t = make_lag_xy(arr, self.n_lags)
        self._model = Ridge(alpha=1.0).fit(X, t)
        self._last_window = arr[-self.n_lags:].copy()

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if self._model is None or self._last_window is None:
            raise RuntimeError("LagRidgeModel не обучена")
        preds = _rollout_predict(self._model, self._last_window, horizon)
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(preds, index=idx, name="yhat")


class BoostingModel:
    """XGBoost (если доступен), иначе HistGradientBoosting."""
    def __init__(self, n_lags: int, max_depth: int, max_iter: int, lr: float) -> None:
        self.n_lags = n_lags
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.lr = lr
        self._model = None
        self._last_window: Optional[np.ndarray] = None
        self._name = "Boosting"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, y: pd.Series) -> None:
        y = y.dropna()
        arr = y.to_numpy(float)
        if len(arr) <= self.n_lags + 40:
            raise ValueError("Недостаточно данных для Boosting")
        X, t = make_lag_xy(arr, self.n_lags)

        try:
            from xgboost import XGBRegressor  # type: ignore
            self._model = XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=self.max_depth,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
            ).fit(X, t)
            self._name = "XGBoost"
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = HistGradientBoostingRegressor(
                max_depth=self.max_depth,
                max_iter=self.max_iter,
                learning_rate=self.lr,
                random_state=42,
            ).fit(X, t)
            self._name = "HistGB"

        self._last_window = arr[-self.n_lags:].copy()

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if self._model is None or self._last_window is None:
            raise RuntimeError("BoostingModel не обучена")
        preds = _rollout_predict(self._model, self._last_window, horizon)
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(preds, index=idx, name="yhat")


class ArimaForecastModel:
    name = "ARIMA"
    def __init__(self, order: Tuple[int, int, int]) -> None:
        self.order = order
        self._fit_res = None

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA
        y = y.dropna()
        if len(y) < 80:
            raise ValueError("Недостаточно данных для ARIMA")
        self._fit_res = ARIMA(y, order=self.order).fit()

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if self._fit_res is None:
            raise RuntimeError("ARIMA не обучена")
        mean = self._fit_res.get_forecast(steps=horizon).predicted_mean.to_numpy(float)
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(mean, index=idx, name="yhat")

    def predict_interval(self, horizon: int, last_date: pd.Timestamp, alpha: float) -> tuple[pd.Series, pd.Series]:
        """Интервал неопределённости """
        if self._fit_res is None:
            raise RuntimeError("ARIMA не обучена")
        res = self._fit_res.get_forecast(steps=horizon)
        ci = res.conf_int(alpha=alpha)
        idx = future_index(last_date, horizon)
        lower = pd.Series(ci.iloc[:, 0].to_numpy(float), index=idx, name="lower")
        upper = pd.Series(ci.iloc[:, 1].to_numpy(float), index=idx, name="upper")
        return lower, upper

    def simulate_paths(self, horizon: int, last_date: pd.Timestamp, n_paths: int, seed: int) -> List[pd.Series]:
        """Сценарные траектории """
        if self._fit_res is None:
            raise RuntimeError("ARIMA не обучена")
        idx = future_index(last_date, horizon)
        sim = self._fit_res.simulate(
            nsimulations=horizon,
            repetitions=n_paths,
            anchor="end",
            random_state=seed,
        )
        arr = np.asarray(sim, float)
        if arr.ndim == 1:
            return [pd.Series(arr, index=idx, name="sim0")]
        reps = min(n_paths, arr.shape[1])
        return [pd.Series(arr[:, j], index=idx, name=f"sim{j}") for j in range(reps)]

class EtsForecastModel:

    name = "ETS"

    def __init__(self, trend: str = "add") -> None:
        self.trend = trend
        self._fit_res = None
        self._sigma = None  # оценка std остатков для интервала

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = y.dropna()
        if len(y) < 80:
            raise ValueError("Недостаточно данных для ETS")

        # сезонность не задаём: универсально для разных тикеров и не усложняет
        model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=None,
            initialization_method="estimated",
        )
        self._fit_res = model.fit(optimized=True)

        # оценка шума для интервала (простая и прозрачная)
        resid = getattr(self._fit_res, "resid", None)
        if resid is not None:
            r = np.asarray(resid, float)
            self._sigma = float(np.nanstd(r)) if np.isfinite(r).any() else None
        else:
            self._sigma = None

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if self._fit_res is None:
            raise RuntimeError("ETS не обучена")

        mean = np.asarray(self._fit_res.forecast(horizon), float)
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(mean, index=idx, name="yhat")

    def predict_interval(self, horizon: int, last_date: pd.Timestamp, alpha: float) -> tuple[pd.Series, pd.Series]:
        """
        Интервал для ETS: mean ± z * sigma.

        """
        if self._fit_res is None:
            raise RuntimeError("ETS не обучена")

        mean = np.asarray(self._fit_res.forecast(horizon), float)
        idx = future_index(last_date, horizon)

        # Если sigma не оценили — интервал не строим
        if self._sigma is None or not np.isfinite(self._sigma) or self._sigma <= 0:
            lower = pd.Series(mean, index=idx, name="lower")
            upper = pd.Series(mean, index=idx, name="upper")
            return lower, upper

        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alpha / 2)

        lower = pd.Series(mean - z * self._sigma, index=idx, name="lower")
        upper = pd.Series(mean + z * self._sigma, index=idx, name="upper")
        return lower, upper


class GruModel:
    name = "GRU"

    def __init__(
        self,
        lookback: int,
        hidden: int,
        epochs: int,
        batch_size: int,
        val_ratio: float,
        patience: int,
        lr: float,
        seed: int,
    ) -> None:
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.patience = patience
        self.lr = lr
        self.seed = seed

        self._fitted = False
        self._model = None
        self._last_window = None
        self._mean = None
        self._std = None

    def fit(self, y: pd.Series) -> None:
        import torch
        import torch.nn as nn

        y = y.dropna()
        arr = y.to_numpy(float)
        if len(arr) <= self.lookback + 60:
            raise ValueError("Недостаточно данных для GRU")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._mean = float(arr.mean())
        self._std = float(arr.std() + 1e-9)
        z = (arr - self._mean) / self._std

        X, t = [], []
        for i in range(self.lookback, len(z)):
            X.append(z[i - self.lookback:i])
            t.append(z[i])
        X = np.asarray(X, dtype=np.float32)[:, :, None]
        t = np.asarray(t, dtype=np.float32)[:, None]

        n = len(X)
        n_val = max(1, int(n * self.val_ratio))
        n_tr = n - n_val
        X_tr, t_tr = X[:n_tr], t[:n_tr]
        X_val, t_val = X[n_tr:], t[n_tr:]

        device = torch.device("cpu")
        X_tr_t = torch.tensor(X_tr, device=device)
        t_tr_t = torch.tensor(t_tr, device=device)
        X_val_t = torch.tensor(X_val, device=device)
        t_val_t = torch.tensor(t_val, device=device)

        class Net(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        model = Net(self.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        bad = 0

        def batches(Xb, yb, bs):
            for i in range(0, len(Xb), bs):
                yield Xb[i:i + bs], yb[i:i + bs]

        for _ in range(self.epochs):
            model.train()
            for xb, yb in batches(X_tr_t, t_tr_t, self.batch_size):
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val_t), t_val_t).cpu().numpy())

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._last_window = z[-self.lookback:].astype(np.float32).copy()
        self._fitted = True

    def predict(self, horizon: int, last_date: pd.Timestamp, align_index: Optional[pd.Index] = None) -> pd.Series:
        if not self._fitted or self._model is None or self._last_window is None:
            raise RuntimeError("GRU не обучена")

        import torch
        device = torch.device("cpu")
        self._model.eval()

        win = self._last_window.copy()
        preds_z: List[float] = []
        for _ in range(horizon):
            x = torch.tensor(win[None, :, None], dtype=torch.float32, device=device)
            with torch.no_grad():
                zhat = float(self._model(x).cpu().numpy().ravel()[0])
            preds_z.append(zhat)
            win = np.roll(win, -1)
            win[-1] = zhat

        preds = np.asarray(preds_z, float) * float(self._std) + float(self._mean)
        idx = align_index if align_index is not None else future_index(last_date, horizon)
        return pd.Series(preds, index=idx, name="yhat")
