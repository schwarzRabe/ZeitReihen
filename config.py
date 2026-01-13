from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    # --- Telegram ---
    bot_token: str

    # --- Data ---
    history_years: int = 2
    price_column: str = "Close"

    # --- Forecasting / evaluation ---
    horizon_days: int = 30           # 30 календарных дней для пользователя
    test_size_days: int = 60         # holdout по времени
    metric: str = "RMSE"             # "RMSE" or "MAPE"

    # Baseline 
    baseline_model_name: str = "Naive"

    # Strategy 
    extrema_use_threshold: bool = True
    extrema_threshold_pct: float = 0.005  # 0.5%: 

    # EMA 
    ema_fast: int = 5
    ema_slow: int = 20

    # Trading
    allow_fractional: bool = True
    commission_rate: float = 0.0

    # --- Logging ---
    log_path: Path = Path("logs.csv")

    # --- Models params ---
    n_lags: int = 20
    arima_order: tuple[int, int, int] = (1, 1, 1)

    # fan chart (ARIMA intervals)
    fan_alpha: float = 0.05  # ~95% интервал

    # Visual-only ARIMA scenarios 
    show_simulated_paths: bool = True
    simulated_paths_n: int = 3
    simulated_paths_seed: int = 42

    # Boosting
    hgb_max_depth: int = 6
    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.05

    # GRU
    nn_lookback: int = 30
    nn_hidden: int = 32
    nn_epochs: int = 60
    nn_batch_size: int = 32
    nn_val_ratio: float = 0.2
    nn_patience: int = 8
    nn_lr: float = 1e-3
    nn_seed: int = 42


def load_settings() -> Settings:
    """Читаем токен из окружения"""
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        token = "DUMMY_TOKEN_FOR_CORE_ONLY"
    return Settings(bot_token=token)
