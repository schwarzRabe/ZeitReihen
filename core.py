from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import csv
import io
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from config import Settings
from models import (
    ForecastModel, ModelReport,
    NaiveModel, LagRidgeModel, BoostingModel, ArimaForecastModel, EtsForecastModel, GruModel,
    rmse, mape,
)


@dataclass
class Trade:
    date: pd.Timestamp
    action: str  # "BUY" / "SELL"
    price: float
    qty: float


@dataclass
class TradeSummary:
    mode: str  # "EXTREMA" / "BUY_HOLD"
    trades: List[Trade]
    profit: float
    cash_end: float


@dataclass
class ForecastResult:
    ticker: str
    current_price: float
    current_date: pd.Timestamp

    chosen_model: str
    metric_name: str
    metric_value: float
    baseline_metric: float
    improvement_ratio: float
    model_scores: str

    forecast_mean: pd.Series
    delta_abs: float
    delta_pct: float

    optimistic_last: Optional[float]
    pessimistic_last: Optional[float]

    trade_summary: TradeSummary

    plot_forecast_png: bytes
    plot_trades_png: bytes



def load_prices_yahoo(ticker: str, years: int, col: str) -> pd.Series:
    """Загрузка котировок из Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError(f"Нет данных по тикеру: {ticker}")

    # Нормализуем выбор колонки
    if col not in df.columns:
        col = "Close" if "Close" in df.columns else col
    if col not in df.columns:
        raise ValueError(f"Колонка {col} отсутствует в данных")

    s = df[col].dropna()

    # --- гарантируем, что на выходе строго 1D pd.Series ---
    # Иногда df[col] может оказаться DataFrame (например, при мульти-колонках),
    # тогда берём первый столбец.
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            raise ValueError(f"Колонка {col} отсутствует в данных")
        s = s.iloc[:, 0]
    elif not isinstance(s, pd.Series):
        s = pd.Series(s)

    # Приводим к числам (на случай object dtype)
    s = s.astype(float)

    # Нормализуем индекс дат
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    return s.sort_index()


def append_csv(path: str, row: Dict[str, Any]) -> None:
    """Логирование одной строкой в CSV."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# Metrics / model 

def split_ts(y: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    y = y.dropna()
    if len(y) <= test_size + 80:
        raise ValueError("Недостаточно истории для выбранного test_size")
    return y.iloc[:-test_size], y.iloc[-test_size:]


def metric_fn(name: str):
    name = name.upper().strip()
    return (mape, "MAPE") if name == "MAPE" else (rmse, "RMSE")


def evaluate_models(
    models: List[ForecastModel],
    y: pd.Series,
    test_size: int,
    metric_name: str,
) -> Tuple[List[ModelReport], str]:
    """
    Считает метрики моделей на time-series holdout (train -> test).
    Возвращает список отчётов (успешные модели) и строку ошибок для лога.
    """
    fn, mname = metric_fn(metric_name)
    train, test = split_ts(y, test_size)

    reports: List[ModelReport] = []
    fails: List[str] = []

    for m in models:
        try:
            m.fit(train)
            pred = m.predict(horizon=len(test), last_date=train.index[-1], align_index=test.index)
            reports.append(ModelReport(m.name, mname, float(fn(test, pred))))
        except Exception as e:
            fails.append(f"{getattr(m, 'name', m.__class__.__name__)}: {e}")

    if not reports:
        raise RuntimeError("Все модели упали: " + "; ".join(fails)[:1000])

    reports.sort(key=lambda r: r.metric_value)
    return reports, "; ".join(fails)[:1000]


# Стратегия

def extrema_signals(forecast: pd.Series, thr_pct: Optional[float]) -> List[Tuple[pd.Timestamp, str]]:
    """
    Локальные экстремумы на forecast(mean):
      - min -> BUY
      - max -> SELL
    Порог thr_pct (например 0.5%) отсекает микро-колебания.
    """
    f = forecast.dropna().astype(float)
    if len(f) < 3:
        return []

    y = f.to_numpy(float)
    idx = list(f.index)

    mins, maxs = [], []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            mins.append(i)
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            maxs.append(i)

    if thr_pct:
        def pass_thr(i: int) -> bool:
            ref = (y[i - 1] + y[i + 1]) / 2.0
            return abs(y[i] - ref) / max(abs(ref), 1e-9) >= thr_pct
        mins = [i for i in mins if pass_thr(i)]
        maxs = [i for i in maxs if pass_thr(i)]

    sigs = [(pd.Timestamp(idx[i]), "BUY") for i in mins] + [(pd.Timestamp(idx[i]), "SELL") for i in maxs]
    sigs.sort(key=lambda x: x[0])
    return sigs


def simulate_trades(
    forecast: pd.Series,
    signals: List[Tuple[pd.Timestamp, str]],
    *,
    current_price: float,
    current_date: pd.Timestamp,
    amount: float,
    allow_fractional: bool,
    fee_rate: float,
) -> TradeSummary:
    """
    - если signals пусты -> Buy&Hold (BUY сегодня по current_price, SELL в конце прогноза)
    - иначе -> торговля по сигналам (1 позиция), с закрытием позиции в конце горизонта
    """
    f = forecast.dropna().astype(float)
    if len(f) < 2:
        return TradeSummary("BUY_HOLD", [], 0.0, float(amount))

    end_date = pd.Timestamp(f.index[-1])
    end_price = float(f.iloc[-1])

    cash = float(amount)
    qty = 0.0
    trades: List[Trade] = []

    def buy(date: pd.Timestamp, price: float) -> None:
        nonlocal cash, qty
        if qty > 0 or cash <= 0:
            return
        q = cash / price
        if not allow_fractional:
            q = float(int(q))
        if q <= 0:
            return
        cost = q * price
        fee = cost * fee_rate
        cash -= cost + fee
        qty = q
        trades.append(Trade(pd.Timestamp(date), "BUY", float(price), float(q)))

    def sell(date: pd.Timestamp, price: float) -> None:
        nonlocal cash, qty
        if qty <= 0:
            return
        revenue = qty * price
        fee = revenue * fee_rate
        cash += revenue - fee
        trades.append(Trade(pd.Timestamp(date), "SELL", float(price), float(qty)))
        qty = 0.0

    # Buy&Hold как fallback (вход всегда "сегодня") 
    if not signals:
        buy(pd.Timestamp(current_date), float(current_price))
        sell(end_date, end_price)
        profit = cash - float(amount)
        return TradeSummary("BUY_HOLD", trades, float(profit), float(cash))

    # Торговля по сигналам 
    for d, act in signals:
        if d not in f.index:
            continue
        price = float(f.loc[d])
        if act == "BUY":
            buy(d, price)
        elif act == "SELL":
            sell(d, price)

    # Закрываем позицию в конце горизонта, если осталась открытой
    if qty > 0:
        sell(end_date, end_price)

    # Если покупок не было — fallback Buy&Hold (по ТЗ и UX)
    if not any(t.action == "BUY" for t in trades):
        return simulate_trades(
            f,
            [],
            current_price=current_price,
            current_date=current_date,
            amount=amount,
            allow_fractional=allow_fractional,
            fee_rate=fee_rate,
        )

    profit = cash - float(amount)
    return TradeSummary("EXTREMA", trades, float(profit), float(cash))


def _format_dates(ax) -> None:
    locator = mdates.AutoDateLocator(maxticks=6)
    formatter = mdates.DateFormatter("%d.%m")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_horizontalalignment("right")


def plot_forecast(history: pd.Series, mean: pd.Series, conf_int: Optional[pd.DataFrame], ticker: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.index, history.values, label="История")
    ax.plot(mean.index, mean.values, label="Прогноз (средний)")

    if conf_int is not None:
        ax.fill_between(mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.25, label="Интервал прогноза")

    ax.axvline(history.index[-1], linestyle="--", color="gray", alpha=0.7, label="Сегодня")
    _format_dates(ax)

    ax.set_title(f"{ticker}: прогноз на 30 дней")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_signals(mean: pd.Series, trades: List[Trade], ema_fast: int, ema_slow: int) -> bytes:
    """График 2: сигналы + EMA (только визуально)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(mean.index, mean.values, label="Прогноз (средний)")
    ema_f = mean.ewm(span=ema_fast, adjust=False).mean()
    ema_s = mean.ewm(span=ema_slow, adjust=False).mean()
    ax.plot(ema_f.index, ema_f.values, label=f"EMA({ema_fast})")
    ax.plot(ema_s.index, ema_s.values, label=f"EMA({ema_slow})")

    buys = [(t.date, t.price) for t in trades if t.action == "BUY"]
    sells = [(t.date, t.price) for t in trades if t.action == "SELL"]
    if buys:
        ax.scatter([d for d, _ in buys], [p for _, p in buys], marker="^", label="BUY")
    if sells:
        ax.scatter([d for d, _ in sells], [p for _, p in sells], marker="v", label="SELL")

    _format_dates(ax)
    ax.set_title("Сигналы и тренд (визуально)")
    ax.legend()

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


class ForecastPipeline:
    """
    Пайплайн:
    1) загрузка данных
    2) оценка моделей на тесте
    3) выбор победителя (Naive только baseline, не победитель)
    4) прогноз на 30 дней выбранной моделью
    5) стратегия по forecast(mean) + fallback Buy&Hold
    6) графики и лог
    """

    def __init__(self, s: Settings) -> None:
        self.s = s

        # Модели по ТЗ: минимум 3 класса
        self.models: List[ForecastModel] = [
            NaiveModel(),  # baseline
            LagRidgeModel(n_lags=s.n_lags),  # ML (линейная)
            BoostingModel(n_lags=s.n_lags, max_depth=s.hgb_max_depth, max_iter=s.hgb_max_iter, lr=s.hgb_learning_rate),  # ML (нелинейная)
            ArimaForecastModel(order=s.arima_order),  # статистика
            EtsForecastModel(trend="add"),  # статистика (ETS)
            GruModel(
                lookback=s.nn_lookback, hidden=s.nn_hidden, epochs=s.nn_epochs,
                batch_size=s.nn_batch_size, val_ratio=s.nn_val_ratio, patience=s.nn_patience,
                lr=s.nn_lr, seed=s.nn_seed,
            ),  # нейросеть (если torch установлен)
        ]

    def run(self, user_id: int, ticker: str, amount: float) -> ForecastResult:
        y = load_prices_yahoo(ticker, self.s.history_years, self.s.price_column)
        current_price = float(y.iloc[-1])
        current_date = pd.Timestamp(y.index[-1])

        reports, fails = evaluate_models(self.models, y, self.s.test_size_days, self.s.metric)
        metric_name = reports[0].metric_name

        # строка сравнения для пользователя/отчёта
        model_scores = ", ".join(f"{r.model_name}={r.metric_value:.4f}" for r in reports)

        baseline = next((r for r in reports if r.model_name == self.s.baseline_model_name), None)
        if baseline is None:
            raise RuntimeError("Baseline Naive не посчитался.")
        baseline_metric = float(baseline.metric_value)

        # победитель среди non-baseline
        non_base = [r for r in reports if r.model_name != self.s.baseline_model_name]
        if not non_base:
            raise RuntimeError("Ни одна не-baseline модель не отработала.")
        chosen = min(non_base, key=lambda r: r.metric_value)

        chosen_name = chosen.model_name
        chosen_metric = float(chosen.metric_value)
        improvement = (baseline_metric - chosen_metric) / (baseline_metric + 1e-12)

        model_map = {m.name: m for m in self.models}
        model = model_map[chosen_name]

        # прогноз победителем
        model.fit(y)
        mean = model.predict(self.s.horizon_days, last_date=y.index[-1], align_index=None).astype(float).clip(lower=0.01)

        delta_abs = float(mean.iloc[-1] - current_price)
        delta_pct = float(delta_abs / current_price * 100.0) if current_price else 0.0

        # интервалы для фан-чарта (если доступны)
        conf_int: Optional[pd.DataFrame] = None
        optimistic_last = pessimistic_last = None
        if hasattr(model, "predict_interval"):
            try:
                lower, upper = model.predict_interval(self.s.horizon_days, last_date=y.index[-1], alpha=self.s.fan_alpha)
                conf_int = pd.concat([lower, upper], axis=1)
                pessimistic_last = float(lower.iloc[-1])
                optimistic_last = float(upper.iloc[-1])
            except Exception:
                conf_int = None

        # стратегия по ТЗ: экстремумы на forecast(mean) + fallback
        thr = self.s.extrema_threshold_pct if self.s.extrema_use_threshold else None
        sigs = extrema_signals(mean, thr)
        summary = simulate_trades(
            mean, sigs,
            current_price=current_price,
            current_date=current_date,
            amount=float(amount),
            allow_fractional=self.s.allow_fractional,
            fee_rate=self.s.commission_rate,
        )

        p1 = plot_forecast(y, mean, conf_int, ticker)
        p2 = plot_signals(mean, summary.trades, self.s.ema_fast, self.s.ema_slow)

        append_csv(str(self.s.log_path), {
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "user_id": int(user_id),
            "ticker": ticker,
            "amount": float(amount),

            "metric_name": metric_name,
            "model_scores": model_scores,

            "chosen_model": chosen_name,
            "metric_value": chosen_metric,
            "baseline_metric": baseline_metric,
            "improvement_ratio": improvement,

            "current_price": current_price,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,

            "strategy_mode": summary.mode,
            "profit": summary.profit,
            "cash_end": summary.cash_end,
            "trades_count": len(summary.trades),

            "optimistic_last": optimistic_last if optimistic_last is not None else "",
            "pessimistic_last": pessimistic_last if pessimistic_last is not None else "",
            "failed_models": fails,
        })

        return ForecastResult(
            ticker=ticker,
            current_price=current_price,
            current_date=current_date,

            chosen_model=chosen_name,
            metric_name=metric_name,
            metric_value=chosen_metric,
            baseline_metric=baseline_metric,
            improvement_ratio=improvement,
            model_scores=model_scores,

            forecast_mean=mean,
            delta_abs=delta_abs,
            delta_pct=delta_pct,

            optimistic_last=optimistic_last,
            pessimistic_last=pessimistic_last,

            trade_summary=summary,

            plot_forecast_png=p1,
            plot_trades_png=p2,
        )
