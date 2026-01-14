from dotenv import load_dotenv
load_dotenv()

import asyncio
import re
import logging
from typing import List, Tuple, Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import BufferedInputFile

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage

from config import load_settings, TradingMode
from core import ForecastPipeline, TradeSummary, Trade
from exceptions import DataLoadError, InsufficientDataError, ModelTrainingError, InvalidTickerError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
MIN_AMOUNT = 1.0
MAX_AMOUNT = 1_000_000_000.0


class Form(StatesGroup):
    ticker = State()
    amount = State()


def kb(actions: List[Tuple[str, str]], one_row: bool = False) -> InlineKeyboardMarkup:
    """Create inline keyboard from actions list."""
    if one_row:
        buttons = [InlineKeyboardButton(text=txt, callback_data=data) for txt, data in actions]
        return InlineKeyboardMarkup(inline_keyboard=[buttons])
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=txt, callback_data=data)] for txt, data in actions
    ])


_KB_MAIN = kb([
    ("🔍 Новый прогноз", "act:new"),
    ("⭐ Популярные тикеры", "act:ticks"),
    ("🧹 Сброс", "act:reset"),
    ("🛑 Завершить", "act:exit"),
])

_KB_AFTER = kb([
    ("🔍 Новый прогноз", "act:new"),
    ("⭐ Популярные тикеры", "act:ticks"),
    ("🏠 В меню", "act:menu"),
    ("🛑 Завершить", "act:exit"),
])


def kb_tickers() -> InlineKeyboardMarkup:
    """Create keyboard with popular tickers."""
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for i, t in enumerate(EXAMPLE_TICKERS, 1):
        row.append(InlineKeyboardButton(text=t, callback_data=f"ticker:{t}"))
        if i % 4 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="act:menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


_KB_TICKERS = kb_tickers()


def validate_and_normalize_ticker(text: str) -> Optional[str]:
    """Validate and normalize ticker symbol. Returns None if invalid."""
    t = text.strip().upper()
    return t if TICKER_RE.match(t) else None


def parse_amount(text: str) -> float:
    """Parse amount from text input."""
    s = text.strip().replace(" ", "").replace("_", "").replace(",", ".")
    return float(s)


def validate_amount(x: float) -> Optional[str]:
    """Validate amount. Returns error message if invalid, None if valid."""
    if x <= 0:
        return "Сумма должна быть > 0."
    if x < MIN_AMOUNT:
        return f"Минимум {MIN_AMOUNT:g}."
    if x > MAX_AMOUNT:
        return f"Слишком большая сумма. Давай до {MAX_AMOUNT:g}."
    return None


def help_text() -> str:
    """Return help text."""
    return (
        "Команды:\n"
        "/start — меню\n"
        "/menu — меню\n"
        "/cancel — сброс\n\n"
        "Тикер можно ввести текстом или выбрать ⭐"
    )


def format_trade_signals(summary: TradeSummary) -> Tuple[str, str, str]:
    """Format trading signals for display. Returns (buy_str, sell_str, mode_note)."""
    if summary.mode == TradingMode.EXTREMA:
        buys = [t for t in summary.trades if t.action == "BUY"]
        sells = [t for t in summary.trades if t.action == "SELL"]
        buy_str = ", ".join(f"{t.date.strftime('%Y-%m-%d')} (~{t.price:.2f})" for t in buys[:3]) or "—"
        sell_str = ", ".join(f"{t.date.strftime('%Y-%m-%d')} (~{t.price:.2f})" for t in sells[:3]) or "—"
        mode_note = "Режим: экстремумы (локальные min/max на прогнозе)."
    else:
        buy = next((t for t in summary.trades if t.action == "BUY"), None)
        sell = next((t for t in summary.trades if t.action == "SELL"), None)
        buy_str = f"сегодня (~{buy.price:.2f})" if buy else "сегодня"
        sell_str = f"{sell.date.strftime('%Y-%m-%d')} (~{sell.price:.2f})" if sell else "в конце горизонта"
        mode_note = "Режим: Buy&Hold (купить сегодня, продать через 30 дней)."
    return buy_str, sell_str, mode_note


def build_router(pipeline: ForecastPipeline) -> Router:
    """Build bot router with all handlers."""
    r = Router()

    @r.message(CommandStart())
    async def start(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Выбирай действие 👇", reply_markup=_KB_MAIN)

    @r.message(Command("menu"))
    async def menu(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Меню 👇", reply_markup=_KB_MAIN)

    @r.message(Command("cancel"))
    async def cancel(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Сбросил. Начнём заново 👇", reply_markup=_KB_MAIN)

    @r.callback_query(F.data.startswith("act:"))
    async def actions(cq: CallbackQuery, state: FSMContext):
        act = cq.data.split(":", 1)[1]
        msg = cq.message

        if act in ("menu", "reset"):
            await state.clear()
            await msg.answer("Меню 👇", reply_markup=_KB_MAIN)
            await cq.answer()
            return

        if act == "exit":
            await state.clear()
            await msg.answer("Ок, пауза 🛑\nЧтобы начать снова — /start")
            await cq.answer()
            return

        if act == "new":
            await state.clear()
            await state.set_state(Form.ticker)
            await msg.answer("Введи тикер (например, AAPL) или выбери ⭐", reply_markup=_KB_TICKERS)
            await cq.answer()
            return

        if act == "ticks":
            await msg.answer("Популярные тикеры:", reply_markup=_KB_TICKERS)
            await cq.answer()
            return

        await cq.answer()

    @r.callback_query(F.data.startswith("ticker:"))
    async def pick_ticker(cq: CallbackQuery, state: FSMContext):
        t = validate_and_normalize_ticker(cq.data.split(":", 1)[1])
        if not t:
            await cq.message.answer("Некорректный тикер. Попробуй снова.", reply_markup=_KB_TICKERS)
            await cq.answer()
            return
        
        await state.update_data(ticker=t)
        await state.set_state(Form.amount)
        await cq.message.answer(f"Ок, {t}. Введи сумму (число).")
        await cq.answer()

    @r.message(Form.ticker)
    async def ticker_input(m: Message, state: FSMContext):
        t = validate_and_normalize_ticker(m.text or "")
        if not t:
            await m.answer("Это не похоже на тикер. Пример: AAPL\nПопробуй ещё раз или выбери ⭐", reply_markup=_KB_TICKERS)
            return
        await state.update_data(ticker=t)
        await state.set_state(Form.amount)
        await m.answer(f"Ок, {t}. Введи сумму (число).")

    @r.message(Form.amount)
    async def amount_input(m: Message, state: FSMContext):
        data = await state.get_data()
        ticker = (data.get("ticker") or "").strip().upper()

        try:
            amount = parse_amount(m.text or "")
        except ValueError:
            await m.answer("Сумма должна быть числом. Пример: 10000")
            return

        err = validate_amount(amount)
        if err:
            await m.answer(err)
            return

        await state.clear()
        await m.answer("Считаю… ⏳")

        try:
            res = await asyncio.to_thread(pipeline.run, user_id=m.from_user.id, ticker=ticker, amount=float(amount))
        except DataLoadError as e:
            logger.warning(f"Data load error for {ticker} by user {m.from_user.id}: {e}")
            await m.answer(f"Не удалось загрузить данные по {ticker}.\nПроверь тикер или попробуй позже.", reply_markup=_KB_MAIN)
            return
        except InsufficientDataError as e:
            logger.warning(f"Insufficient data for {ticker} by user {m.from_user.id}: {e}")
            await m.answer(f"Недостаточно исторических данных по {ticker}.\nПопробуй другой тикер.", reply_markup=_KB_MAIN)
            return
        except ModelTrainingError as e:
            logger.error(f"Model training error for {ticker} by user {m.from_user.id}: {e}")
            await m.answer(f"Не удалось обучить модели для {ticker}.\nПопробуй позже.", reply_markup=_KB_MAIN)
            return
        except Exception as e:
            logger.exception(f"Unexpected error for {ticker} by user {m.from_user.id}")
            await m.answer(f"Не смог обработать {ticker}.\nПопробуй позже.", reply_markup=_KB_MAIN)
            return

        direction = "вырастут" if res.delta_abs >= 0 else "упадут"
        shares = amount / res.current_price if res.current_price else 0.0

        buy_str, sell_str, mode_note = format_trade_signals(res.trade_summary)

        msg = (
            f"{res.ticker}\n"
            f"Цена сейчас: {res.current_price:.2f}\n"
            f"Модель: {res.chosen_model} ({res.metric_name}={res.metric_value:.4f})\n"
            f"Сравнение на тесте: {res.model_scores}\n"
            f"По среднему прогнозу за 30 дней {direction} на: {res.delta_abs:.2f} ({res.delta_pct:.2f}%)\n\n"
            f"Сигналы:\n"
            f"BUY: {buy_str}\n"
            f"SELL: {sell_str}\n\n"
            f"При вложении {amount:.0f}: куплено ~{shares:.2f} акций\n"
            f"Доход по стратегии: {res.trade_summary.profit:.2f}\n"
            f"{mode_note}\n"
            f"Дисклеймер: учебный проект, не фин. рекомендация."
        )
        await m.answer(msg)

        has_interval = res.optimistic_last is not None and res.pessimistic_last is not None

        cap1 = ["Прогноз цены на 30 дней:"]
        if has_interval:
            cap1.extend([
                "наиболее вероятная цена и диапазон возможных значений",
                f"Оптимистично: до ~{res.optimistic_last:.2f}",
                f"Пессимистично: до ~{res.pessimistic_last:.2f}",
            ])
        else:
            cap1.extend([
                "наиболее вероятная цена (точечный прогноз)",
                "Для выбранной модели интервал неопределённости не оценивается.",
            ])

        try:
            await m.answer_photo(
                BufferedInputFile(res.plot_forecast_png, filename=f"{res.ticker}_forecast.png"),
                caption="\n".join(cap1),
            )
            await m.answer_photo(
                BufferedInputFile(res.plot_trades_png, filename=f"{res.ticker}_signals.png"),
                caption=(
                    "Сигналы и тренд.\n"
                    "EMA(5) — краткосрочный импульс внутри прогноза.\n"
                    "EMA(20) — общий тренд на горизонте месяца.\n"
                    "Только для визуального анализа."
                ),
            )
        except Exception as e:
            logger.error(f"Failed to send plots for {ticker} to user {m.from_user.id}: {e}")
            await m.answer("Не смог отправить графики.")

        await m.answer("Что дальше?", reply_markup=_KB_AFTER)

    @r.message()
    async def fallback(m: Message, state: FSMContext):
        if (m.text or "").startswith("/"):
            await m.answer(help_text(), reply_markup=_KB_MAIN)
        else:
            await m.answer("Выбирай действие 👇", reply_markup=_KB_MAIN)

    return r


async def main() -> None:
    """Main bot entry point."""
    s = load_settings()
    if s.bot_token.startswith("DUMMY"):
        raise RuntimeError("BOT_TOKEN не задан. Укажи переменную окружения BOT_TOKEN.")

    bot = Bot(token=s.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    pipeline = ForecastPipeline(s)
    dp.include_router(build_router(pipeline))

    logger.info("Bot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
