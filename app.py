from dotenv import load_dotenv
load_dotenv()

import asyncio
import re
from typing import List, Tuple, Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import BufferedInputFile

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage

from config import load_settings
from core import ForecastPipeline


EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
MIN_AMOUNT = 1.0
MAX_AMOUNT = 1_000_000_000.0



class Form(StatesGroup):
    ticker = State()
    amount = State()


# Keyboards 

def kb(actions: List[Tuple[str, str]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=txt, callback_data=data)] for txt, data in actions
    ])


def kb_main() -> InlineKeyboardMarkup:
    return kb([
        ("🔍 Новый прогноз", "act:new"),
        ("⭐ Популярные тикеры", "act:ticks"),
        ("🧹 Сброс", "act:reset"),
        ("🛑 Завершить", "act:exit"),
    ])


def kb_after() -> InlineKeyboardMarkup:
    return kb([
        ("🔍 Новый прогноз", "act:new"),
        ("⭐ Популярные тикеры", "act:ticks"),
        ("🏠 В меню", "act:menu"),
        ("🛑 Завершить", "act:exit"),
    ])


def kb_tickers() -> InlineKeyboardMarkup:
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


def normalize_ticker(s: str) -> str:
    return s.strip().upper()


def ticker_ok(t: str) -> bool:
    return bool(TICKER_RE.match(t))


def parse_amount(text: str) -> float:
    s = text.strip().replace(" ", "").replace("_", "").replace(",", ".")
    return float(s)


def validate_amount(x: float) -> Optional[str]:
    if x <= 0:
        return "Сумма должна быть > 0."
    if x < MIN_AMOUNT:
        return f"Минимум {MIN_AMOUNT:g}."
    if x > MAX_AMOUNT:
        return f"Слишком большая сумма. Давай до {MAX_AMOUNT:g}."
    return None


def help_text() -> str:
    return (
        "Команды:\n"
        "/start — меню\n"
        "/menu — меню\n"
        "/cancel — сброс\n\n"
        "Тикер можно ввести текстом или выбрать ⭐"
    )


# Bot 

def build_router(pipeline: ForecastPipeline) -> Router:
    r = Router()

    @r.message(CommandStart())
    async def start(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Выбирай действие 👇", reply_markup=kb_main())

    @r.message(Command("menu"))
    async def menu(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Меню 👇", reply_markup=kb_main())

    @r.message(Command("cancel"))
    async def cancel(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("Сбросил. Начнём заново 👇", reply_markup=kb_main())

    @r.callback_query(F.data.startswith("act:"))
    async def actions(cq: CallbackQuery, state: FSMContext):
        act = cq.data.split(":", 1)[1]
        msg = cq.message

        if act in ("menu", "reset"):
            await state.clear()
            await msg.answer("Меню 👇", reply_markup=kb_main())
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
            await msg.answer("Введи тикер (например, AAPL) или выбери ⭐", reply_markup=kb_tickers())
            await cq.answer()
            return

        if act == "ticks":
            await msg.answer("Популярные тикеры:", reply_markup=kb_tickers())
            await cq.answer()
            return

        await cq.answer()

    @r.callback_query(F.data.startswith("ticker:"))
    async def pick_ticker(cq: CallbackQuery, state: FSMContext):
        t = cq.data.split(":", 1)[1].strip().upper()
        await state.update_data(ticker=t)
        await state.set_state(Form.amount)
        await cq.message.answer(f"Ок, {t}. Введи сумму (число).")
        await cq.answer()

    @r.message(Form.ticker)
    async def ticker_input(m: Message, state: FSMContext):
        t = normalize_ticker(m.text or "")
        if not ticker_ok(t):
            await m.answer("Это не похоже на тикер. Пример: AAPL\nПопробуй ещё раз или выбери ⭐", reply_markup=kb_tickers())
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
        except Exception:
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
        except Exception as e:
            await m.answer(f"Не смог обработать {ticker}.\nПричина: {e}", reply_markup=kb_main())
            return

        direction = "вырастут" if res.delta_abs >= 0 else "упадут"
        shares = amount / res.current_price if res.current_price else 0.0

        if res.trade_summary.mode == "EXTREMA":
            buys = [t for t in res.trade_summary.trades if t.action == "BUY"]
            sells = [t for t in res.trade_summary.trades if t.action == "SELL"]
            buy_str = ", ".join(f"{t.date.strftime('%Y-%m-%d')} (~{t.price:.2f})" for t in buys[:3]) or "—"
            sell_str = ", ".join(f"{t.date.strftime('%Y-%m-%d')} (~{t.price:.2f})" for t in sells[:3]) or "—"
            mode_note = "Режим: экстремумы (локальные min/max на прогнозе)."
        else:
            buy = next((t for t in res.trade_summary.trades if t.action == "BUY"), None)
            sell = next((t for t in res.trade_summary.trades if t.action == "SELL"), None)
            buy_str = f"сегодня (~{buy.price:.2f})" if buy else "сегодня"
            sell_str = f"{sell.date.strftime('%Y-%m-%d')} (~{sell.price:.2f})" if sell else "в конце горизонта"
            mode_note = "Режим: Buy&Hold (купить сегодня, продать через 30 дней)."

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

        # График 1 
        cap1 = [
            "Прогноз цены на 30 дней.",
            "Оранжевая линия — наиболее вероятная цена и диапазон возможных значений.",
        ]
        if res.optimistic_last is not None and res.pessimistic_last is not None:
            cap1.append(f"Оптимистично: до ~{res.optimistic_last:.2f}")
            cap1.append(f"Пессимистично: до ~{res.pessimistic_last:.2f}")

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
        except Exception:
            await m.answer("Не смог отправить графики.")

        await m.answer("Что дальше?", reply_markup=kb_after())

    @r.message()
    async def fallback(m: Message, state: FSMContext):

        if (m.text or "").startswith("/"):
            await m.answer(help_text(), reply_markup=kb_main())
        else:
            await m.answer("Выбирай действие 👇", reply_markup=kb_main())

    return r


async def main() -> None:
    s = load_settings()
    if s.bot_token.startswith("DUMMY"):
        raise RuntimeError("BOT_TOKEN не задан. Укажи переменную окружения BOT_TOKEN.")

    bot = Bot(token=s.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    pipeline = ForecastPipeline(s)
    dp.include_router(build_router(pipeline))

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
