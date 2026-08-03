"""SMC OMEGA: escáner autónomo, sin lookahead, para Binance y Telegram.

El proceso reconstruye todo el estado desde OHLCV público en cada ejecución.
No coloca órdenes: publica señales con entrada, invalidación, objetivos y tamaño
teórico. Esto hace segura su ejecución en un cron sin almacenamiento persistente.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import requests


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG = logging.getLogger("smc-omega")


class ConfigurationError(ValueError):
    """Configuración de entorno inválida."""


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} debe ser true o false")


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} debe ser entero") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} debe ser >= {minimum}")
    return value


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} debe ser numérico") from exc
    if not math.isfinite(value) or value < minimum:
        raise ConfigurationError(f"{name} debe ser finito y >= {minimum}")
    return value


def timeframe_ms(exchange: ccxt.Exchange, timeframe: str) -> int:
    try:
        return int(exchange.parse_timeframe(timeframe) * 1000)
    except Exception as exc:
        raise ConfigurationError(f"Temporalidad no válida: {timeframe}") from exc


def timeframe_minutes(timeframe: str) -> float:
    unit = timeframe[-1]
    number = int(timeframe[:-1]) if timeframe[:-1] else 1
    factors = {"m": 1, "h": 60, "d": 1440, "w": 10080, "M": 43200}
    if unit not in factors:
        raise ConfigurationError(f"Temporalidad CCXT no soportada: {timeframe}")
    return number * factors[unit]


def auto_htf(timeframe: str) -> str:
    minutes = timeframe_minutes(timeframe)
    if minutes <= 1:
        return "15m"
    if minutes <= 3:
        return "30m"
    if minutes <= 5:
        return "1h"
    if minutes <= 15:
        return "4h"
    if minutes <= 60:
        return "1d"
    if minutes <= 240:
        return "1w"
    return "1M"


def auto_scale(timeframe: str) -> float:
    minutes = timeframe_minutes(timeframe)
    if minutes <= 1:
        return 0.6
    if minutes <= 5:
        return 0.8
    if minutes <= 15:
        return 1.0
    if minutes <= 60:
        return 1.2
    if minutes <= 240:
        return 1.5
    if minutes <= 1440:
        return 2.0
    return 2.5


@dataclass(frozen=True)
class Config:
    exchange_id: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframe: str = "15m"
    bars: int = 1500
    use_auto_tf: bool = True
    htf: str = "4h"
    daily_tf: str = "1d"
    h4_tf: str = "4h"
    macro_pivot_len: int = 3
    pivot_len_input: int = 3
    structure_buffer_atr: float = 0.05
    equal_tol_atr: float = 0.10
    min_fvg_atr: float = 0.10
    fvg_mitigation: str = "Midpoint"
    max_fvgs: int = 24
    ob_search_depth: int = 12
    ob_impulse_atr: float = 0.8
    ob_min_body_pct: float = 0.40
    ob_zone_mode: str = "Body to Wick"
    invalidate_ob: bool = True
    max_obs: int = 16
    use_engulfing: bool = True
    use_pinbar: bool = True
    pin_ratio: float = 2.5
    pin_max_body: float = 0.35
    require_rejection: bool = True
    require_structure_trend: bool = True
    require_fvg_confluence: bool = True
    require_micro_bos: bool = True
    limit_entry_mode: str = "OB Midpoint"
    ema_fast_input: int = 35
    ema_slow_input: int = 50
    ema_macro_len: int = 200
    atr_input: int = 14
    require_macro: bool = True
    require_ema: bool = True
    use_volume: bool = True
    volume_len_input: int = 100
    use_adx: bool = True
    adx_len: int = 14
    adx_min: float = 20.0
    use_rsi: bool = True
    rsi_len: int = 14
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    use_high_liquidity_sessions: bool = True
    enable_london: bool = True
    enable_new_york: bool = True
    filter_structure_by_killzone: bool = True
    block_friday_afternoon: bool = True
    friday_cutoff_hour: int = 13
    block_critical_windows: bool = True
    cooldown_input: int = 3
    max_trades_input: int = 3
    risk_pct: float = 1.0
    account_equity: float = 10_000.0
    rr: float = 2.0
    sl_atr_buffer: float = 0.25
    min_tp_atr: float = 1.2
    use_atr_risk_scaling: bool = True
    atr_risk_lookback: int = 100
    atr_spike_threshold: float = 1.5
    minimum_risk_scale: float = 0.25
    allow_pyramiding: bool = True
    max_pyramid_entries: int = 3
    commission_pct: float = 0.08
    slippage_ticks: int = 4
    target_mode: str = "Liquidity"
    liquidity_min_distance_atr: float = 0.25
    use_ob_scoring: bool = True
    minimum_ob_score: int = 60
    score_volume_weight: int = 30
    score_sweep_weight: int = 25
    score_htf_weight: int = 25
    strict_htf_score: bool = True
    channel_len_input: int = 50
    channel_mult: float = 2.0
    liquidity_lookback_input: int = 100
    liquidity_buckets: int = 30
    freshness_minutes: int = 20
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_silent: bool = False
    alert_events: tuple[str, ...] = (
        "LIMIT_BUY_PLACED",
        "LIMIT_SELL_PLACED",
        "STRUCTURE_BREAK",
        "LIQUIDITY_SWEEP",
        "FVG",
        "CHANNEL_BREAK",
    )

    @classmethod
    def from_env(cls) -> "Config":
        symbols = tuple(
            item.strip().upper()
            for item in os.getenv("SYMBOLS", "BTC/USDT").split(",")
            if item.strip()
        )
        events = tuple(
            item.strip().upper()
            for item in os.getenv(
                "ALERT_EVENTS",
                "LIMIT_BUY_PLACED,LIMIT_SELL_PLACED,STRUCTURE_BREAK,"
                "LIQUIDITY_SWEEP,FVG,CHANNEL_BREAK",
            ).split(",")
            if item.strip()
        )
        cfg = cls(
            exchange_id=os.getenv("EXCHANGE_ID", "binance").strip().lower(),
            symbols=symbols,
            timeframe=os.getenv("TIMEFRAME", "15m").strip(),
            bars=env_int("HISTORY_BARS", 1500, 500),
            use_auto_tf=env_bool("USE_AUTO_TF", True),
            htf=os.getenv("HTF", "4h").strip(),
            daily_tf=os.getenv("DAILY_TF", "1d").strip(),
            h4_tf=os.getenv("FOUR_HOUR_TF", "4h").strip(),
            require_micro_bos=env_bool("REQUIRE_MICRO_BOS", True),
            use_high_liquidity_sessions=env_bool("USE_KILLZONES", True),
            filter_structure_by_killzone=env_bool("FILTER_STRUCTURE_BY_KILLZONE", True),
            account_equity=env_float("ACCOUNT_EQUITY", 10_000.0, 1.0),
            risk_pct=env_float("RISK_PCT", 1.0, 0.01),
            freshness_minutes=env_int("ALERT_FRESHNESS_MINUTES", 20, 1),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            telegram_silent=env_bool("TELEGRAM_SILENT", False),
            alert_events=events,
        )
        if not cfg.symbols:
            raise ConfigurationError("SYMBOLS no puede estar vacío")
        telegram_missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", cfg.telegram_token),
                ("TELEGRAM_CHAT_ID", cfg.telegram_chat_id),
            )
            if not value
        ]
        if len(telegram_missing) == 1:
            raise ConfigurationError(
                f"Falta el secreto de GitHub {telegram_missing[0]}; "
                "Telegram necesita TOKEN y CHAT_ID"
            )
        if cfg.fvg_mitigation not in {"Touch", "Midpoint", "Full Fill"}:
            raise ConfigurationError("Mitigación FVG inválida")
        return cfg


@dataclass
class FVG:
    bull: bool
    top: float
    bottom: float
    created: int


@dataclass
class OrderBlock:
    bull: bool
    top: float
    bottom: float
    created: int
    source: int
    score: int
    identifier: int
    mitigated: bool = False


@dataclass
class Event:
    name: str
    symbol: str
    timestamp: pd.Timestamp
    side: str
    price: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        raw = f"{self.symbol}|{self.timestamp.isoformat()}|{self.name}|{self.side}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]


@dataclass
class EngineState:
    last_ph: float = math.nan
    prev_ph: float = math.nan
    last_pl: float = math.nan
    prev_pl: float = math.nan
    last_ph_bar: int = -1
    last_pl_bar: int = -1
    last_eq_high: float = math.nan
    last_eq_low: float = math.nan
    structure_trend: int = 0
    fvgs: list[FVG] = field(default_factory=list)
    buy_ob: OrderBlock | None = None
    sell_ob: OrderBlock | None = None
    obs: list[OrderBlock] = field(default_factory=list)
    ob_serial: int = 0
    last_trade_bar: int = -1
    trades_by_day: dict[str, int] = field(default_factory=dict)


def fetch_ohlcv(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, bars: int
) -> pd.DataFrame:
    """Descarga con paginación y elimina la vela aún abierta."""
    step = timeframe_ms(exchange, timeframe)
    now = exchange.milliseconds()
    since = now - step * (bars + 10)
    rows: list[list[float]] = []
    while len(rows) < bars + 10:
        limit = min(1000, bars + 10 - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_since = int(batch[-1][0]) + step
        if next_since <= since or len(batch) < limit:
            break
        since = next_since
        time.sleep(exchange.rateLimit / 1000)
    unique = {int(row[0]): row[:6] for row in rows}
    closed = [row for ts, row in sorted(unique.items()) if ts + step <= now]
    if len(closed) < min(500, bars):
        raise RuntimeError(
            f"Historial insuficiente para {symbol} {timeframe}: {len(closed)} velas"
        )
    frame = pd.DataFrame(
        closed[-bars:],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    return frame.astype(float)


def confirmed_pivots(frame: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series]:
    """Valor del pivot en la barra donde queda confirmado, no en su centro."""
    high = frame["high"]
    low = frame["low"]
    window = 2 * length + 1
    centers_high = high.eq(high.rolling(window, center=True).max())
    centers_low = low.eq(low.rolling(window, center=True).min())
    ph = high.where(centers_high).shift(length)
    pl = low.where(centers_low).shift(length)
    return ph, pl


def add_indicators(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = frame.copy()
    scale = auto_scale(cfg.timeframe)
    pivot_len = max(2, round(cfg.pivot_len_input * scale))
    fast = max(5, round(cfg.ema_fast_input * scale))
    slow = max(10, round(cfg.ema_slow_input * scale))
    atr_len = max(3, round(cfg.atr_input * scale))
    volume_len = max(10, round(cfg.volume_len_input / scale))
    channel_len = max(10, round(cfg.channel_len_input / scale))
    liquidity_len = max(20, round(cfg.liquidity_lookback_input / scale))
    out["ema_fast"] = ta.ema(out["close"], length=fast)
    out["ema_slow"] = ta.ema(out["close"], length=slow)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=atr_len)
    out["atr_baseline"] = out["atr"].rolling(cfg.atr_risk_lookback).mean()
    out["rsi"] = ta.rsi(out["close"], length=cfg.rsi_len)
    adx = ta.adx(
        out["high"], out["low"], out["close"],
        length=cfg.adx_len, lensig=cfg.adx_len,
    )
    adx_column = next((c for c in adx.columns if c.startswith("ADX_")), None)
    if adx_column is None:
        raise RuntimeError("pandas-ta no devolvió la columna ADX")
    out["adx"] = adx[adx_column]
    out["avg_volume"] = out["volume"].rolling(volume_len).mean()
    out["ph"], out["pl"] = confirmed_pivots(out, pivot_len)
    x = np.arange(channel_len, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def linreg(values: np.ndarray) -> float:
        return float(values.mean() + np.dot(values - values.mean(), x_centered) / denominator * x_centered[-1])

    out["reg_mid"] = out["close"].rolling(channel_len).apply(linreg, raw=True)
    reg_dev = out["close"].rolling(channel_len).std(ddof=0) * cfg.channel_mult
    out["channel_upper"] = out["reg_mid"] + reg_dev
    out["channel_lower"] = out["reg_mid"] - reg_dev
    out["profile_high"] = out["high"].rolling(liquidity_len).max()
    out["profile_low"] = out["low"].rolling(liquidity_len).min()
    return out


def build_context(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    out = frame.copy()
    out["ema_macro"] = ta.ema(out["close"], length=length)
    out["ph"], out["pl"] = confirmed_pivots(out, 3)
    out["stored_ph"] = out["ph"].ffill()
    out["stored_pl"] = out["pl"].ffill()
    return out


def merge_confirmed(
    base: pd.DataFrame,
    context: pd.DataFrame,
    context_tf: str,
    prefix: str,
) -> pd.DataFrame:
    """Equivalente a security(..., lookahead_off): dato disponible al cierre HTF."""
    left = base.reset_index().rename(columns={"timestamp": "base_open"})
    left["available_at"] = left["base_open"] + pd.to_timedelta(
        timeframe_minutes(base.attrs["timeframe"]), unit="m"
    )
    right = context.reset_index().rename(columns={"timestamp": "context_open"})
    right["available_at"] = right["context_open"] + pd.to_timedelta(
        timeframe_minutes(context_tf), unit="m"
    )
    keep = ["available_at", "close", "ema_macro", "stored_ph", "stored_pl"]
    right = right[keep].rename(
        columns={column: f"{prefix}_{column}" for column in keep if column != "available_at"}
    )
    merged = pd.merge_asof(
        left.sort_values("available_at"),
        right.sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("base_open").drop(columns=["available_at"])


def previous_period_levels(frame: pd.DataFrame, base_tf: str) -> pd.DataFrame:
    out = frame.copy()
    source = frame[["high", "low", "volume"]].copy()
    for rule, prefix in (("1D", "pd"), ("W-MON", "pw")):
        grouped = source.resample(rule, label="left", closed="left").agg(
            {"high": "max", "low": "min", "volume": "sum"}
        )
        available = grouped.shift(1)
        mapped = available.reindex(out.index, method="ffill")
        out[f"{prefix}h"] = mapped["high"]
        out[f"{prefix}l"] = mapped["low"]
        out[f"{prefix}v"] = mapped["volume"]
    out.attrs["timeframe"] = base_tf
    return out


def in_clock_window(ts: pd.Timestamp, zone: str, start: str, end: str) -> bool:
    local = ts.tz_convert(ZoneInfo(zone))
    value = local.hour * 60 + local.minute
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    lower, upper = sh * 60 + sm, eh * 60 + em
    return lower <= value < upper if lower <= upper else value >= lower or value < upper


def session_flags(ts: pd.Timestamp, cfg: Config) -> tuple[bool, bool, bool]:
    london = cfg.enable_london and in_clock_window(
        ts, "Europe/London", "08:00", "11:00"
    )
    new_york = cfg.enable_new_york and in_clock_window(
        ts, "America/New_York", "09:30", "12:00"
    )
    high_liquidity = not cfg.use_high_liquidity_sessions or london or new_york
    return london, new_york, high_liquidity


def calendar_ok(ts: pd.Timestamp, cfg: Config) -> bool:
    weekday = ts.weekday()
    if weekday >= 5:
        return False
    if cfg.block_friday_afternoon and weekday == 4 and ts.hour >= cfg.friday_cutoff_hour:
        return False
    if cfg.block_critical_windows and (
        in_clock_window(ts, "UTC", "09:30", "10:00")
        or in_clock_window(ts, "UTC", "15:50", "16:10")
    ):
        return False
    return True


def body_pct(row: pd.Series) -> float:
    span = row.high - row.low
    return abs(row.close - row.open) / span if span > 0 else 0.0


def bull_engulf(current: pd.Series, previous: pd.Series) -> bool:
    return (
        current.close > current.open
        and previous.close < previous.open
        and current.close >= previous.open
        and current.open <= previous.close
    )


def bear_engulf(current: pd.Series, previous: pd.Series) -> bool:
    return (
        current.close < current.open
        and previous.close > previous.open
        and current.close <= previous.open
        and current.open >= previous.close
    )


def bull_pin(row: pd.Series, cfg: Config) -> bool:
    body, span = abs(row.close - row.open), row.high - row.low
    lower = min(row.open, row.close) - row.low
    return (
        span > 0 and row.close > row.open and body / span <= cfg.pin_max_body
        and lower >= body * cfg.pin_ratio
    )


def bear_pin(row: pd.Series, cfg: Config) -> bool:
    body, span = abs(row.close - row.open), row.high - row.low
    upper = row.high - max(row.open, row.close)
    return (
        span > 0 and row.close < row.open and body / span <= cfg.pin_max_body
        and upper >= body * cfg.pin_ratio
    )


def fvg_threshold(fvg: FVG, mode: str) -> float:
    if mode == "Touch":
        return fvg.top if fvg.bull else fvg.bottom
    if mode == "Midpoint":
        return (fvg.top + fvg.bottom) / 2
    return fvg.bottom if fvg.bull else fvg.top


def nearest_above(base: float, minimum: float, candidates: Iterable[float]) -> float:
    valid = [x for x in candidates if pd.notna(x) and x >= base + minimum]
    return min(valid) if valid else math.nan


def nearest_below(base: float, minimum: float, candidates: Iterable[float]) -> float:
    valid = [x for x in candidates if pd.notna(x) and x <= base - minimum]
    return max(valid) if valid else math.nan


def micro_bos(micro: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, bool]:
    bars = micro[(micro.index >= start) & (micro.index < end)]
    if len(bars) < 2:
        return False, False
    previous_high = bars["high"].shift(1)
    previous_low = bars["low"].shift(1)
    bull = ((bars.close > bars.open) & (bars.close > previous_high)).any()
    bear = ((bars.close < bars.open) & (bars.close < previous_low)).any()
    return bool(bull), bool(bear)


def unit_risk(fill: float, stop: float, cfg: Config, tick: float) -> float:
    market = abs(fill - stop)
    commission = fill * cfg.commission_pct / 100
    slippage = tick * cfg.slippage_ticks
    return max(market + commission + slippage, tick)


def limit_price(
    bull: bool, ob: OrderBlock, fvg: FVG | None, close: float, mode: str
) -> float:
    ob_mid = (ob.top + ob.bottom) / 2
    fvg_mid = (fvg.top + fvg.bottom) / 2 if fvg else ob_mid
    if mode == "OB Extreme":
        raw = ob.bottom if bull else ob.top
    elif mode == "FVG Midpoint":
        raw = fvg_mid
    elif mode == "OB/FVG Equilibrium":
        raw = (ob_mid + fvg_mid) / 2
    else:
        raw = ob_mid
    return min(raw, close) if bull else max(raw, close)


def liquidity_profile(frame: pd.DataFrame, cfg: Config) -> tuple[float, float]:
    scale = auto_scale(cfg.timeframe)
    length = max(20, round(cfg.liquidity_lookback_input / scale))
    sample = frame.iloc[-length:]
    low, high = float(sample.low.min()), float(sample.high.max())
    if high <= low:
        return low, 0.0
    size = (high - low) / cfg.liquidity_buckets
    money = np.zeros(cfg.liquidity_buckets)
    hlc3 = (sample.high + sample.low + sample.close) / 3
    indexes = np.floor((hlc3 - low) / size).clip(0, cfg.liquidity_buckets - 1)
    for bucket, value in zip(indexes.astype(int), sample.volume * hlc3, strict=True):
        money[bucket] += value
    best = int(money.argmax())
    return low + (best + 0.5) * size, float(money[best])


class SMCStrategy:
    def __init__(
        self,
        cfg: Config,
        symbol: str,
        frame: pd.DataFrame,
        micro: pd.DataFrame,
        tick_size: float,
    ) -> None:
        self.cfg = cfg
        self.symbol = symbol
        self.df = frame
        self.micro = micro
        self.tick = tick_size
        self.state = EngineState()
        self.events: list[Event] = []
        minutes = timeframe_minutes(cfg.timeframe)
        self.bar_delta = pd.to_timedelta(minutes, unit="m")
        self.cooldown = 6 if minutes <= 5 else 3 if minutes <= 60 else 2
        self.max_trades = 5 if minutes <= 5 else 4 if minutes <= 60 else 3

    def emit(
        self, i: int, name: str, side: str, price: float, **details: Any
    ) -> None:
        self.events.append(
            Event(name, self.symbol, self.df.index[i], side, float(price), details)
        )

    def run(self) -> list[Event]:
        start = max(5, self.cfg.ema_macro_len)
        for i in range(start, len(self.df)):
            self.process_bar(i)
        return self.events

    def process_bar(self, i: int) -> None:
        row, previous = self.df.iloc[i], self.df.iloc[i - 1]
        if pd.isna(row.atr):
            return
        s, cfg = self.state, self.cfg
        _, _, high_liquidity = session_flags(self.df.index[i], cfg)
        structure_session = not cfg.filter_structure_by_killzone or high_liquidity

        if pd.notna(row.ph):
            s.prev_ph, s.last_ph = s.last_ph, float(row.ph)
            s.last_ph_bar = i - max(2, round(cfg.pivot_len_input * auto_scale(cfg.timeframe)))
            if pd.notna(s.prev_ph) and abs(s.last_ph - s.prev_ph) <= row.atr * cfg.equal_tol_atr:
                s.last_eq_high = (s.last_ph + s.prev_ph) / 2
        if pd.notna(row.pl):
            s.prev_pl, s.last_pl = s.last_pl, float(row.pl)
            s.last_pl_bar = i - max(2, round(cfg.pivot_len_input * auto_scale(cfg.timeframe)))
            if pd.notna(s.prev_pl) and abs(s.last_pl - s.prev_pl) <= row.atr * cfg.equal_tol_atr:
                s.last_eq_low = (s.last_pl + s.prev_pl) / 2

        buffer = row.atr * cfg.structure_buffer_atr
        bull_break = (
            pd.notna(s.last_ph) and row.close > row.open
            and row.close > s.last_ph + buffer
            and previous.close <= s.last_ph + buffer and structure_session
        )
        bear_break = (
            pd.notna(s.last_pl) and row.close < row.open
            and row.close < s.last_pl - buffer
            and previous.close >= s.last_pl - buffer and structure_session
        )
        if bull_break:
            kind = "CHOCH" if s.structure_trend == -1 else "BOS"
            s.structure_trend = 1
            self.emit(i, "STRUCTURE_BREAK", "BULL", row.close, kind=kind, level=s.last_ph)
        if bear_break:
            kind = "CHOCH" if s.structure_trend == 1 else "BOS"
            s.structure_trend = -1
            self.emit(i, "STRUCTURE_BREAK", "BEAR", row.close, kind=kind, level=s.last_pl)

        sweep_high = (
            structure_session and pd.notna(s.last_ph)
            and row.high > s.last_ph and row.close < s.last_ph
        )
        sweep_low = (
            structure_session and pd.notna(s.last_pl)
            and row.low < s.last_pl and row.close > s.last_pl
        )
        if sweep_high:
            self.emit(i, "LIQUIDITY_SWEEP", "HIGH", row.close, level=s.last_ph)
        if sweep_low:
            self.emit(i, "LIQUIDITY_SWEEP", "LOW", row.close, level=s.last_pl)

        mitigated_bull: FVG | None = None
        mitigated_bear: FVG | None = None
        survivors: list[FVG] = []
        for gap in s.fvgs:
            mitigated = i > gap.created and (
                row.low <= fvg_threshold(gap, cfg.fvg_mitigation)
                if gap.bull else row.high >= fvg_threshold(gap, cfg.fvg_mitigation)
            )
            if mitigated:
                if gap.bull:
                    mitigated_bull = gap
                else:
                    mitigated_bear = gap
            else:
                survivors.append(gap)
        s.fvgs = survivors
        two_back = self.df.iloc[i - 2]
        bull_fvg = row.low > two_back.high and row.low - two_back.high >= row.atr * cfg.min_fvg_atr
        bear_fvg = row.high < two_back.low and two_back.low - row.high >= row.atr * cfg.min_fvg_atr
        if bull_fvg:
            gap = FVG(True, float(row.low), float(two_back.high), i)
            s.fvgs.append(gap)
            self.emit(i, "FVG", "BULL", row.close, top=gap.top, bottom=gap.bottom)
        if bear_fvg:
            gap = FVG(False, float(two_back.low), float(row.high), i)
            s.fvgs.append(gap)
            self.emit(i, "FVG", "BEAR", row.close, top=gap.top, bottom=gap.bottom)
        s.fvgs = s.fvgs[-cfg.max_fvgs:]

        volume_ok = not cfg.use_volume or row.volume > row.avg_volume
        prior5 = self.df.iloc[i - 5:i]
        bull_impulse = (
            row.close > row.open and abs(row.close - row.open) >= row.atr * cfg.ob_impulse_atr
            and row.close > prior5.high.max() and volume_ok
        )
        bear_impulse = (
            row.close < row.open and abs(row.close - row.open) >= row.atr * cfg.ob_impulse_atr
            and row.close < prior5.low.min() and volume_ok
        )
        recent_events = self.events[-12:]
        recent_sweep_low = any(
            e.name == "LIQUIDITY_SWEEP" and e.side == "LOW"
            and e.timestamp in {self.df.index[i - 1], self.df.index[i - 2]}
            for e in recent_events
        )
        recent_sweep_high = any(
            e.name == "LIQUIDITY_SWEEP" and e.side == "HIGH"
            and e.timestamp in {self.df.index[i - 1], self.df.index[i - 2]}
            for e in recent_events
        )
        macro_bull = row.htf_close > row.htf_ema_macro
        daily_bull = row.daily_close > row.daily_ema_macro
        h4_bull = row.h4_close > row.h4_ema_macro
        if bull_impulse:
            source = self.find_opposite(i, bull=True)
            candle = self.df.iloc[source]
            bull_htf_score_ok = (
                macro_bull and daily_bull and h4_bull
                if cfg.strict_htf_score
                else macro_bull
            )
            score = min(
                100, 20
                + (cfg.score_volume_weight if row.volume > row.avg_volume * 1.2 else 0)
                + (cfg.score_sweep_weight if recent_sweep_low else 0)
                + (cfg.score_htf_weight if bull_htf_score_ok else 0),
            )
            s.ob_serial += 1
            s.buy_ob = OrderBlock(
                True,
                float(candle.high if cfg.ob_zone_mode == "Full Candle" else candle.open),
                float(candle.low), i, source, score, s.ob_serial,
            )
            s.obs.append(s.buy_ob)
        if bear_impulse:
            source = self.find_opposite(i, bull=False)
            candle = self.df.iloc[source]
            bear_htf_score_ok = (
                (not macro_bull) and (not daily_bull) and (not h4_bull)
                if cfg.strict_htf_score
                else not macro_bull
            )
            score = min(
                100, 20
                + (cfg.score_volume_weight if row.volume > row.avg_volume * 1.2 else 0)
                + (cfg.score_sweep_weight if recent_sweep_high else 0)
                + (cfg.score_htf_weight if bear_htf_score_ok else 0),
            )
            s.ob_serial += 1
            s.sell_ob = OrderBlock(
                False, float(candle.high),
                float(candle.low if cfg.ob_zone_mode == "Full Candle" else candle.open),
                i, source, score, s.ob_serial,
            )
            s.obs.append(s.sell_ob)
        s.obs = s.obs[-cfg.max_obs:]
        active_ids = {ob.identifier for ob in s.obs}
        if s.buy_ob and s.buy_ob.identifier not in active_ids:
            s.buy_ob = None
        if s.sell_ob and s.sell_ob.identifier not in active_ids:
            s.sell_ob = None
        for ob in (s.buy_ob, s.sell_ob):
            if ob and i > ob.created and row.low <= ob.top and row.high >= ob.bottom:
                ob.mitigated = True
        if cfg.invalidate_ob and s.buy_ob and row.close < s.buy_ob.bottom:
            s.buy_ob = None
        if cfg.invalidate_ob and s.sell_ob and row.close > s.sell_ob.top:
            s.sell_ob = None

        previous_upper, previous_lower = previous.channel_upper, previous.channel_lower
        if previous.close <= previous_upper and row.close > row.channel_upper:
            self.emit(i, "CHANNEL_BREAK", "UP", row.close, level=row.channel_upper)
        if previous.close >= previous_lower and row.close < row.channel_lower:
            self.emit(i, "CHANNEL_BREAK", "DOWN", row.close, level=row.channel_lower)

        self.evaluate_signals(i, mitigated_bull, mitigated_bear, high_liquidity)

    def find_opposite(self, i: int, bull: bool) -> int:
        for offset in range(1, self.cfg.ob_search_depth + 1):
            candidate = self.df.iloc[i - offset]
            opposite = candidate.close < candidate.open if bull else candidate.close > candidate.open
            if opposite and body_pct(candidate) >= self.cfg.ob_min_body_pct:
                return i - offset
        return i - 1

    def latest_fvg(self, bull: bool) -> FVG | None:
        return next((gap for gap in reversed(self.state.fvgs) if gap.bull == bull), None)

    def evaluate_signals(
        self,
        i: int,
        mitigated_bull: FVG | None,
        mitigated_bear: FVG | None,
        high_liquidity: bool,
    ) -> None:
        cfg, s = self.cfg, self.state
        row, previous = self.df.iloc[i], self.df.iloc[i - 1]
        bull_gap = self.latest_fvg(True) or mitigated_bull
        bear_gap = self.latest_fvg(False) or mitigated_bear
        buy_ob, sell_ob = s.buy_ob, s.sell_ob
        bull_trigger = (
            (cfg.use_engulfing and bull_engulf(row, previous))
            or (cfg.use_pinbar and bull_pin(row, cfg))
        )
        bear_trigger = (
            (cfg.use_engulfing and bear_engulf(row, previous))
            or (cfg.use_pinbar and bear_pin(row, cfg))
        )
        macro_bull = row.htf_close > row.htf_ema_macro
        macro_bear = row.htf_close < row.htf_ema_macro
        volume_ok = not cfg.use_volume or row.volume > row.avg_volume
        adx_ok = not cfg.use_adx or row.adx >= cfg.adx_min
        day_key = self.df.index[i].strftime("%Y-%m-%d")
        cooldown_ok = s.last_trade_bar < 0 or i - s.last_trade_bar >= self.cooldown
        common = (
            high_liquidity and calendar_ok(self.df.index[i], cfg)
            and volume_ok and adx_ok and cooldown_ok
            and s.trades_by_day.get(day_key, 0) < self.max_trades
        )
        start, end = self.df.index[i], self.df.index[i] + self.bar_delta
        micro_bull, micro_bear = micro_bos(self.micro, start, end)
        buy_in_ob = bool(
            buy_ob and row.low <= buy_ob.top and row.close >= buy_ob.bottom
        )
        sell_in_ob = bool(
            sell_ob and row.high >= sell_ob.bottom and row.close <= sell_ob.top
        )
        buy_fvg_ok = (
            not cfg.require_fvg_confluence
            or bool(
                buy_ob and bull_gap and buy_ob.top >= bull_gap.bottom
                and buy_ob.bottom <= bull_gap.top and row.low <= bull_gap.top
                and row.high >= bull_gap.bottom
            )
        )
        sell_fvg_ok = (
            not cfg.require_fvg_confluence
            or bool(
                sell_ob and bear_gap and sell_ob.top >= bear_gap.bottom
                and sell_ob.bottom <= bear_gap.top and row.high >= bear_gap.bottom
                and row.low <= bear_gap.top
            )
        )
        buy_setup = (
            bull_trigger and (not cfg.require_macro or macro_bull)
            and (not cfg.require_ema or row.ema_fast > row.ema_slow)
            and (not cfg.require_structure_trend or s.structure_trend == 1)
            and (not cfg.use_rsi or row.rsi < cfg.rsi_ob)
        )
        sell_setup = (
            bear_trigger and (not cfg.require_macro or macro_bear)
            and (not cfg.require_ema or row.ema_fast < row.ema_slow)
            and (not cfg.require_structure_trend or s.structure_trend == -1)
            and (not cfg.use_rsi or row.rsi > cfg.rsi_os)
        )
        buy_signal = bool(
            common and buy_ob and buy_in_ob and buy_fvg_ok and buy_setup
            and (not cfg.use_ob_scoring or buy_ob.score >= cfg.minimum_ob_score)
            and (not cfg.require_micro_bos or micro_bull)
            and (
                not cfg.require_rejection
                or row.close >= (buy_ob.top + buy_ob.bottom) / 2
            )
        )
        sell_signal = bool(
            common and sell_ob and sell_in_ob and sell_fvg_ok and sell_setup
            and (not cfg.use_ob_scoring or sell_ob.score >= cfg.minimum_ob_score)
            and (not cfg.require_micro_bos or micro_bear)
            and (
                not cfg.require_rejection
                or row.close <= (sell_ob.top + sell_ob.bottom) / 2
            )
        )
        if buy_signal:
            self.create_trade_event(i, True, buy_ob, bull_gap)
        elif sell_signal:
            self.create_trade_event(i, False, sell_ob, bear_gap)

    def create_trade_event(
        self, i: int, bull: bool, ob: OrderBlock, gap: FVG | None
    ) -> None:
        cfg, row, s = self.cfg, self.df.iloc[i], self.state
        entry = limit_price(bull, ob, gap, row.close, cfg.limit_entry_mode)
        stop = (
            ob.bottom - row.atr * cfg.sl_atr_buffer
            if bull else ob.top + row.atr * cfg.sl_atr_buffer
        )
        distance = entry - stop if bull else stop - entry
        if distance <= self.tick:
            return
        atr_ratio = (
            row.atr / row.atr_baseline
            if pd.notna(row.atr_baseline) and row.atr_baseline > 0 else 1.0
        )
        risk_scale = (
            1.0
            if not cfg.use_atr_risk_scaling or atr_ratio <= cfg.atr_spike_threshold
            else max(cfg.minimum_risk_scale, cfg.atr_spike_threshold / atr_ratio)
        )
        effective_entries = cfg.max_pyramid_entries if cfg.allow_pyramiding else 1
        cash_risk = cfg.account_equity * cfg.risk_pct / 100 * risk_scale
        tranche = cash_risk / effective_entries
        quantity = tranche / unit_risk(entry, stop, cfg, self.tick)
        minimum = row.atr * cfg.liquidity_min_distance_atr
        if bull:
            liquidity = nearest_above(
                entry, minimum,
                (s.last_eq_high, s.last_ph, row.h4_stored_ph, row.daily_stored_ph, row.pdh, row.pwh),
            )
            fixed = entry + max(distance * cfg.rr, row.atr * cfg.min_tp_atr)
        else:
            liquidity = nearest_below(
                entry, minimum,
                (s.last_eq_low, s.last_pl, row.h4_stored_pl, row.daily_stored_pl, row.pdl, row.pwl),
            )
            fixed = entry - max(distance * cfg.rr, row.atr * cfg.min_tp_atr)
        target = liquidity if cfg.target_mode == "Liquidity" and pd.notna(liquidity) else fixed
        side = "LONG" if bull else "SHORT"
        name = "LIMIT_BUY_PLACED" if bull else "LIMIT_SELL_PLACED"
        self.emit(
            i, name, side, row.close, entry=entry, stop_loss=stop,
            take_profit=target, quantity=quantity, ob_score=ob.score,
            atr=float(row.atr), atr_ratio=float(atr_ratio),
            effective_risk_pct=cfg.risk_pct * risk_scale,
        )
        day_key = self.df.index[i].strftime("%Y-%m-%d")
        s.trades_by_day[day_key] = s.trades_by_day.get(day_key, 0) + 1
        s.last_trade_bar = i


def prepare_symbol(
    exchange: ccxt.Exchange, symbol: str, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    base = fetch_ohlcv(exchange, symbol, cfg.timeframe, cfg.bars)
    base.attrs["timeframe"] = cfg.timeframe
    base = add_indicators(base, cfg)
    base = previous_period_levels(base, cfg.timeframe)
    base.attrs["timeframe"] = cfg.timeframe
    htf = auto_htf(cfg.timeframe) if cfg.use_auto_tf else cfg.htf
    contexts = (
        (htf, "htf"),
        (cfg.daily_tf, "daily"),
        (cfg.h4_tf, "h4"),
    )
    for context_tf, prefix in contexts:
        context = build_context(
            fetch_ohlcv(exchange, symbol, context_tf, max(500, cfg.ema_macro_len * 3)),
            cfg.ema_macro_len,
        )
        base.attrs["timeframe"] = cfg.timeframe
        base = merge_confirmed(base, context, context_tf, prefix)
        base.attrs["timeframe"] = cfg.timeframe
    micro_bars = min(1500, max(500, int(timeframe_minutes(cfg.timeframe) * 20)))
    micro = (
        base[["open", "high", "low", "close", "volume"]].copy()
        if cfg.timeframe == "1m"
        else fetch_ohlcv(exchange, symbol, "1m", micro_bars)
    )
    market = exchange.market(symbol)
    precision = market.get("precision", {}).get("price")
    if isinstance(precision, int):
        tick = 10.0 ** -precision
    elif isinstance(precision, (float, str)) and float(precision) > 0:
        tick = float(precision)
    else:
        tick = 1e-8
    return base, micro, tick


def format_number(value: Any, digits: int = 6) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}".rstrip("0").rstrip(".")


def telegram_message(event: Event) -> str:
    icons = {
        "LIMIT_BUY_PLACED": "🟢",
        "LIMIT_SELL_PLACED": "🔴",
        "STRUCTURE_BREAK": "🏗️",
        "LIQUIDITY_SWEEP": "💧",
        "FVG": "⚡",
        "CHANNEL_BREAK": "📈",
    }
    lines = [
        f"{icons.get(event.name, '🔔')} <b>SMC OMEGA · {html.escape(event.name)}</b>",
        f"<b>Activo:</b> <code>{html.escape(event.symbol)}</code>",
        f"<b>Lado:</b> {html.escape(event.side)}",
        f"<b>Cierre:</b> <code>{format_number(event.price, 8)}</code>",
        f"<b>Vela UTC:</b> {event.timestamp.strftime('%Y-%m-%d %H:%M')}",
    ]
    labels = {
        "entry": "Entrada límite",
        "stop_loss": "Stop loss",
        "take_profit": "Take profit",
        "quantity": "Cantidad teórica",
        "ob_score": "Score OB",
        "effective_risk_pct": "Riesgo efectivo %",
        "kind": "Estructura",
        "level": "Nivel",
        "top": "Techo",
        "bottom": "Suelo",
    }
    for key, label in labels.items():
        if key in event.details:
            value = event.details[key]
            shown = format_number(value, 8) if isinstance(value, (int, float)) else str(value)
            lines.append(f"<b>{label}:</b> <code>{html.escape(shown)}</code>")
    lines.append(f"<b>ID:</b> <code>{event.event_id}</code>")
    return "\n".join(lines)


def send_telegram(cfg: Config, event: Event) -> None:
    if not cfg.telegram_token:
        LOG.info("Telegram no configurado; evento=%s", json.dumps(event.details))
        return
    url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": cfg.telegram_chat_id,
            "text": telegram_message(event),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "disable_notification": cfg.telegram_silent,
        },
        timeout=(5, 20),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram rechazó el mensaje: {payload}")


def create_exchange(cfg: Config) -> ccxt.Exchange:
    try:
        exchange_class = getattr(ccxt, cfg.exchange_id)
    except AttributeError as exc:
        raise ConfigurationError(f"Exchange CCXT desconocido: {cfg.exchange_id}") from exc
    exchange = exchange_class(
        {"enableRateLimit": True, "timeout": 30_000, "options": {"defaultType": "spot"}}
    )
    exchange.load_markets()
    return exchange


def is_fresh(event: Event, cfg: Config) -> bool:
    candle_close = event.timestamp + pd.to_timedelta(
        timeframe_minutes(cfg.timeframe), unit="m"
    )
    age = datetime.now(UTC) - candle_close.to_pydatetime()
    return 0 <= age.total_seconds() <= cfg.freshness_minutes * 60


def main() -> int:
    try:
        cfg = Config.from_env()
        exchange = create_exchange(cfg)
        sent = 0
        for symbol in cfg.symbols:
            if symbol not in exchange.markets:
                raise ConfigurationError(f"{symbol} no existe en {cfg.exchange_id}")
            LOG.info("Analizando %s en %s", symbol, cfg.timeframe)
            frame, micro, tick = prepare_symbol(exchange, symbol, cfg)
            poc, poc_money = liquidity_profile(frame, cfg)
            events = SMCStrategy(cfg, symbol, frame, micro, tick).run()
            eligible = [
                event for event in events
                if event.name in cfg.alert_events and is_fresh(event, cfg)
            ]
            for event in eligible:
                event.details.setdefault("profile_poc", poc)
                event.details.setdefault("profile_money", poc_money)
                send_telegram(cfg, event)
                sent += 1
                LOG.info(
                    "Alerta enviada event=%s symbol=%s id=%s",
                    event.name, symbol, event.event_id,
                )
        LOG.info("Ejecución completada; alertas=%d", sent)
        return 0
    except (ConfigurationError, RuntimeError, requests.RequestException, ccxt.BaseError) as exc:
        LOG.error("Ejecución fallida: %s", exc)
        return 1
    except Exception:
        LOG.exception("Error inesperado")
        return 1


if __name__ == "__main__":
    sys.exit(main())
