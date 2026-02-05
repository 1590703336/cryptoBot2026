#!/usr/bin/env python3
import base64
import hashlib
import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, quote

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


LOG = logging.getLogger("htx-telegram-bot")


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (key=value, # comments)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Config:
    telegram_token: str
    telegram_chat_id: int
    telegram_poll_timeout: int

    htx_access_key: str
    htx_secret_key: str
    htx_base_url: str
    htx_account_id: Optional[str]

    symbols: List[str]
    order_source: str
    order_states: List[str]
    order_types: List[str]
    order_poll_seconds: int

    daily_report_time: str
    timezone: str

    ai_api_url: Optional[str]
    ai_api_key: Optional[str]
    ai_model: Optional[str]
    ai_timeout: int

    total_money: float

    total_money: float


class HTXClient:
    def __init__(self, access_key: str, secret_key: str, base_url: str, timeout: int = 15) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        parsed = urlparse(self.base_url)
        self.host = parsed.netloc.lower()

    def _sign_params(self, method: str, path: str, params: Dict[str, str]) -> Dict[str, str]:
        signed = dict(params)
        signed.update({
            "AccessKeyId": self.access_key,
            "SignatureMethod": "HmacSHA256",
            "SignatureVersion": "2",
            "Timestamp": _now_utc_iso(),
        })
        # Sort by ASCII and encode with RFC3986 (space as %20, not +)
        ordered = sorted(signed.items(), key=lambda kv: kv[0])
        canonical = urlencode(ordered, quote_via=quote)
        payload = "\n".join([method.upper(), self.host, path, canonical])
        digest = hmac.new(self.secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode("utf-8")
        signed["Signature"] = signature
        return signed

    def _request(self, method: str, path: str, params: Optional[Dict[str, str]] = None, auth: bool = False) -> Dict:
        params = params or {}
        if auth:
            params = self._sign_params(method, path, params)
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(method, url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"HTX request failed: {exc}") from exc

        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"HTX error: {data.get('err-msg', data)}")
        return data

    def get_accounts(self) -> List[Dict]:
        data = self._request("GET", "/v1/account/accounts", auth=True)
        return data.get("data", [])

    def get_spot_account_id(self) -> Optional[str]:
        for account in self.get_accounts():
            if account.get("type") == "spot" and account.get("state") == "working":
                return str(account.get("id"))
        return None

    def get_open_orders(self, account_id: str, symbol: str, side: str = "buy") -> List[Dict]:
        params = {"account-id": account_id, "symbol": symbol}
        if side:
            params["side"] = side
        data = self._request("GET", "/v1/order/openOrders", params=params, auth=True)
        return data.get("data", [])

    def get_orders(self, symbol: str, states: List[str], types: List[str], size: int = 100) -> List[Dict]:
        params = {
            "symbol": symbol,
            "states": ",".join(states),
            "size": str(size),
        }
        if types:
            params["types"] = ",".join(types)
        data = self._request("GET", "/v1/order/orders", params=params, auth=True)
        return data.get("data", [])

    def get_market_detail(self, symbol: str) -> Dict:
        data = self._request("GET", "/market/detail", params={"symbol": symbol}, auth=False)
        return data.get("tick", {})

    def get_kline(self, symbol: str, period: str, size: int = 2) -> List[Dict]:
        params = {"symbol": symbol, "period": period, "size": str(size)}
        data = self._request("GET", "/market/history/kline", params=params, auth=False)
        return data.get("data", [])

    def get_account_balance(self, account_id: str) -> List[Dict]:
        data = self._request("GET", f"/v1/account/accounts/{account_id}/balance", auth=True)
        return data.get("data", {}).get("list", [])


class TelegramBot:
    def __init__(self, token: str, chat_id: int, timeout: int = 30) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id: Optional[int] = None

    def send_message(self, text: str) -> None:
        payload = {"chat_id": self.chat_id, "text": text}
        for attempt in range(2):
            try:
                resp = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok", False):
                    LOG.warning("Telegram sendMessage returned error: %s", data)
                return
            except Exception as exc:
                if attempt == 0:
                    time.sleep(1)
                    continue
                LOG.warning("Telegram send_message failed after retry: %s", exc)

    def get_updates(self) -> List[Dict]:
        params = {"timeout": self.timeout}
        if self.last_update_id is not None:
            params["offset"] = self.last_update_id + 1
        try:
            resp = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=self.timeout + 5)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok", False):
                LOG.warning("Telegram getUpdates returned error: %s", data)
                return []
            return data.get("result", [])
        except Exception as exc:
            LOG.warning("Telegram get_updates failed: %s", exc)
            return []

    def handle_updates(self, handler) -> None:
        for update in self.get_updates():
            self.last_update_id = update.get("update_id")
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            text = (message.get("text") or "").strip()
            if not text:
                continue
            if chat_id != self.chat_id:
                continue
            try:
                handler(text)
            except Exception as exc:
                LOG.warning("Command handler failed for message '%s': %s", text, exc)


def _get_timezone(name: str):
    if name.lower() == "local":
        return datetime.now().astimezone().tzinfo
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


class OrderTracker:
    def __init__(self, htx: HTXClient, bot: TelegramBot, config: Config) -> None:
        self.htx = htx
        self.bot = bot
        self.config = config
        self.last_seen: Dict[str, str] = {}
        self._seeded = False

    def _fetch_orders(self) -> List[Dict]:
        if self.config.order_source == "open":
            account_id = self.config.htx_account_id or self.htx.get_spot_account_id()
            if not account_id:
                raise RuntimeError("Unable to determine HTX spot account id; set HTX_ACCOUNT_ID")
            orders: List[Dict] = []
            for symbol in self.config.symbols:
                orders.extend(self.htx.get_open_orders(account_id, symbol, side="buy"))
            return orders
        orders: List[Dict] = []
        for symbol in self.config.symbols:
            orders.extend(self.htx.get_orders(symbol, self.config.order_states, self.config.order_types))
        return orders

    def run(self, stop_event: threading.Event) -> None:
        LOG.info("Order tracker started")
        while not stop_event.is_set():
            try:
                orders = self._fetch_orders()
                if not self._seeded:
                    for order in orders:
                        order_id = str(order.get("id"))
                        state = order.get("state", "")
                        self.last_seen[order_id] = state
                    self._seeded = True
                    LOG.info("Order tracker seeded with %d existing orders", len(orders))
                    stop_event.wait(self.config.order_poll_seconds)
                    continue
                for order in orders:
                    order_id = str(order.get("id"))
                    state = order.get("state", "")
                    if order_id not in self.last_seen:
                        self.last_seen[order_id] = state
                        self.bot.send_message("New Buy Order\n" + _order_text(order))
                    elif self.last_seen[order_id] != state:
                        self.last_seen[order_id] = state
                        self.bot.send_message("Order Updated\n" + _order_text(order))
            except Exception as exc:
                LOG.warning("Order tracker error: %s", exc)
            stop_event.wait(self.config.order_poll_seconds)


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}" if value < 1 else f"{value:.2f}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_symbol(symbol: str) -> str:
    upper = (symbol or "").upper()
    if upper.endswith("USDT") and len(upper) > 4:
        return f"{upper[:-4]}/USDT"
    return upper or "N/A"


def _format_order_type(order_type: str) -> str:
    return (order_type or "").replace("-", " ").upper() or "N/A"


def _order_value_usdt(order: Dict) -> float:
    price = _safe_float(order.get("price"))
    amount = _safe_float(order.get("amount"))
    return price * amount


def _order_text(order: Dict) -> str:
    symbol = _format_symbol(order.get("symbol", ""))
    order_type = _format_order_type(order.get("type", ""))
    state = (order.get("state", "") or "").replace("-", " ").upper() or "N/A"
    price = _safe_float(order.get("price"))
    value = _order_value_usdt(order)
    return (
        f"Pair: {symbol}\n"
        f"Type: {order_type}\n"
        f"State: {state}\n"
        f"Limit Price: {price:,.2f} USDT\n"
        f"Order Value: {value:,.2f} USDT"
    )


def _compute_change(open_price: Optional[float], close_price: Optional[float]) -> Tuple[str, str]:
    if open_price is None or close_price is None or open_price == 0:
        return "n/a", "n/a"
    delta = close_price - open_price
    pct = (delta / open_price) * 100
    return f"{delta:+.2f}", f"{pct:+.2f}%"


def _trend_for_period(htx: "HTXClient", symbol: str, period: str) -> Tuple[str, str]:
    klines = htx.get_kline(symbol, period, size=1)
    if not klines:
        return "n/a", "n/a"
    kline = klines[0]
    return _compute_change(kline.get("open"), kline.get("close"))


def _symbol_for_currency(currency: str) -> str:
    return f"{currency.lower()}usdt"


def _build_ai_chat_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    lower = base.lower()
    if lower.endswith("/chat/completions"):
        return base
    if lower.endswith("/api/v1") or lower.endswith("/v1"):
        return base + "/chat/completions"
    if lower.endswith("/api"):
        return base + "/v1/chat/completions"

    parsed = urlparse(base)
    if parsed.netloc.lower() == "openrouter.ai":
        return base + "/api/v1/chat/completions"
    return base + "/v1/chat/completions"


def _call_ai_report(config: Config, facts: str) -> str:
    if not (config.ai_api_url and config.ai_api_key and config.ai_model):
        return "AI report is not configured. Set AI_API_URL, AI_API_KEY, and AI_MODEL."
    url = _build_ai_chat_url(config.ai_api_url)
    headers = {
        "Authorization": f"Bearer {config.ai_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/huanzhang/cryptoBot2026", # Required by OpenRouter
        "X-Title": "CryptoBot2026", # Optional: Title for OpenRouter
    }
    prompt = (
        "You are a crypto market assistant. Write a concise daily BTC report based only on the facts. "
        "Use 3-5 bullet points and end with a one-line takeaway. Avoid financial advice.\n\n"
        f"Facts:\n{facts}"
    )
    payload = {
        "model": config.ai_model,
        "messages": [
            {"role": "system", "content": "You produce clear, compact market summaries."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=config.ai_timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return (
            f"AI report request failed ({status}) at {url}. "
            "Check AI_API_URL and AI_MODEL."
        )
    except Exception as exc:
        return f"AI report request failed: {exc}"

    choices = data.get("choices") or []
    if not choices:
        return "AI report API returned no choices."
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        return "AI report API returned empty content."
    return content


class DailyReporter:
    def __init__(self, htx: HTXClient, bot: TelegramBot, config: Config) -> None:
        self.htx = htx
        self.bot = bot
        self.config = config
        self.tz = _get_timezone(config.timezone)
        self.last_sent_date: Optional[str] = None

    def _build_facts(self) -> str:
        tick = self.htx.get_market_detail("btcusdt")
        open_price = tick.get("open")
        close_price = tick.get("close")
        high = tick.get("high")
        low = tick.get("low")
        vol = tick.get("vol")
        delta, pct = _compute_change(open_price, close_price)
        return (
            f"BTC 24h open: {open_price}\n"
            f"BTC 24h close: {close_price}\n"
            f"BTC 24h high: {high}\n"
            f"BTC 24h low: {low}\n"
            f"BTC 24h volume: {vol}\n"
            f"Change: {delta} ({pct})"
        )

    def _should_send_now(self, now: datetime) -> bool:
        try:
            hour, minute = [int(x) for x in self.config.daily_report_time.split(":", 1)]
        except Exception:
            hour, minute = 9, 0
        if now.hour != hour or now.minute != minute:
            return False
        date_key = now.strftime("%Y-%m-%d")
        if self.last_sent_date == date_key:
            return False
        self.last_sent_date = date_key
        return True

    def run(self, stop_event: threading.Event) -> None:
        LOG.info("Daily reporter started")
        while not stop_event.is_set():
            try:
                now = datetime.now(self.tz)
                if self._should_send_now(now):
                    facts = self._build_facts()
                    report = _call_ai_report(self.config, facts)
                    self.bot.send_message(report)
            except Exception as exc:
                LOG.warning("Daily reporter error: %s", exc)
            stop_event.wait(30)


def _fetch_prices(htx: HTXClient, symbols: List[str]) -> str:
    lines = []
    for symbol in symbols:
        tick = htx.get_market_detail(symbol)
        close_price = tick.get("close")
        delta1, pct1 = _trend_for_period(htx, symbol, "60min")
        delta4, pct4 = _trend_for_period(htx, symbol, "4hour")
        lines.append(
            f"{symbol.upper()}: {_format_price(close_price)} | 1h {delta1} ({pct1}) | 4h {delta4} ({pct4})"
        )
    return "\n".join(lines)


def _fetch_orders_text(tracker: OrderTracker) -> str:
    orders = tracker._fetch_orders()
    if not orders:
        return "No buy orders found."
    lines: List[str] = []
    for order in orders[:10]:
        lines.append(_order_text(order))
    return "\n\n".join(lines)


def _ensure_account_id(htx: HTXClient, config: Config) -> str:
    account_id = config.htx_account_id or htx.get_spot_account_id()
    if not account_id:
        raise RuntimeError("Unable to determine HTX spot account id; set HTX_ACCOUNT_ID")
    return account_id


def _price_map(htx: HTXClient, symbols: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for symbol in symbols:
        if not symbol:
            continue
        try:
            tick = htx.get_market_detail(symbol)
        except Exception as exc:
            LOG.info("Skip unsupported symbol '%s' for pricing: %s", symbol, exc)
            continue
        price = tick.get("close")
        if price is not None:
            prices[symbol] = float(price)
    return prices


def _open_orders_text(htx: HTXClient, config: Config) -> str:
    account_id = _ensure_account_id(htx, config)
    orders: List[Dict] = []
    for symbol in config.symbols:
        # Include both buy and sell to cover all open spot orders
        orders.extend(htx.get_open_orders(account_id, symbol, side=""))
    if not orders:
        return "No open spot orders."

    lines: List[str] = []
    for order in orders:
        lines.append(_order_text(order))
    return "\n\n".join(lines)


def _history_text(htx: HTXClient, config: Config) -> str:
    account_id = _ensure_account_id(htx, config)
    now_ms = int(time.time() * 1000)
    week_ago_ms = now_ms - 7 * 24 * 3600 * 1000
    states = ["filled", "partial-filled", "partial-canceled", "canceled"]
    orders: List[Dict] = []
    for symbol in config.symbols:
        orders.extend(htx.get_orders(symbol, states, config.order_types, size=200))
    recent = [o for o in orders if o.get("created-at", 0) >= week_ago_ms]
    if not recent:
        return "No orders in the last 7 days."

    prices = _price_map(htx, list({o.get("symbol", "") for o in recent if o.get("symbol")}))
    lines = []
    total_money = config.total_money or 0
    for order in sorted(recent, key=lambda o: o.get("created-at", 0), reverse=True):
        symbol = order.get("symbol", "")
        price = float(order.get("price") or 0)
        amount = float(order.get("amount") or 0)
        side = order.get("type", "")
        created = datetime.fromtimestamp(order.get("created-at", 0) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        current_price = prices.get(symbol)
        pnl = None
        if current_price is not None:
            if side.startswith("buy"):
                pnl = (current_price - price) * amount
            elif side.startswith("sell"):
                pnl = (price - current_price) * amount
        value_usd = (current_price or 0) * amount
        percent = f"{(value_usd / total_money * 100):.2f}%" if total_money else "n/a"
        pnl_txt = f"{pnl:+.2f} USDT" if pnl is not None else "n/a"
        lines.append(
            f"{created} {symbol.upper()} {side} price={price} amount={amount} now={_format_price(current_price)} "
            f"value_usdt={value_usd:.2f} pnl_vs_now={pnl_txt} share_of_total={percent}"
        )
    return "\n".join(lines)


def _percent_text(htx: HTXClient, config: Config) -> str:
    account_id = _ensure_account_id(htx, config)
    balances = htx.get_account_balance(account_id)
    if not balances:
        return "No balances found."

    # Aggregate available + frozen per currency
    amounts: Dict[str, float] = {}
    for bal in balances:
        currency = bal.get("currency")
        if not currency:
            continue
        amt = float(bal.get("balance") or 0)
        amounts[currency.lower()] = amounts.get(currency.lower(), 0.0) + amt

    symbols = [ _symbol_for_currency(c) for c in amounts.keys() if c != "usdt" ]
    prices = _price_map(htx, symbols)
    total_money = config.total_money or 0

    total_asset_usd = 0.0
    lines = []
    for cur, amt in sorted(amounts.items()):
        if cur == "usdt":
            price = 1.0
        else:
            symbol = _symbol_for_currency(cur)
            price = prices.get(symbol)
        if price is None:
            value_usd = 0.0
        else:
            value_usd = price * amt
        total_asset_usd += value_usd
        share_total_money = f"{(value_usd / total_money * 100):.2f}%" if total_money else "n/a"
        lines.append(
            f"{cur.upper()}: amount={amt:.8f} vs_usdt_value={value_usd:.2f} share_of_total_money={share_total_money}"
        )

    total_vs_money = f"{(total_asset_usd / total_money * 100):.2f}%" if total_money else "n/a"
    lines.append(f"Total assets in USDT: {total_asset_usd:.2f}; total/total_money={total_vs_money}")
    lines.append("Note: amount is coin units; vs_usdt_value is that amount converted with real-time price to USDT.")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv()

    total_money_val = os.getenv("TOTAL_MONEY") or os.getenv("TOTALMONEY") or "0"
    try:
        total_money_float = float(total_money_val)
    except ValueError:
        total_money_float = 0.0

    config = Config(
        telegram_token=_env("TELEGRAM_BOT_TOKEN", required=True),
        telegram_chat_id=int(_env("TELEGRAM_CHAT_ID", required=True)),
        telegram_poll_timeout=int(_env("TELEGRAM_POLL_TIMEOUT", "30")),
        htx_access_key=_env("HTX_ACCESS_KEY", required=True),
        htx_secret_key=_env("HTX_SECRET_KEY", required=True),
        htx_base_url=_env("HTX_BASE_URL", "https://api.huobi.pro"),
        htx_account_id=_env("HTX_ACCOUNT_ID", None),
        symbols=[s.lower() for s in _parse_csv(_env("SYMBOLS", "btcusdt,ethusdt,adausdt"))],
        order_source=_env("ORDER_SOURCE", "open").lower(),
        order_states=_parse_csv(_env("ORDER_STATES", "submitted,partial-filled,filled")),
        order_types=_parse_csv(
            _env(
                "ORDER_TYPES",
                "buy-market,buy-limit,buy-ioc,buy-stop-limit,buy-limit-fok,buy-stop-limit-fok,"
                "sell-market,sell-limit,sell-ioc,sell-stop-limit,sell-limit-fok,sell-stop-limit-fok",
            )
        ),
        order_poll_seconds=int(_env("ORDER_POLL_SECONDS", "60")),
        daily_report_time=_env("DAILY_REPORT_TIME", "09:00"),
        timezone=_env("TIMEZONE", "local"),
        ai_api_url=_env("AI_API_URL", None),
        ai_api_key=_env("AI_API_KEY", None),
        ai_model=_env("AI_MODEL", None),
        ai_timeout=int(_env("AI_TIMEOUT", "30")),
        total_money=total_money_float,
    )

    htx = HTXClient(config.htx_access_key, config.htx_secret_key, config.htx_base_url)
    bot = TelegramBot(config.telegram_token, config.telegram_chat_id, config.telegram_poll_timeout)
    tracker = OrderTracker(htx, bot, config)
    reporter = DailyReporter(htx, bot, config)

    stop_event = threading.Event()

    order_thread = threading.Thread(target=tracker.run, args=(stop_event,), daemon=True)
    order_thread.start()

    report_thread = threading.Thread(target=reporter.run, args=(stop_event,), daemon=True)
    report_thread.start()

    LOG.info("Bot is running. Commands: /prices /orders /history /percent /report")

    def on_command(text: str) -> None:
        if text.startswith("/prices"):
            bot.send_message(_fetch_prices(htx, config.symbols))
        elif text.startswith("/orders"):
            bot.send_message(_open_orders_text(htx, config))
        elif text.startswith("/history"):
            bot.send_message(_history_text(htx, config))
        elif text.startswith("/percent"):
            bot.send_message(_percent_text(htx, config))
        elif text.startswith("/report"):
            facts = reporter._build_facts()
            # Add short-term trends for context
            trends = _fetch_prices(htx, ["btcusdt"])
            report = _call_ai_report(config, facts + "\nShort-term: " + trends)
            bot.send_message(report)
        else:
            bot.send_message("Commands: /prices /orders /history /percent /report")

    try:
        while True:
            try:
                bot.handle_updates(on_command)
            except Exception as exc:
                LOG.warning("Main loop polling error: %s", exc)
                time.sleep(2)
    except KeyboardInterrupt:
        LOG.info("Shutting down")
        stop_event.set()


if __name__ == "__main__":
    main()
