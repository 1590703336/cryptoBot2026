import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import requests

import telegram_htx_bot as botmod


class DummyResponse:
    def __init__(self, payload, http_error=None):
        self._payload = payload
        self._http_error = http_error

    def raise_for_status(self):
        if self._http_error:
            raise self._http_error

    def json(self):
        return self._payload


class FakeStopEvent:
    def __init__(self, max_waits=1):
        self.max_waits = max_waits
        self.wait_calls = 0
        self._is_set = False

    def is_set(self):
        return self._is_set

    def wait(self, _seconds):
        self.wait_calls += 1
        if self.wait_calls >= self.max_waits:
            self._is_set = True


def make_config(**overrides):
    data = {
        "telegram_token": "token",
        "telegram_chat_id": 123,
        "telegram_poll_timeout": 30,
        "htx_access_key": "ak",
        "htx_secret_key": "sk",
        "htx_base_url": "https://api.huobi.pro",
        "htx_account_id": "acc-1",
        "symbols": ["btcusdt", "ethusdt"],
        "order_source": "open",
        "order_states": ["submitted", "filled"],
        "order_types": ["buy-limit"],
        "order_poll_seconds": 1,
        "daily_report_time": "09:00",
        "timezone": "UTC",
        "ai_api_url": "https://openrouter.ai/api/v1",
        "ai_api_key": "k",
        "ai_model": "m",
        "ai_timeout": 10,
        "total_money": 1000.0,
    }
    data.update(overrides)
    return botmod.Config(**data)


class TestEnvAndHelpers(unittest.TestCase):
    def test_env_required_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                botmod._env("MISSING", required=True)

    def test_load_dotenv_reads_and_does_not_override_existing(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            fp.write("A=from-file\nB=from-file\n")
            dotenv_path = fp.name
        try:
            with patch.dict(os.environ, {"B": "already"}, clear=True):
                botmod._load_dotenv(dotenv_path)
                self.assertEqual(os.environ["A"], "from-file")
                self.assertEqual(os.environ["B"], "already")
        finally:
            os.unlink(dotenv_path)

    def test_parse_csv(self):
        self.assertEqual(botmod._parse_csv("a, b,,c "), ["a", "b", "c"])

    def test_now_utc_iso(self):
        value = botmod._now_utc_iso()
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        self.assertIsInstance(parsed, datetime)

    def test_format_and_order_helpers(self):
        self.assertEqual(botmod._format_price(None), "n/a")
        self.assertEqual(botmod._format_price(0.12), "0.120000")
        self.assertEqual(botmod._format_price(100.123), "100.12")
        self.assertEqual(botmod._safe_float("2.5"), 2.5)
        self.assertEqual(botmod._safe_float("bad", 7.0), 7.0)
        self.assertEqual(botmod._format_symbol("btcusdt"), "BTC/USDT")
        self.assertEqual(botmod._format_order_type("buy-limit"), "BUY LIMIT")
        self.assertTrue(botmod._is_invalid_symbol_error(RuntimeError("HTX error: invalid symbol")))

        order = {"symbol": "btcusdt", "type": "buy-limit", "state": "submitted", "price": "10", "amount": "2"}
        text = botmod._order_text(order)
        self.assertIn("Pair: BTC/USDT", text)
        self.assertIn("Type: BUY LIMIT", text)
        self.assertIn("State: SUBMITTED", text)
        self.assertIn("Order Value: 20.00 USDT", text)

    def test_compute_change(self):
        self.assertEqual(botmod._compute_change(None, 2), ("n/a", "n/a"))
        self.assertEqual(botmod._compute_change(10, 12), ("+2.00", "+20.00%"))

    def test_get_timezone_fallbacks(self):
        tz = botmod._get_timezone("UTC")
        self.assertIsNotNone(tz)
        tz2 = botmod._get_timezone("not-a-real-timezone")
        self.assertIsNotNone(tz2)

    def test_symbol_for_currency(self):
        self.assertEqual(botmod._symbol_for_currency("btc"), "btcusdt")


class TestHTXClient(unittest.TestCase):
    def test_sign_params_adds_signature(self):
        client = botmod.HTXClient("ak", "sk", "https://api.huobi.pro")
        with patch("telegram_htx_bot._now_utc_iso", return_value="2026-01-01T00:00:00"):
            signed = client._sign_params("GET", "/v1/test", {"b": "2", "a": "1"})
        self.assertIn("Signature", signed)
        self.assertEqual(signed["AccessKeyId"], "ak")
        self.assertEqual(signed["Timestamp"], "2026-01-01T00:00:00")

    def test_request_success(self):
        client = botmod.HTXClient("ak", "sk", "https://api.huobi.pro")
        with patch("telegram_htx_bot.requests.request", return_value=DummyResponse({"status": "ok", "data": 1})):
            data = client._request("GET", "/x", auth=False)
        self.assertEqual(data["data"], 1)

    def test_request_wraps_http_error(self):
        client = botmod.HTXClient("ak", "sk", "https://api.huobi.pro")
        with patch(
            "telegram_htx_bot.requests.request",
            return_value=DummyResponse({}, http_error=requests.HTTPError("boom")),
        ):
            with self.assertRaises(RuntimeError):
                client._request("GET", "/x", auth=False)

    def test_request_raises_htx_error_payload(self):
        client = botmod.HTXClient("ak", "sk", "https://api.huobi.pro")
        with patch("telegram_htx_bot.requests.request", return_value=DummyResponse({"status": "error", "err-msg": "bad"})):
            with self.assertRaises(RuntimeError):
                client._request("GET", "/x", auth=False)

    def test_get_spot_account_id(self):
        client = botmod.HTXClient("ak", "sk", "https://api.huobi.pro")
        with patch.object(
            client,
            "get_accounts",
            return_value=[{"type": "margin"}, {"type": "spot", "state": "working", "id": 9}],
        ):
            self.assertEqual(client.get_spot_account_id(), "9")


class TestTelegramBot(unittest.TestCase):
    def test_send_message_retries_once(self):
        bot = botmod.TelegramBot("token", 1)
        ok_resp = DummyResponse({"ok": True})
        with patch("telegram_htx_bot.requests.post", side_effect=[Exception("x"), ok_resp]) as post_mock, patch(
            "telegram_htx_bot.time.sleep"
        ) as sleep_mock:
            bot.send_message("hello")
        self.assertEqual(post_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_send_message_no_raise_when_all_fail(self):
        bot = botmod.TelegramBot("token", 1)
        with patch("telegram_htx_bot.requests.post", side_effect=[Exception("x"), Exception("y")]):
            bot.send_message("hello")

    def test_get_updates_success(self):
        bot = botmod.TelegramBot("token", 1)
        with patch("telegram_htx_bot.requests.get", return_value=DummyResponse({"ok": True, "result": [{"update_id": 1}]})):
            updates = bot.get_updates()
        self.assertEqual(len(updates), 1)

    def test_get_updates_returns_empty_on_error(self):
        bot = botmod.TelegramBot("token", 1)
        with patch("telegram_htx_bot.requests.get", side_effect=Exception("net")):
            updates = bot.get_updates()
        self.assertEqual(updates, [])

    def test_handle_updates_filters_chat_and_catches_handler_errors(self):
        bot = botmod.TelegramBot("token", 10)
        with patch.object(
            bot,
            "get_updates",
            return_value=[
                {"update_id": 1, "message": {"chat": {"id": 99}, "text": "/prices"}},
                {"update_id": 2, "message": {"chat": {"id": 10}, "text": "/ok"}},
            ],
        ):
            handler = Mock(side_effect=[RuntimeError("bad")])
            bot.handle_updates(handler)
        handler.assert_called_once_with("/ok")


class TestOrderTracker(unittest.TestCase):
    def test_fetch_orders_open_requires_account_id(self):
        config = make_config(htx_account_id=None, order_source="open")
        htx = Mock()
        htx.get_spot_account_id.return_value = None
        tracker = botmod.OrderTracker(htx, Mock(), config)
        with self.assertRaises(RuntimeError):
            tracker._fetch_orders()

    def test_fetch_orders_open(self):
        config = make_config(order_source="open", symbols=["btcusdt", "ethusdt"])
        htx = Mock()
        htx.get_open_orders.side_effect = [[{"id": 1}], [{"id": 2}]]
        tracker = botmod.OrderTracker(htx, Mock(), config)
        orders = tracker._fetch_orders()
        self.assertEqual(len(orders), 2)

    def test_fetch_orders_history_source(self):
        config = make_config(order_source="history", symbols=["btcusdt"])
        htx = Mock()
        htx.get_orders.return_value = [{"id": 1}]
        tracker = botmod.OrderTracker(htx, Mock(), config)
        orders = tracker._fetch_orders()
        self.assertEqual(orders, [{"id": 1}])

    def test_run_seeds_then_sends_new_and_updated(self):
        config = make_config(order_poll_seconds=0)
        bot = Mock()
        tracker = botmod.OrderTracker(Mock(), bot, config)
        tracker._fetch_orders = Mock(
            side_effect=[
                [{"id": 1, "state": "submitted", "symbol": "btcusdt", "type": "buy-limit", "price": "10", "amount": "1"}],
                [
                    {"id": 1, "state": "filled", "symbol": "btcusdt", "type": "buy-limit", "price": "10", "amount": "1"},
                    {"id": 2, "state": "submitted", "symbol": "ethusdt", "type": "buy-limit", "price": "20", "amount": "1"},
                ],
            ]
        )
        event = FakeStopEvent(max_waits=2)
        tracker.run(event)
        self.assertEqual(bot.send_message.call_count, 2)
        first_msg = bot.send_message.call_args_list[0].args[0]
        second_msg = bot.send_message.call_args_list[1].args[0]
        self.assertIn("Order Updated", first_msg)
        self.assertIn("New Buy Order", second_msg)


class TestAiAndReporter(unittest.TestCase):
    def test_build_ai_chat_url_variants(self):
        self.assertEqual(
            botmod._build_ai_chat_url("https://openrouter.ai"),
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            botmod._build_ai_chat_url("https://example.com/v1"),
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(
            botmod._build_ai_chat_url("https://example.com/chat/completions"),
            "https://example.com/chat/completions",
        )

    def test_call_ai_report_not_configured(self):
        config = make_config(ai_api_url=None, ai_api_key=None, ai_model=None)
        text = botmod._call_ai_report(config, "facts")
        self.assertIn("not configured", text)

    def test_call_ai_report_http_error(self):
        config = make_config(ai_api_url="https://openrouter.ai")
        err = requests.HTTPError("404")
        err.response = Mock(status_code=404)
        with patch("telegram_htx_bot.requests.post", return_value=DummyResponse({}, http_error=err)):
            text = botmod._call_ai_report(config, "facts")
        self.assertIn("failed (404)", text)

    def test_call_ai_report_success(self):
        config = make_config(ai_api_url="https://openrouter.ai")
        payload = {"choices": [{"message": {"content": "report text"}}]}
        with patch("telegram_htx_bot.requests.post", return_value=DummyResponse(payload)):
            text = botmod._call_ai_report(config, "facts")
        self.assertEqual(text, "report text")

    def test_call_ai_report_empty_choices(self):
        config = make_config(ai_api_url="https://openrouter.ai")
        with patch("telegram_htx_bot.requests.post", return_value=DummyResponse({"choices": []})):
            text = botmod._call_ai_report(config, "facts")
        self.assertIn("no choices", text)

    def test_daily_reporter_should_send_once_per_day(self):
        config = make_config(daily_report_time="10:30")
        reporter = botmod.DailyReporter(Mock(), Mock(), config)
        now = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
        self.assertTrue(reporter._should_send_now(now))
        self.assertFalse(reporter._should_send_now(now))

    def test_daily_reporter_should_send_with_invalid_time_fallback(self):
        config = make_config(daily_report_time="bad-value")
        reporter = botmod.DailyReporter(Mock(), Mock(), config)
        at_default = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
        self.assertTrue(reporter._should_send_now(at_default))

    def test_daily_reporter_build_facts(self):
        htx = Mock()
        htx.get_market_detail.return_value = {"open": 1, "close": 2, "high": 3, "low": 0.5, "vol": 123}
        reporter = botmod.DailyReporter(htx, Mock(), make_config())
        facts = reporter._build_facts()
        self.assertIn("BTC 24h open: 1", facts)
        self.assertIn("Change: +1.00 (+100.00%)", facts)

    def test_daily_reporter_run_sends_report(self):
        htx = Mock()
        bot = Mock()
        reporter = botmod.DailyReporter(htx, bot, make_config())
        reporter._should_send_now = Mock(return_value=True)
        reporter._build_facts = Mock(return_value="facts")
        event = FakeStopEvent(max_waits=1)
        with patch("telegram_htx_bot._call_ai_report", return_value="report"):
            reporter.run(event)
        bot.send_message.assert_called_once_with("report")


class TestTextFeatures(unittest.TestCase):
    def setUp(self):
        with botmod.UNSUPPORTED_PRICE_SYMBOLS_LOCK:
            botmod.UNSUPPORTED_PRICE_SYMBOLS.clear()

    def test_trend_for_period(self):
        htx = Mock()
        htx.get_kline.return_value = [{"open": 10, "close": 12}]
        self.assertEqual(botmod._trend_for_period(htx, "btcusdt", "60min"), ("+2.00", "+20.00%"))

    def test_fetch_prices(self):
        htx = Mock()
        htx.get_market_detail.return_value = {"close": 100}
        htx.get_kline.return_value = [{"open": 100, "close": 110}]
        text = botmod._fetch_prices(htx, ["btcusdt"])
        self.assertIn("BTCUSDT", text)
        self.assertIn("+10.00 (+10.00%)", text)

    def test_fetch_orders_text(self):
        tracker = Mock()
        tracker._fetch_orders.return_value = []
        self.assertEqual(botmod._fetch_orders_text(tracker), "No buy orders found.")
        tracker._fetch_orders.return_value = [{"symbol": "btcusdt", "type": "buy-limit", "state": "submitted", "price": 1, "amount": 2}]
        self.assertIn("Order Value: 2.00 USDT", botmod._fetch_orders_text(tracker))

    def test_ensure_account_id(self):
        htx = Mock()
        config = make_config(htx_account_id="abc")
        self.assertEqual(botmod._ensure_account_id(htx, config), "abc")
        config2 = make_config(htx_account_id=None)
        htx.get_spot_account_id.return_value = "zzz"
        self.assertEqual(botmod._ensure_account_id(htx, config2), "zzz")
        htx.get_spot_account_id.return_value = None
        with self.assertRaises(RuntimeError):
            botmod._ensure_account_id(htx, config2)

    def test_price_map_caches_invalid_symbol(self):
        htx = Mock()
        htx.get_market_detail.side_effect = RuntimeError("HTX error: invalid symbol")
        first = botmod._price_map(htx, ["badusdt"])
        second = botmod._price_map(htx, ["badusdt"])
        self.assertEqual(first, {})
        self.assertEqual(second, {})
        self.assertEqual(htx.get_market_detail.call_count, 1)

    def test_price_map_warns_non_symbol_errors(self):
        htx = Mock()
        htx.get_market_detail.side_effect = RuntimeError("boom")
        with patch.object(botmod.LOG, "warning") as warning_mock:
            botmod._price_map(htx, ["btcusdt"])
        warning_mock.assert_called_once()

    def test_open_orders_text(self):
        htx = Mock()
        htx.get_open_orders.return_value = [
            {"symbol": "btcusdt", "type": "buy-limit", "state": "submitted", "price": "10", "amount": "3"}
        ]
        text = botmod._open_orders_text(htx, make_config(symbols=["btcusdt"]))
        self.assertIn("Pair: BTC/USDT", text)
        self.assertIn("Order Value: 30.00 USDT", text)

    def test_history_text(self):
        htx = Mock()
        now = 2_000_000
        recent_ms = (now - 60) * 1000
        old_ms = (now - 8 * 24 * 3600) * 1000
        htx.get_orders.return_value = [
            {"symbol": "btcusdt", "type": "buy-limit", "price": "10", "amount": "2", "created-at": recent_ms},
            {"symbol": "btcusdt", "type": "buy-limit", "price": "10", "amount": "2", "created-at": old_ms},
        ]
        config = make_config(symbols=["btcusdt"], total_money=1000.0)
        with patch("telegram_htx_bot.time.time", return_value=now), patch(
            "telegram_htx_bot._price_map", return_value={"btcusdt": 11.0}
        ):
            text = botmod._history_text(htx, config)
        self.assertIn("pnl_vs_now=+2.00 USDT", text)

    def test_history_text_empty(self):
        htx = Mock()
        htx.get_orders.return_value = []
        with patch("telegram_htx_bot.time.time", return_value=2_000_000):
            text = botmod._history_text(htx, make_config(symbols=["btcusdt"]))
        self.assertEqual(text, "No orders in the last 7 days.")

    def test_percent_text(self):
        htx = Mock()
        htx.get_account_balance.return_value = [
            {"currency": "usdt", "balance": "100"},
            {"currency": "btc", "balance": "0.1"},
            {"currency": "btc", "balance": "0.2"},
        ]
        with patch("telegram_htx_bot._price_map", return_value={"btcusdt": 50000.0}):
            text = botmod._percent_text(htx, make_config(total_money=100000))
        self.assertIn("USDT: amount=100.00000000", text)
        self.assertIn("BTC: amount=0.30000000", text)
        self.assertIn("Total assets in USDT", text)


if __name__ == "__main__":
    unittest.main()
