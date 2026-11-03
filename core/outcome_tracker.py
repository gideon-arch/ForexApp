# core/outcome_tracker.py
# v5.5.0 — Signal outcome tracking: records every fired signal, watches
# subsequent candles for a TP or SL touch, and keeps a running win/loss
# record per asset. Built so "is the win rate actually good" stops being a
# matter of scrolling screenshots and becomes a real, queryable number.
#
# Storage: Redis only (already part of the stack) — no new database, no
# new service. Each tracked signal is one Redis hash; a sorted set per
# asset (plus one "all") tracks open signal IDs for fast iteration.
#
# Resolution logic: on every monitor cycle, for each OPEN tracked signal on
# that asset, scan the candles fetched THIS cycle that are newer than the
# signal's open time, in chronological order, for the first touch of SL or
# TP. If a single candle's range contains both levels (a fast, violent
# move), the signal is resolved as a LOSS — the conservative assumption,
# so the tracker never overstates how well the bot is doing.
# Signals open longer than OUTCOME_TIMEOUT_HOURS are closed as EXPIRED
# (excluded from win-rate math — neither a win nor a loss, just inconclusive).

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class OutcomeTracker:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.timeout_hours = float(os.getenv("OUTCOME_TIMEOUT_HOURS", "48"))
        self.enabled = redis_client is not None
        logger.info(
            f"OutcomeTracker v5.5.3 | Enabled: {self.enabled} | "
            f"Timeout: {self.timeout_hours}h"
        )

    def _key(self, signal_id: str) -> str:
        return f"outcome:{signal_id}"

    def _open_set_key(self, asset: str) -> str:
        return f"outcome:open:{asset}"

    def _closed_list_key(self, asset: str) -> str:
        return f"outcome:closed:{asset}"

    async def record_signal(self, signal_data: Dict) -> Optional[str]:
        """Called right after a signal is sent. Stores it as OPEN."""
        if not self.enabled:
            return None
        try:
            asset = signal_data["asset"]
            signal_id = f"{asset}_{signal_data['timestamp']}"
            record = {
                "signal_id": signal_id,
                "asset": asset,
                "display_name": signal_data.get("display_name", asset),
                "direction": signal_data["signal"],
                "entry_price": signal_data["entry_price"],
                "stop_loss": signal_data["stop_loss"],
                "take_profit": signal_data["take_profit"],
                "confidence": signal_data["confidence"],
                "is_early": signal_data.get("is_early", False),
                "context": signal_data.get("context") or {},
                "opened_at": signal_data["timestamp"],
                "status": "OPEN",
                "closed_at": None,
                "close_price": None,
                "r_multiple": None,
            }
            await self.redis.set(self._key(signal_id), json.dumps(record))
            await self.redis.sadd(self._open_set_key(asset), signal_id)
            logger.info(f"OUTCOME TRACKED [{asset}] {signal_id}: {signal_data['signal']} "
                        f"@ {signal_data['entry_price']}")
            return signal_id
        except Exception as e:
            logger.error(f"OutcomeTracker.record_signal error: {e}")
            return None

    async def check_outcomes(self, asset: str, df: Optional[pd.DataFrame]) -> List[Dict]:
        """
        Called each monitor cycle with the freshly-fetched candle data for
        that asset. Checks every OPEN tracked signal for that asset against
        the new candles. Returns a list of newly-resolved records (for
        optional Telegram notification), or [] if nothing resolved.
        """
        if not self.enabled or df is None or len(df) == 0:
            return []

        resolved = []
        try:
            open_ids = await self.redis.smembers(self._open_set_key(asset))
            for signal_id in open_ids:
                raw = await self.redis.get(self._key(signal_id))
                if not raw:
                    await self.redis.srem(self._open_set_key(asset), signal_id)
                    continue
                record = json.loads(raw)
                opened_at = pd.to_datetime(record["opened_at"])

                # Timeout check first — don't bother scanning candles for a
                # signal that's aged out regardless of outcome.
                age_hours = (datetime.utcnow() - opened_at.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
                if age_hours >= self.timeout_hours:
                    record["status"] = "EXPIRED"
                    record["closed_at"] = datetime.utcnow().isoformat()
                    await self._close(asset, signal_id, record)
                    resolved.append(record)
                    continue

                new_candles = df[df["timestamp"] > opened_at]
                if new_candles.empty:
                    continue

                direction = record["direction"]
                sl = record["stop_loss"]
                tp = record["take_profit"]
                entry = record["entry_price"]

                outcome = None
                close_price = None
                for _, candle in new_candles.iterrows():
                    hit_tp = candle["high"] >= tp if direction == "BUY" else candle["low"] <= tp
                    hit_sl = candle["low"] <= sl if direction == "BUY" else candle["high"] >= sl
                    if hit_tp and hit_sl:
                        # Both touched in the same candle — can't know which
                        # came first from OHLC alone. Assume the loss, so the
                        # tracker never overstates the win rate.
                        outcome, close_price = "LOSS", sl
                        break
                    elif hit_sl:
                        outcome, close_price = "LOSS", sl
                        break
                    elif hit_tp:
                        outcome, close_price = "WIN", tp
                        break

                if outcome:
                    risk = abs(entry - sl)
                    reward = abs(close_price - entry)
                    r_multiple = round((reward / risk) if outcome == "WIN" else -(reward / risk), 2) if risk > 0 else 0.0
                    record["status"] = outcome
                    record["closed_at"] = datetime.utcnow().isoformat()
                    record["close_price"] = close_price
                    record["r_multiple"] = r_multiple
                    await self._close(asset, signal_id, record)
                    resolved.append(record)
                    logger.info(f"OUTCOME RESOLVED [{asset}] {signal_id}: {outcome} (R: {r_multiple:+.2f})")
        except Exception as e:
            logger.error(f"OutcomeTracker.check_outcomes error for {asset}: {e}")

        return resolved

    async def _close(self, asset: str, signal_id: str, record: Dict):
        await self.redis.set(self._key(signal_id), json.dumps(record))
        await self.redis.srem(self._open_set_key(asset), signal_id)
        await self.redis.lpush(self._closed_list_key(asset), signal_id)
        await self.redis.ltrim(self._closed_list_key(asset), 0, 499)  # keep last 500/asset

    async def get_stats(self, asset: Optional[str] = None, since: Optional[str] = None,
                        breakdown: bool = False) -> Dict:
        """
        Win rate / avg R, overall or for one asset.
        `since`: optional ISO date/datetime string (e.g. "2026-10-04" or a
        full timestamp). Only signals OPENED at or after this point count —
        lets you mark "the bot stabilized here" and exclude earlier trades
        from versions that have since been fixed, instead of them silently
        dragging the win rate down forever.
        """
        if not self.enabled:
            return {"enabled": False}

        since_dt = None
        if since:
            try:
                since_dt = pd.to_datetime(since)
            except Exception:
                return {"enabled": True, "error": f"Could not parse 'since' value: {since!r}. "
                                                    f"Use a date like 2026-10-04 or a full ISO timestamp."}

        assets = [asset] if asset else await self._all_tracked_assets()
        per_asset = {}
        all_records = []

        for a in assets:
            ids = await self.redis.lrange(self._closed_list_key(a), 0, -1)
            records = []
            for sid in ids:
                raw = await self.redis.get(self._key(sid))
                if raw:
                    rec = json.loads(raw)
                    if since_dt is not None and pd.to_datetime(rec["opened_at"]) < since_dt:
                        continue
                    records.append(rec)
            per_asset[a] = self._summarize(records)
            all_records.extend(records)

        overall = self._summarize(all_records)
        result = {"enabled": True, "overall": overall, "by_asset": per_asset}
        if since:
            result["since"] = since
        if breakdown:
            result["breakdown"] = self._breakdowns(all_records)
        return result

    async def _all_tracked_assets(self) -> List[str]:
        assets = set()
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match="outcome:closed:*", count=100)
            for k in keys:
                assets.add(k.split("outcome:closed:", 1)[1])
            if cursor == 0:
                break
        return sorted(assets)

    def _summarize(self, records: List[Dict]) -> Dict:
        decided = [r for r in records if r["status"] in ("WIN", "LOSS")]
        wins = [r for r in decided if r["status"] == "WIN"]
        losses = [r for r in decided if r["status"] == "LOSS"]
        expired = [r for r in records if r["status"] == "EXPIRED"]

        total = len(decided)
        win_rate = round(len(wins) / total * 100, 1) if total else None
        avg_r = round(sum(r["r_multiple"] for r in decided) / total, 2) if total else None
        avg_win_r = round(sum(r["r_multiple"] for r in wins) / len(wins), 2) if wins else None
        avg_loss_r = round(sum(r["r_multiple"] for r in losses) / len(losses), 2) if losses else None

        return {
            "total_closed": total,
            "wins": len(wins),
            "losses": len(losses),
            "expired": len(expired),
            "win_rate_pct": win_rate,
            "avg_r_multiple": avg_r,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
        }

    def _slice(self, records: List[Dict], keyfn) -> Dict:
        buckets: Dict[str, List[Dict]] = {}
        for r in records:
            if r.get("status") not in ("WIN", "LOSS"):
                continue
            buckets.setdefault(keyfn(r), []).append(r)
        out = {}
        for k in sorted(buckets):
            s = self._summarize(buckets[k])
            out[k] = {"n": s["total_closed"], "wins": s["wins"],
                      "win_rate_pct": s["win_rate_pct"], "avg_r": s["avg_r_multiple"]}
        return out

    def _breakdowns(self, records: List[Dict]) -> Dict:
        """
        v5.8.0: win rate sliced by the features recorded when each signal
        fired. Signals from before v5.8.0 have no context and land in the
        "no_data" bucket, so they never get mixed into a yes/no comparison.
        Small buckets (n under ~15) are noise — read them as hints, not facts.
        """
        NO = "no_data(old signal)"

        def flag(name):
            def f(r):
                ctx = r.get("context") or {}
                return NO if name not in ctx else ("yes" if ctx[name] else "no")
            return f

        def stop_vs_atr(r):
            v = (r.get("context") or {}).get("sl_atr_ratio")
            if v is None:
                return NO
            return "a) stop < 0.75x ATR" if v < 0.75 else ("b) 0.75-1.25x ATR" if v < 1.25 else "c) > 1.25x ATR")

        def confidence(r):
            c = r.get("confidence", 0) or 0
            return "a) < 60%" if c < 0.60 else ("b) 60-80%" if c < 0.80 else "c) 80%+")

        def session(r):
            h = pd.to_datetime(r["opened_at"]).hour
            return ("a) 00-08 UTC (Asia)" if h < 8 else "b) 08-13 UTC (Europe)" if h < 13
                    else "c) 13-21 UTC (US)" if h < 21 else "d) 21-24 UTC (late)")

        return {
            "primary_layer_opposes_trade": self._slice(records, flag("primary_opposes")),
            "range_chop_flagged": self._slice(records, flag("range_chop")),
            "stop_size_vs_5m_ATR": self._slice(records, stop_vs_atr),
            "displayed_confidence": self._slice(records, confidence),
            "session": self._slice(records, session),
            "direction": self._slice(records, lambda r: r.get("direction", "?")),
        }