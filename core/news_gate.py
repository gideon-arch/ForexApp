# core/news_gate.py
# v5.9.0 — USD high-impact ("red folder") news blackout, for prop-firm rules.
#
# DATA SOURCE: ForexFactory has no official API. This reads the free weekly
# JSON feed that mirrors its calendar (default: Fair Economy's
# nfs.faireconomy.media/ff_calendar_thisweek.json — change it any time with the
# NEWS_FEED_URL setting). That feed is rate limited per IP, so it is fetched at
# most once an hour and the last good copy is kept (in memory and in Redis).
#
# WHAT IT CAN AND CANNOT DO (read this):
#   * It can stop NEW signals / NEW trades before and during the news window.
#   * It can tell the EA to CLOSE its own trades just before the window starts
#     (the "flatten" window), so nothing is open when news hits.
#   * It can NOT pause a stop-loss or take-profit. Those live on the broker's
#     server and fire by themselves. The only way to avoid one firing inside the
#     window is to be flat before the window starts.
#   * It only knows SCHEDULED events. Surprise headlines are invisible to it.
#
# TIMELINE for one event at time T (defaults, all adjustable by settings):
#     T-38  no NEW signals/trades start being blocked   (NEWS_NO_NEW_TRADES_LEAD_MIN=30 before the window)
#     T-10  flatten window: EA closes its own open trades (NEWS_FLATTEN_LEAD_MIN=2 before the window)
#     T-8   BLACKOUT starts                               (NEWS_BLOCK_BEFORE_MIN=8)
#     T+8   BLACKOUT ends                                 (NEWS_BLOCK_AFTER_MIN=8)
#
# FAIL-SAFE: if the calendar can't be loaded, NEWS_FAIL_MODE decides.
#   closed (default) = block new trades until it's back (protects the account)
#   open             = carry on trading without news protection

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str) -> datetime:
    """ISO timestamp WITH a UTC offset -> aware UTC datetime. No offset = rejected (can't know the zone)."""
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"no timezone in {s!r}")
    return dt.astimezone(timezone.utc)


class NewsGate:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.enabled = os.getenv("NEWS_GATE_ENABLED", "true").lower() == "true"
        self.feed_url = os.getenv("NEWS_FEED_URL", DEFAULT_FEED)
        self.before = float(os.getenv("NEWS_BLOCK_BEFORE_MIN", "8"))
        self.after = float(os.getenv("NEWS_BLOCK_AFTER_MIN", "8"))
        self.no_new_lead = float(os.getenv("NEWS_NO_NEW_TRADES_LEAD_MIN", "30"))
        self.flatten_lead = float(os.getenv("NEWS_FLATTEN_LEAD_MIN", "2"))
        self.fail_mode = os.getenv("NEWS_FAIL_MODE", "closed").lower()
        self.refresh_minutes = float(os.getenv("NEWS_REFRESH_MINUTES", "60"))
        self.retry_minutes = float(os.getenv("NEWS_RETRY_MINUTES", "10"))
        self.max_stale_hours = float(os.getenv("NEWS_MAX_STALE_HOURS", "12"))
        self.utc_offset = float(os.getenv("DISPLAY_UTC_OFFSET", "1"))   # only for display (Nigeria = +1)
        self.events: List[Dict] = []        # [{"time": aware-UTC datetime, "title": str}]
        self.last_ok: Optional[datetime] = None
        self.last_attempt: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._sent_local: set = set()
        logger.info(
            f"NewsGate | enabled={self.enabled} | window -{self.before:g}/+{self.after:g} min | "
            f"no-new-trades lead {self.no_new_lead:g} min | flatten lead {self.flatten_lead:g} min | "
            f"fail mode={self.fail_mode} | feed={self.feed_url}"
        )

    # ---------------- fetching & parsing ----------------
    @staticmethod
    def parse_events(raw) -> List[Dict]:
        """Keep USD + High impact only. Raises if the feed looks broken (so the caller keeps the old copy)."""
        if not isinstance(raw, list) or not raw:
            raise ValueError("calendar feed was empty or not a list")
        out, parsed_ok, parse_bad = [], 0, 0
        for e in raw:
            try:
                if not isinstance(e, dict):
                    continue
                t = _parse_dt(e.get("date", ""))
                parsed_ok += 1
            except Exception:
                parse_bad += 1
                continue
            if str(e.get("country", "")).upper() != "USD":
                continue
            if str(e.get("impact", "")).strip().lower() != "high":
                continue
            out.append({"time": t, "title": str(e.get("title", "USD high-impact event"))})
        if parsed_ok == 0:
            raise ValueError(f"none of {parse_bad} calendar dates could be parsed — feed format changed?")
        out.sort(key=lambda x: x["time"])
        return out

    async def _http_get_json(self):
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(self.feed_url, timeout=aiohttp.ClientTimeout(total=20),
                             headers={"User-Agent": "Mozilla/5.0 (news-gate)"}) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return await r.json(content_type=None)

    async def refresh(self, force: bool = False) -> bool:
        """Fetch (rate-limited). Returns True if we currently hold usable data."""
        now = _utcnow()
        if not self.enabled:
            return True
        wait = self.retry_minutes if self.last_error else self.refresh_minutes
        if not force and self.last_attempt and (now - self.last_attempt) < timedelta(minutes=wait):
            return self.is_data_ok(now)
        self.last_attempt = now
        try:
            events = self.parse_events(await self._http_get_json())
            self.events, self.last_ok, self.last_error = events, now, None
            logger.info(f"NewsGate: loaded {len(events)} USD high-impact events this week")
            await self._save()
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(f"NewsGate refresh failed (keeping last copy): {self.last_error}")
            return self.is_data_ok(now)

    async def _save(self):
        if not self.redis:
            return
        try:
            payload = {"fetched_at": self.last_ok.isoformat(),
                       "events": [{"time": e["time"].isoformat(), "title": e["title"]} for e in self.events]}
            await self.redis.setex("news:calendar", 8 * 86400, json.dumps(payload))
        except Exception as e:
            logger.warning(f"NewsGate: couldn't cache calendar in Redis: {e}")

    async def load_cached(self):
        """After a restart, reuse the last good calendar instead of starting blind."""
        if not self.redis:
            return
        try:
            raw = await self.redis.get("news:calendar")
            if not raw:
                return
            d = json.loads(raw)
            self.events = [{"time": _parse_dt(e["time"]), "title": e["title"]} for e in d["events"]]
            self.last_ok = _parse_dt(d["fetched_at"])
            logger.info(f"NewsGate: restored {len(self.events)} events from Redis cache ({d['fetched_at']})")
        except Exception as e:
            logger.warning(f"NewsGate: cache restore failed: {e}")

    # ---------------- the actual decision ----------------
    def is_data_ok(self, now: Optional[datetime] = None) -> bool:
        now = now or _utcnow()
        return self.last_ok is not None and (now - self.last_ok) <= timedelta(hours=self.max_stale_hours)

    def _iso(self, dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    def _hhmm(self, dt: datetime) -> str:
        local = dt + timedelta(hours=self.utc_offset)
        sign = "+" if self.utc_offset >= 0 else "-"
        return f"{dt:%H:%M} UTC ({local:%H:%M} UTC{sign}{abs(self.utc_offset):g})"

    def status(self, now: Optional[datetime] = None) -> Dict:
        now = now or _utcnow()
        out = {"enabled": self.enabled, "data_ok": True, "blocked": False, "no_new_trades": False,
               "flatten_now": False, "reason": "", "event": None, "event_time_utc": None,
               "blackout_start_utc": None, "blackout_end_utc": None, "until_utc": None,
               "next_event": None, "next_event_time_utc": None, "minutes_to_next_blackout": None}
        if not self.enabled:
            out["reason"] = "news gate disabled"
            return out
        if not self.is_data_ok(now):
            out["data_ok"] = False
            if self.fail_mode == "closed":
                out["blocked"] = out["no_new_trades"] = True
                out["reason"] = (f"News calendar unavailable ({self.last_error or 'never loaded'}) — "
                                 f"blocking new trades to protect the account")
            else:
                out["reason"] = "News calendar unavailable — trading WITHOUT news protection (fail-open)"
            return out

        before, after = timedelta(minutes=self.before), timedelta(minutes=self.after)
        lead, flat = timedelta(minutes=self.no_new_lead), timedelta(minutes=self.flatten_lead)
        driver, until = None, None
        for ev in self.events:
            start, end = ev["time"] - before, ev["time"] + after
            if start <= now <= end:                                  # inside the blackout
                out["blocked"] = out["no_new_trades"] = True
                until = max(until, end) if until else end
                driver = driver or ev
            elif start - lead <= now < start:                        # approaching: no new trades
                out["no_new_trades"] = True
                until = max(until, end) if until else end
                driver = driver or ev
                if start - flat <= now < start:                      # ... and time to be flat
                    out["flatten_now"] = True
            elif start > now and out["next_event"] is None:
                out["next_event"], out["next_event_time_utc"] = ev["title"], self._iso(ev["time"])
                out["minutes_to_next_blackout"] = round((start - now).total_seconds() / 60, 1)
        if driver:
            s, e = driver["time"] - before, driver["time"] + after
            out.update(event=driver["title"], event_time_utc=self._iso(driver["time"]),
                       blackout_start_utc=self._iso(s), blackout_end_utc=self._iso(e), until_utc=self._iso(until))
            if out["blocked"]:
                out["reason"] = f"News blackout: {driver['title']} at {self._hhmm(driver['time'])} — no trading until {self._hhmm(until)}"
            else:
                out["reason"] = (f"News approaching: {driver['title']} at {self._hhmm(driver['time'])} — "
                                 f"no NEW trades; blackout starts {self._hhmm(s)}")
        return out

    def summary_line(self, now: Optional[datetime] = None) -> str:
        st = self.status(now)
        if not st["enabled"]:
            return "📰 News gate: OFF"
        if not st["data_ok"]:
            return "📰 News calendar UNAVAILABLE — " + ("new trades BLOCKED" if st["blocked"] else "trading unprotected")
        if st["blocked"]:
            return f"📰 NEWS BLACKOUT ACTIVE — {st['event']} (until {st['until_utc'][11:16]} UTC)"
        if st["no_new_trades"]:
            return f"📰 News soon — {st['event']}: no new trades, flat before {st['blackout_start_utc'][11:16]} UTC"
        if st["next_event"]:
            m = st["minutes_to_next_blackout"]
            eta = f"{int(m // 60)}h {int(m % 60)}m" if m >= 60 else f"{int(m)}m"
            return f"📰 Next USD red-folder: {st['next_event']} {st['next_event_time_utc'][11:16]} UTC (blackout in {eta})"
        return "📰 No more USD red-folder news this week"

    # ---------------- Telegram notices (each sent once) ----------------
    async def _first_time(self, key: str, ttl: int = 86400) -> bool:
        if self.redis:
            try:
                return bool(await self.redis.set(f"news:sent:{key}", "1", nx=True, ex=ttl))
            except Exception:
                pass
        if key in self._sent_local:
            return False
        self._sent_local.add(key)
        return True

    async def pop_due_notices(self, now: Optional[datetime] = None) -> List[str]:
        now = now or _utcnow()
        texts: List[str] = []
        if not (self.enabled and self.is_data_ok(now)):
            return texts
        before, after = timedelta(minutes=self.before), timedelta(minutes=self.after)
        lead, flat = timedelta(minutes=self.no_new_lead), timedelta(minutes=self.flatten_lead)
        for ev in self.events:
            start, end = ev["time"] - before, ev["time"] + after
            tag = ev["time"].isoformat()
            if start - lead <= now < start and await self._first_time(f"{tag}:headsup"):
                texts.append(
                    f"📰 USD HIGH-IMPACT NEWS AHEAD\n━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{ev['title']} at {self._hhmm(ev['time'])}\n"
                    f"• No new signals or trades from now\n"
                    f"• Blackout: {self._hhmm(start)} → {self._hhmm(end)}\n"
                    f"• Close any open trade BEFORE {self._hhmm(start)} — a stop-loss or "
                    f"take-profit that fires inside the window can break prop-firm rules")
            if start - flat <= now < start and await self._first_time(f"{tag}:flatten"):
                texts.append(f"⏱ FLAT NOW — blackout starts at {self._hhmm(start)} ({ev['title']}).\n"
                             f"The EA is closing its own trades; close any manual trades yourself.")
            if end <= now < end + timedelta(minutes=10) and await self._first_time(f"{tag}:ended"):
                texts.append(f"✅ News blackout over ({ev['title']}). Signals resume.")
        return texts

    # ---------------- helper for the post-news breakout setup ----------------
    def clusters_matching(self, keywords: List[str], gap_min: float = 5.0) -> List[Dict]:
        """Events whose title contains any keyword (case-insensitive). Releases within
        gap_min of each other (CPI m/m + Core CPI + CPI y/y all drop at the same minute)
        become ONE cluster, timed at the first of them."""
        kws = [k.lower() for k in keywords if k]
        evs = sorted((e for e in self.events if any(k in e["title"].lower() for k in kws)), key=lambda e: e["time"])
        clusters: List[Dict] = []
        for e in evs:
            if clusters and (e["time"] - clusters[-1]["last"]) <= timedelta(minutes=gap_min):
                clusters[-1]["titles"].append(e["title"])
                clusters[-1]["last"] = e["time"]
            else:
                clusters.append({"time": e["time"], "last": e["time"], "titles": [e["title"]]})
        return [{"time": c["time"], "title": " + ".join(dict.fromkeys(c["titles"]))} for c in clusters]