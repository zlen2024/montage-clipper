"""Minimal tier/credit gating — enough to gate the AI tier and cap free usage.

No real auth or payments yet, but the seams are clean: `Account.id` is the only
thing real authentication has to supply (swap the anonymous/header id for a JWT
subject), and `credits` is the only field a future Stripe webhook needs to top up.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.jobs import Tier


@dataclass
class Account:
    id: str
    is_vip: bool = False
    credits: int = 0
    free_uses: int = 0
    free_window_start: float = field(default_factory=time.time)


_DAY_SECONDS = 24 * 60 * 60


class CreditError(Exception):
    """Raised when an account isn't allowed to run the requested tier."""


class CreditService:
    """Account store + gating logic, persisted to a single JSON file."""

    def __init__(self, path: Path, free_daily_limit: int, bypass: bool = False):
        self._path = path
        self._free_daily_limit = free_daily_limit
        self._bypass = bypass
        self._lock = threading.Lock()
        self._accounts: dict[str, Account] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._accounts = {k: Account(**v) for k, v in data.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({k: asdict(v) for k, v in self._accounts.items()}, indent=2),
            encoding="utf-8",
        )

    def _get(self, account_id: str) -> Account:
        acct = self._accounts.get(account_id)
        if acct is None:
            acct = Account(id=account_id)
            self._accounts[account_id] = acct
        # Roll the free-usage window if a day has passed.
        if time.time() - acct.free_window_start >= _DAY_SECONDS:
            acct.free_uses = 0
            acct.free_window_start = time.time()
        return acct

    def authorize(self, account_id: str, tier: Tier) -> None:
        """Raise `CreditError` if the account may not run `tier`; else consume usage."""
        if self._bypass:
            return  # dev mode: skip every check
        with self._lock:
            acct = self._get(account_id)
            if tier == Tier.FREE:
                if not acct.is_vip and acct.free_uses >= self._free_daily_limit:
                    raise CreditError(
                        f"Daily free limit reached ({self._free_daily_limit}/day). "
                        "Upgrade to VIP or come back tomorrow."
                    )
                acct.free_uses += 1
            else:  # Tier.AI
                if not acct.is_vip and acct.credits <= 0:
                    raise CreditError("AI scoring requires VIP or credits.")
                if not acct.is_vip:
                    acct.credits -= 1
            self._save()

    # --- admin/testing helpers (a Stripe webhook would call grant_credits) ---
    def grant_credits(self, account_id: str, n: int) -> None:
        with self._lock:
            self._get(account_id).credits += n
            self._save()

    def set_vip(self, account_id: str, is_vip: bool) -> None:
        with self._lock:
            self._get(account_id).is_vip = is_vip
            self._save()
