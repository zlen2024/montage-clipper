"""Credit-gating tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.credits import CreditError, CreditService
from app.core.jobs import Tier


def test_ai_tier_blocked_without_credits(tmp_path: Path):
    svc = CreditService(tmp_path / "a.json", free_daily_limit=5)
    with pytest.raises(CreditError):
        svc.authorize("user", Tier.AI)


def test_ai_tier_allowed_with_credits(tmp_path: Path):
    svc = CreditService(tmp_path / "a.json", free_daily_limit=5)
    svc.grant_credits("user", 1)
    svc.authorize("user", Tier.AI)  # does not raise


def test_free_daily_limit_enforced(tmp_path: Path):
    svc = CreditService(tmp_path / "a.json", free_daily_limit=2)
    svc.authorize("user", Tier.FREE)
    svc.authorize("user", Tier.FREE)
    with pytest.raises(CreditError):
        svc.authorize("user", Tier.FREE)


def test_bypass_skips_all_checks(tmp_path: Path):
    svc = CreditService(tmp_path / "a.json", free_daily_limit=1, bypass=True)
    # Would normally exhaust the free quota, then 402 on AI — bypass lets it all through.
    for _ in range(5):
        svc.authorize("user", Tier.FREE)
        svc.authorize("user", Tier.AI)
