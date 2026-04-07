"""
notifier.py
Minimal alerting hooks. Extend with SMTP/Slack as needed.
"""

from __future__ import annotations
from typing import Optional, Dict


def send_alert(subject: str, message: str, extra: Optional[Dict]=None) -> None:
    """
    Default: prints to console so monitoring runs don't fail if email/webhooks aren't configured yet.
    Replace with SMTP/Slack/etc. in your environment.
    """
    banner = "=" * 60
    print(banner)
    print("[ALERT]", subject)
    print(message)
    if extra:
        print("Extra:", extra)
    print(banner)
