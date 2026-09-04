"""Accounts: login / register / remembered session, the subscription gate,
and the admin operations. Backed by Supabase auth + a `profiles` table
(see README, "Accounts"). Sessions are remembered in the macOS keychain.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from config import SUPABASE_URL, SUPABASE_KEY, KEYCHAIN_SERVICE


class AuthError(Exception):
    pass


@dataclass
class Profile:
    id: str
    email: str
    role: str
    active: bool
    expires_at: datetime | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def licensed(self) -> bool:
        if self.is_admin:
            return True
        if not self.active:
            return False
        return self.expires_at is None or self.expires_at > datetime.now(timezone.utc)

    @property
    def status_text(self) -> str:
        if self.is_admin:
            return "admin"
        if not self.active:
            return "subscription inactive"
        if self.expires_at is None:
            return "subscription active"
        days = (self.expires_at - datetime.now(timezone.utc)).days
        return f"subscription active, {days} day(s) left" if days >= 0 else "subscription expired"


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------- keychain

def _keychain_get(account: str) -> str | None:
    r = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _keychain_set(account: str, value: str) -> None:
    subprocess.run(
        ["security", "add-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE,
         "-w", value, "-U"],
        capture_output=True, text=True,
    )


def _keychain_delete(account: str) -> None:
    subprocess.run(
        ["security", "delete-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE],
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------- client

class Accounts:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise AuthError("Supabase URL/key not configured (config.py)")
        from supabase import create_client  # imported lazily: slow
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.profile: Profile | None = None

    # --- sessions
    def restore(self) -> Profile | None:
        """Log back in from a remembered session, if any."""
        access = _keychain_get("access")
        refresh = _keychain_get("refresh")
        if not access or not refresh:
            return None
        try:
            self.client.auth.set_session(access, refresh)
            res = self.client.auth.refresh_session()
            self._remember(res.session)
            return self._load_profile(res.user.id, res.user.email)
        except Exception:
            self.forget()
            return None

    def login(self, email: str, password: str, remember: bool) -> Profile:
        try:
            res = self.client.auth.sign_in_with_password(
                {"email": email.strip(), "password": password})
        except Exception as e:
            raise AuthError(_friendly(e)) from e
        if remember and res.session:
            self._remember(res.session)
        return self._load_profile(res.user.id, res.user.email)

    def register(self, email: str, password: str) -> str:
        """Returns a message: either logged in, or 'check your email'."""
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters")
        try:
            res = self.client.auth.sign_up({"email": email.strip(), "password": password})
        except Exception as e:
            raise AuthError(_friendly(e)) from e
        if res.session is None:
            return "Account created. Confirm the email we sent, then log in."
        return "Account created."

    def logout(self) -> None:
        try:
            self.client.auth.sign_out()
        except Exception:
            pass
        self.forget()
        self.profile = None

    def forget(self) -> None:
        _keychain_delete("access")
        _keychain_delete("refresh")

    def _remember(self, session) -> None:
        if session and session.access_token and session.refresh_token:
            _keychain_set("access", session.access_token)
            _keychain_set("refresh", session.refresh_token)

    # --- profiles
    def _load_profile(self, uid: str, email: str | None) -> Profile:
        row = (self.client.table("profiles").select("*").eq("id", uid)
               .single().execute()).data
        self.profile = Profile(
            id=uid, email=row.get("email") or email or "",
            role=row.get("role", "user"), active=bool(row.get("active")),
            expires_at=_parse_ts(row.get("expires_at")),
        )
        return self.profile

    # --- admin
    def list_users(self) -> list[Profile]:
        rows = (self.client.table("profiles").select("*")
                .order("created_at", desc=True).execute()).data
        return [Profile(id=r["id"], email=r.get("email") or "", role=r.get("role", "user"),
                        active=bool(r.get("active")), expires_at=_parse_ts(r.get("expires_at")))
                for r in rows]

    def update_user(self, uid: str, *, role: str | None = None,
                    active: bool | None = None, expires_at: datetime | None | str = "keep") -> None:
        fields: dict = {}
        if role is not None:
            fields["role"] = role
        if active is not None:
            fields["active"] = active
        if expires_at != "keep":
            fields["expires_at"] = expires_at.isoformat() if isinstance(expires_at, datetime) else None
        if fields:
            self.client.table("profiles").update(fields).eq("id", uid).execute()


def _friendly(e: Exception) -> str:
    msg = str(e)
    if "Invalid login credentials" in msg:
        return "Wrong email or password"
    if "already registered" in msg or "already exists" in msg:
        return "That email already has an account"
    if "Email not confirmed" in msg:
        return "Confirm your email first (check your inbox)"
    if "getaddrinfo" in msg or "Connection" in msg or "Max retries" in msg:
        return "No connection to the account server"
    return msg[:140]
