"""Sign-in card shown before the menu, and the admin panel."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QTableWidget, QTableWidgetItem, QComboBox, QWidget, QApplication,
                             QHeaderView)

from auth import Accounts, AuthError, Profile
from ui_theme import Card, section, ACC, MUTED, RED, AMBER


class LoginWindow(Card):
    """Emits `signed_in(profile)` once a licensed (or admin) user continues."""

    signed_in = pyqtSignal(object)

    def __init__(self, accounts: Accounts | None, config_error: str | None = None):
        super().__init__(width=380, on_close=QApplication.quit)
        self.accounts = accounts
        self.profile: Profile | None = None
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)
        lay.addWidget(self.header())
        self.tag = QLabel("Sign in to start."); self.tag.setObjectName("dim"); lay.addWidget(self.tag)
        lay.addSpacing(6)

        self.form = QWidget(); f = QVBoxLayout(self.form); f.setContentsMargins(0, 0, 0, 0); f.setSpacing(10)
        self.email = QLineEdit(); self.email.setPlaceholderText("Email")
        self.password = QLineEdit(); self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.remember = QCheckBox("Keep me signed in on this Mac"); self.remember.setChecked(True)
        self.err = QLabel(""); self.err.setObjectName("err"); self.err.setWordWrap(True); self.err.hide()
        self.login_btn = QPushButton("Sign in"); self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.register_btn = QPushButton("Create an account"); self.register_btn.setObjectName("link")
        self.register_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        for w in (self.email, self.password, self.remember, self.err, self.login_btn):
            f.addWidget(w)
        f.addWidget(self.register_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.form)

        self.done = QWidget(); d = QVBoxLayout(self.done); d.setContentsMargins(0, 0, 0, 0); d.setSpacing(10)
        self.who = QLabel(""); self.who.setWordWrap(True)
        self.state = QLabel(""); self.state.setObjectName("ok"); self.state.setWordWrap(True)
        self.continue_btn = QPushButton("Continue"); self.continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.admin_btn = QPushButton("Admin panel"); self.admin_btn.setObjectName("ghost")
        self.admin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logout_btn = QPushButton("Sign out"); self.logout_btn.setObjectName("link")
        d.addWidget(self.who); d.addWidget(self.state); d.addSpacing(4)
        d.addWidget(self.continue_btn); d.addWidget(self.admin_btn)
        d.addWidget(self.logout_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.done.hide(); lay.addWidget(self.done)

        self.status = QLabel(""); self.status.setObjectName("dim"); self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.login_btn.clicked.connect(self._login)
        self.register_btn.clicked.connect(self._register)
        self.password.returnPressed.connect(self._login)
        self.continue_btn.clicked.connect(self._continue)
        self.admin_btn.clicked.connect(self._open_admin)
        self.logout_btn.clicked.connect(self.reset)
        if config_error:
            self._error(config_error)
            self.login_btn.setEnabled(False); self.register_btn.setEnabled(False)

    # ---- flows
    def try_restore(self) -> bool:
        if self.accounts is None:
            return False
        profile = self.accounts.restore()
        if profile:
            self._signed_in(profile); return True
        return False

    def _busy(self, on: bool, text: str = ""):
        for w in (self.login_btn, self.register_btn, self.email, self.password):
            w.setEnabled(not on)
        self.status.setText(text); QApplication.processEvents()

    def _error(self, text: str):
        self.err.setText(text); self.err.setVisible(bool(text))

    def _login(self):
        self._error("")
        if "@" not in self.email.text() or not self.password.text():
            self._error("Enter your email and password."); return
        self._busy(True, "Signing in…")
        try:
            profile = self.accounts.login(self.email.text(), self.password.text(), self.remember.isChecked())
        except AuthError as e:
            self._busy(False); self._error(str(e)); return
        except Exception as e:
            self._busy(False); self._error(f"Sign-in failed: {e}"[:160]); return
        self._busy(False); self._signed_in(profile)

    def _register(self):
        self._error("")
        if "@" not in self.email.text():
            self._error("Enter a valid email."); return
        self._busy(True, "Creating your account…")
        try:
            msg = self.accounts.register(self.email.text(), self.password.text())
        except AuthError as e:
            self._busy(False); self._error(str(e)); return
        except Exception as e:
            self._busy(False); self._error(f"Could not register: {e}"[:160]); return
        self._busy(False)
        self.status.setText(msg + " New accounts are inactive until an admin enables them; sign in to check.")

    def _signed_in(self, profile: Profile):
        self.profile = profile
        self.form.hide(); self.done.show(); self.status.setText("")
        self.tag.setText("Signed in.")
        self.who.setText(profile.email)
        self.state.setText(profile.status_text.capitalize())
        self.state.setObjectName("ok" if profile.licensed else "err"); self.state.setStyleSheet("")
        self.admin_btn.setVisible(profile.is_admin)
        self.continue_btn.setEnabled(profile.licensed)
        if not profile.licensed:
            self.status.setText("Ask an admin to activate your subscription, then sign in again.")
        self.adjustSize()

    def _continue(self):
        if self.profile and self.profile.licensed:
            self.signed_in.emit(self.profile); self.hide()

    def _open_admin(self):
        AdminWindow(self.accounts).exec()

    def reset(self):
        """Back to the empty form (also used for sign-out from the menu)."""
        if self.accounts:
            self.accounts.logout()
        self.profile = None
        self.done.hide(); self.form.show(); self.tag.setText("Sign in to start.")
        self.password.clear(); self.status.setText(""); self._error("")
        self.adjustSize()



def _status_of(p: Profile) -> tuple[str, str]:
    if p.is_admin: return "admin", ACC
    if not p.active: return "inactive", MUTED
    if p.expires_at and p.expires_at < datetime.now(timezone.utc): return "expired", RED
    return "active", ACC


class AdminWindow(Card):
    COLS = ("email", "status", "role", "active", "expires (YYYY-MM-DD)", "")

    def __init__(self, accounts: Accounts):
        super().__init__(width=860)
        self.accounts = accounts
        self.setMinimumHeight(520)
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 20, 24, 20); lay.setSpacing(10)
        lay.addWidget(self.header("Accounts"))
        top = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText("Search email"); self.search.setFixedWidth(260)
        self.search.textChanged.connect(self._filter)
        self.refresh_btn = QPushButton("Refresh"); self.refresh_btn.setObjectName("ghost")
        self.refresh_btn.clicked.connect(self.reload)
        top.addWidget(self.search); top.addStretch(); top.addWidget(self.refresh_btn)
        lay.addLayout(top)
        self.table = QTableWidget(0, len(self.COLS)); self.table.setHorizontalHeaderLabels([c.upper() for c in self.COLS])
        self.table.verticalHeader().setVisible(False); self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False); self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table)
        self.msg = QLabel("Tick active and set an expiry to license someone. Empty expiry = no end date.")
        self.msg.setObjectName("dim"); lay.addWidget(self.msg)
        self.reload()

    def reload(self):
        try:
            users = self.accounts.list_users()
        except Exception as e:
            self.msg.setText(f"Could not load users: {e}"[:200]); return
        self.table.setRowCount(0)
        for u in users:
            r = self.table.rowCount(); self.table.insertRow(r); self.table.setRowHeight(r, 44)
            email = QTableWidgetItem(u.email); email.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, email)
            text, colour = _status_of(u)
            st = QLabel(text); st.setStyleSheet(f"color: {colour}; font-family: SF Mono, Menlo; font-size: 11px; padding-left: 6px;")
            self.table.setCellWidget(r, 1, st)
            role = QComboBox(); role.addItems(["user", "admin"]); role.setCurrentText(u.role)
            self.table.setCellWidget(r, 2, role)
            active = QCheckBox(); active.setChecked(u.active)
            cell = QWidget(); cl = QHBoxLayout(cell); cl.setContentsMargins(0, 0, 0, 0)
            cl.setAlignment(Qt.AlignmentFlag.AlignCenter); cl.addWidget(active)
            self.table.setCellWidget(r, 3, cell)
            exp = QLineEdit(u.expires_at.strftime("%Y-%m-%d") if u.expires_at else "")
            exp.setPlaceholderText("no end date"); exp.setStyleSheet("padding: 5px 8px; border-radius: 8px;")
            self.table.setCellWidget(r, 4, exp)
            btns = QWidget(); bl = QHBoxLayout(btns); bl.setContentsMargins(4, 4, 4, 4); bl.setSpacing(6)
            plus = QPushButton("+30d"); plus.setObjectName("ghost"); plus.setStyleSheet("padding: 6px 10px;")
            save = QPushButton("Save"); save.setStyleSheet("padding: 6px 12px;")
            bl.addWidget(plus); bl.addWidget(save)
            self.table.setCellWidget(r, 5, btns)
            plus.clicked.connect(lambda _, e=exp: self._plus30(e))
            save.clicked.connect(lambda _, uid=u.id, ro=role, ac=active, e=exp, em=u.email: self._save(uid, ro, ac, e, em))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._filter(self.search.text())
        self.msg.setText(f"{len(users)} account(s)")

    def _filter(self, text: str):
        t = text.strip().lower()
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r, bool(t) and t not in self.table.item(r, 0).text().lower())

    @staticmethod
    def _plus30(edit: QLineEdit):
        base = datetime.now(timezone.utc)
        try:
            cur = datetime.strptime(edit.text().strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if cur > base: base = cur
        except ValueError:
            pass
        edit.setText((base + timedelta(days=30)).strftime("%Y-%m-%d"))

    def _save(self, uid, role: QComboBox, active: QCheckBox, exp: QLineEdit, email: str):
        text = exp.text().strip(); expires = None
        if text:
            try:
                expires = datetime.strptime(text, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=timezone.utc)
            except ValueError:
                self.msg.setText(f"{email}: expiry must be YYYY-MM-DD"); return
        try:
            self.accounts.update_user(uid, role=role.currentText(), active=active.isChecked(), expires_at=expires)
        except Exception as e:
            self.msg.setText(f"{email}: save failed: {e}"[:200]); return
        self.msg.setText(f"{email}: saved"); self.reload()

