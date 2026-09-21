# -*- coding: utf-8 -*-
"""账号系统：本地多用户账号管理。

账号数据保存在 %APPDATA%\MatLite\ 下：
    accounts.json           所有账号的元信息（用户名 + 加盐哈希）
    users/<用户名>/         每个用户的专属目录（配置/历史/数据）

密码用 PBKDF2-HMAC-SHA256 加盐哈希存储，不保存明文。
"""

import hashlib
import json
import os
import secrets
import time

APP_DIR_NAME = "MatLite"
PBKDF2_ITER = 120_000
ILLEGAL_CHARS = set('\\/:*?"<>|')


def _app_data_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, APP_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _hash_pw(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"),
        bytes.fromhex(salt_hex), PBKDF2_ITER,
    ).hex()


def _new_salt():
    return secrets.token_hex(16)


class AccountError(Exception):
    """账号相关的业务错误（信息会直接展示给用户）。"""


class AccountManager:
    """负责账号的增删查、登录状态和用户目录。"""

    def __init__(self):
        self.base = _app_data_dir()
        self.users_dir = os.path.join(self.base, "users")
        self.meta_path = os.path.join(self.base, "accounts.json")
        os.makedirs(self.users_dir, exist_ok=True)
        self.current_user = None      # None 表示游客
        self._meta = self._load_meta()

    # ---------------- 元数据 ----------------
    def _load_meta(self):
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": [], "remember": None}

    def _save_meta(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    def list_users(self):
        return [u["username"] for u in self._meta["users"]]

    def has_accounts(self):
        return bool(self._meta["users"])

    def is_guest(self):
        return self.current_user is None

    def user_dir(self, username=None):
        name = username or self.current_user
        return os.path.join(self.users_dir, name)

    def remember_username(self):
        return self._meta.get("remember")

    # ---------------- 账号操作 ----------------
    def register(self, username, password, confirm=None):
        username = (username or "").strip()
        if not username:
            raise AccountError("用户名不能为空。")
        if len(username) > 20:
            raise AccountError("用户名最多 20 个字符。")
        if any(ch in ILLEGAL_CHARS or ch.isspace() for ch in username):
            raise AccountError("用户名不能包含空格或 \\ / : * ? \" < > | 等字符。")
        if any(u["username"] == username for u in self._meta["users"]):
            raise AccountError(f"账号「{username}」已存在，请直接登录。")
        if len(password) < 4:
            raise AccountError("密码至少 4 位。")
        if confirm is not None and password != confirm:
            raise AccountError("两次输入的密码不一致。")

        salt = _new_salt()
        self._meta["users"].append({
            "username": username,
            "salt": salt,
            "hash": _hash_pw(password, salt),
            "created": time.strftime("%Y-%m-%d %H:%M"),
        })
        self._save_meta()
        os.makedirs(self.user_dir(username), exist_ok=True)
        os.makedirs(os.path.join(self.user_dir(username), "history"), exist_ok=True)
        return True

    def verify(self, username, password):
        for u in self._meta["users"]:
            if u["username"] == username:
                return secrets.compare_digest(
                    u["hash"], _hash_pw(password, u["salt"]))
        return False

    def login(self, username, remember=True):
        if not any(u["username"] == username for u in self._meta["users"]):
            raise AccountError(f"账号「{username}」不存在。")
        self.current_user = username
        self._meta["remember"] = username if remember else None
        self._save_meta()
        return True

    def guest(self):
        """进入游客模式：不记录历史，但可用全部功能。"""
        self.current_user = None
        self._meta["remember"] = None
        self._save_meta()
        return True

    def logout(self):
        self.current_user = None
        return True