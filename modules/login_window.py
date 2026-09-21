# -*- coding: utf-8 -*-
"""登录 / 注册 / 游客模式窗口。登录成功后由 app.py 打开主窗口。"""

import customtkinter as ctk

from modules.account import AccountManager, AccountError

APP_TITLE = "MatLite 数学工作台"
LOGIN, REGISTER, GUEST = "login", "register", "guest"


class LoginWindow(ctk.CTk):
    def __init__(self, manager: AccountManager):
        super().__init__()
        self.manager = manager
        self.success = False          # 登录/注册/游客 任一种成功都为 True
        self.title(f"{APP_TITLE} - 账号")
        self.geometry("430x600")
        self.resizable(False, False)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(self, text="🧮 MatLite", font=ctk.CTkFont(size=30, weight="bold")).grid(
            row=0, column=0, pady=(34, 0))
        ctk.CTkLabel(self, text="数 学 工 作 台", font=ctk.CTkFont(size=14),
                     text_color="gray60").grid(row=1, column=0)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=2, column=0, sticky="nsew", padx=48, pady=14)
        self.body.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Made by zoilzo & Claude", font=ctk.CTkFont(size=11),
                     text_color="gray").grid(row=3, column=0, pady=(0, 12))

        # 首次启动（没有任何账号）→ 直接进注册；否则进登录
        if not manager.has_accounts():
            self._show_register(first=True)
        else:
            self._show_login()

    # ---------------- 工具 ----------------
    def _clear(self):
        for w in self.body.winfo_children():
            w.destroy()

    def _title(self, text, row):
        ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 10))

    def _label(self, text, row):
        ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, sticky="w", pady=(6, 2))

    def _entry(self, row, show=None, default=""):
        e = ctk.CTkEntry(self.body, height=38, font=ctk.CTkFont(size=14), show=show)
        e.grid(row=row, column=0, sticky="ew", pady=(0, 2))
        if default:
            e.insert(0, default)
        return e

    def _err(self, text=""):
        lab = ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=12),
                           text_color="#c0392b", anchor="w")
        lab.grid(row=99, column=0, sticky="w", pady=(6, 0))
        return lab

    def _link(self, text, row, command):
        ctk.CTkButton(self.body, text=text, fg_color="transparent", hover_color="#e8e8ee",
                      text_color="#2d6fbf", font=ctk.CTkFont(size=12), height=28,
                      command=command).grid(row=row, column=0, sticky="w")

    # ---------------- 登录 ----------------
    def _show_login(self):
        self._clear()
        self._title("登 录", 0)
        self._label("用户名", 1)
        self.u = self._entry(2, default=self.manager.remember_username() or "")
        self._label("密码", 3)
        self.p = self._entry(4, show="●")
        self.remember = ctk.CTkCheckBox(self.body, text="记住我", width=80)
        self.remember.grid(row=5, column=0, sticky="w", pady=(4, 4))
        self.remember.select()
        ctk.CTkButton(self.body, text="登 录", height=42, command=self._do_login).grid(
            row=6, column=0, sticky="ew", pady=(8, 2))
        self.err = self._err()
        self._link("还没有账号？点这里注册", 101, self._show_register)
        self._link("以游客身份使用（不记录历史）", 102, self._do_guest)
        self.p.bind("<Return>", lambda _e: self._do_login())
        self.u.bind("<Return>", lambda _e: self._do_login())

    def _do_login(self):
        u = self.u.get().strip()
        p = self.p.get()
        if not u or not p:
            self.err.configure(text="请输入用户名和密码。")
            return
        if self.manager.verify(u, p):
            self.manager.login(u, remember=bool(self.remember.get()))
            self.success = True
            self.destroy()
        else:
            self.err.configure(text="用户名或密码错误。")

    # ---------------- 注册 ----------------
    def _show_register(self, first=False):
        self._clear()
        if first:
            ctk.CTkLabel(self.body, text="🎉 首次使用：创建本机第一个账号",
                         font=ctk.CTkFont(size=12), text_color="#2a7f5c").grid(
                row=0, column=0, sticky="w", pady=(0, 6))
            r0 = 1
        else:
            self._title("注 册 新 账 号", 0)
            r0 = 1
        self._label("用户名（登录用，中文也可以）", r0)
        self.u = self._entry(r0 + 1)
        self._label("密码（至少 4 位）", r0 + 2)
        self.p = self._entry(r0 + 3, show="●")
        self._label("确认密码", r0 + 4)
        self.p2 = self._entry(r0 + 5, show="●")
        ctk.CTkButton(self.body, text="创 建 账 号", height=42, command=self._do_register).grid(
            row=r0 + 6, column=0, sticky="ew", pady=(8, 2))
        self.err = self._err()
        if not first:
            self._link("已有账号？返回登录", 101, self._show_login)
        self.p2.bind("<Return>", lambda _e: self._do_register())

    def _do_register(self):
        try:
            self.manager.register(self.u.get(), self.p.get(), self.p2.get())
            self.manager.login(self.u.get().strip(), remember=True)
            self.success = True
            self.destroy()
        except AccountError as e:
            self.err.configure(text=str(e))

    # ---------------- 游客 ----------------
    def _do_guest(self):
        self.manager.guest()
        self.success = True
        self.destroy()