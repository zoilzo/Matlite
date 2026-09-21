# -*- coding: utf-8 -*-
"""意见反馈：可填写 SMTP 发件设置，在软件内直接发送到收件人（开发者）邮箱。

若未填发件邮箱/授权码，则自动降级为 本地保存 + 复制剪贴板 + 打开邮件程序。
发件凭据保存在本机 config.json，仅用于发送本反馈邮件，不上传到任何服务器。
"""

import json
import os
import threading
import smtplib
import urllib.parse
import webbrowser
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText

import customtkinter as ctk

from modules import app_settings as _S

APP_VERSION = _S.APP_VERSION
DEFAULT_EMAIL = "17588853057@163.com"
CATEGORIES = ["建议", "Bug 反馈", "功能需求", "其他"]

# 常见邮箱的 SMTP 服务器与端口（SSL 直连）
_SMTP_HINTS = {
    "@163.com": ("smtp.163.com", 465),
    "@126.com": ("smtp.126.com", 465),
    "@qq.com": ("smtp.qq.com", 465),
    "@foxmail.com": ("smtp.qq.com", 465),
    "@gmail.com": ("smtp.gmail.com", 465),
    "@outlook.com": ("smtp-mail.outlook.com", 465),
}


def _smtp_for(email):
    low = (email or "").lower()
    for key, val in _SMTP_HINTS.items():
        if low.endswith(key):
            return val
    return ("smtp." + low.split("@")[-1], 465)


def _send_smtp(sender, auth, server, port, to_addr, subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = sender
    msg["To"] = to_addr
    sv = smtplib.SMTP_SSL(server, int(port), timeout=10)
    sv.login(sender, auth)
    sv.sendmail(sender, [to_addr], msg.as_string())
    sv.quit()


class FeedbackDialog(ctk.CTkToplevel):
    def __init__(self, master, history=None):
        super().__init__(master)
        self.title("📮 意见反馈")
        self.resizable(True, True)
        self._fit_window_to_screen()
        self.transient(master)
        self.history = history
        self.cfg = _S.load()
        self.email = self.cfg.get("feedback_email", DEFAULT_EMAIL)
        self._busy = False
        self._build()
        # 窗口映射后才能实测到真实边框高度，此时再校正一次位置，
        # 避免首次用估算值导致窗口略微超出屏幕、底部按钮被截掉
        self.after(120, self._refit_after_map)
        self.after(100, self.focus_set)

    def _refit_after_map(self):
        try:
            if self.winfo_exists() and self.winfo_ismapped():
                self._fit_window_to_screen()
        except Exception:
            pass

    def _work_area_rect(self):
        """返回当前窗口所在显示器的工作区 (x, y, w, h)，物理像素、已扣任务栏。"""
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        try:
            import ctypes

            user32 = ctypes.windll.user32

            class _RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            # 取窗口实际所在显示器（多屏时不会误判到主屏）
            mon = user32.MonitorFromWindow(self.winfo_id(), 2)  # DEFAULTTONEAREST
            if mon:
                class _MI(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("rcMonitor", _RECT),
                                ("rcWork", _RECT), ("dwFlags", ctypes.c_uint)]

                mi = _MI()
                mi.cbSize = ctypes.sizeof(_MI)
                if user32.GetMonitorInfoW(mon, ctypes.byref(mi)):
                    return (mi.rcWork.l, mi.rcWork.t,
                            mi.rcWork.r - mi.rcWork.l, mi.rcWork.b - mi.rcWork.t)

            rc = _RECT()
            # SPI_GETWORKAREA：主显示器扣除任务栏后的可用区域
            if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rc), 0):
                return (rc.l, rc.t, rc.r - rc.l, rc.b - rc.t)
        except Exception:
            pass
        return (0, 0, sw, sh)

    def _window_chrome(self):
        """实测窗口外框相对客户区多出的 (顶部, 底部) 高度，物理像素。

        系统度量接口在 DPI 虚拟化下会返回不可信的值，所以这里用真实窗口量出来。
        量不到（窗口还没实现）时返回保守默认值，后续会再校正一次。
        """
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = self.winfo_id()
            frame = user32.GetParent(hwnd) or hwnd

            class _RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            rc = _RECT()
            client_h = self.winfo_height()
            if client_h > 1 and user32.GetWindowRect(frame, ctypes.byref(rc)):
                top = self.winfo_rooty() - rc.t
                bottom = rc.b - (self.winfo_rooty() + client_h)
                if 0 <= top < 300 and 0 <= bottom < 200:
                    return (top, bottom)
        except Exception:
            pass
        return (38, 9)  # Windows 11 125% 缩放下的典型值

    def _fit_window_to_screen(self, want_w=640, want_h=760):
        """把窗口收进屏幕工作区并居中，确保底部“发送反馈”按钮始终可见。

        两个必须同时处理的换算，否则高 DPI / 矮屏上窗口会比屏幕还高，
        钉在底部的按钮被顶出可视区，用户怎么拉伸都看不到：
        ① CTk 的 geometry() 宽高是“逻辑像素”，内部再乘系统缩放系数；坐标 x/y 不缩放。
        ② +y 定位的是窗口外框顶部，标题栏和边框是叠加在客户区之外的额外高度。
        """
        work_x, work_y, work_w, work_h = self._work_area_rect()
        try:
            scale = float(self._get_window_scaling())
        except Exception:
            scale = 1.0
        if scale < 0.5:
            scale = 1.0

        chrome_top, chrome_bottom = self._window_chrome()

        # 允许的客户区高度（换算回逻辑像素），并设下限避免窗口被压得过小
        avail_h = max(320.0, (work_h - chrome_top - chrome_bottom) / scale)
        avail_w = max(420.0, work_w / scale)

        w = int(min(want_w, avail_w))
        h = int(min(want_h, avail_h))

        phys_w = int(w * scale)
        phys_h = int(h * scale) + chrome_top + chrome_bottom
        x = work_x + max(0, (work_w - phys_w) // 2)
        y = work_y + max(0, (work_h - phys_h) // 2)

        self.geometry("%dx%d+%d+%d" % (w, h, x, y))

        # 限制最小尺寸：用 tkinter 原始接口（物理像素），避免 CTk 的 minsize()
        # 反向抬高 _current_height 而与上面的自适应结果打架；
        # 上限取当前窗口尺寸，保证设完最小值不会把矮屏上的窗口又撑大出去
        min_w = int(min(400, w * scale))
        min_h = int(min(300, h * scale))
        try:
            tkinter.Toplevel.minsize(self, min_w, min_h)
        except Exception:
            pass

    def _build(self):
        """底部按钮栏先占位（任何尺寸下都不被挤掉），中部字段放进可滚动区。"""
        # 1) 先钉住底部：按钮栏 + 结果提示，永远有空间，不会被上方字段压缩成 1px
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

        btns = ctk.CTkFrame(bottom, fg_color="transparent")
        btns.pack(side="bottom", fill="x", pady=(6, 0))
        ctk.CTkButton(btns, text="📨 发送反馈", height=36, command=self._send).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(btns, text="取消", height=36, fg_color="gray40", command=self.destroy).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        self.result = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=13),
                                   text_color="#2a7f5c", wraplength=580, justify="left", anchor="w")
        self.result.pack(side="bottom", fill="x", anchor="w")

        # 2) 中部字段可滚动：空间不足时出现滚动条，而不是裁掉控件
        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=16, pady=(4, 0))

        ctk.CTkLabel(body, text="📮 意见反馈", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(2, 8))

        ctk.CTkLabel(body, text="收件人邮箱", font=ctk.CTkFont(size=13)).pack(anchor="w", pady=(0, 4))
        self.email_entry = ctk.CTkEntry(body, height=32)
        self.email_entry.pack(fill="x", pady=(0, 8))
        self.email_entry.insert(0, self.email)

        ctk.CTkLabel(body, text="反馈类型", font=ctk.CTkFont(size=13), text_color="gray45").pack(anchor="w", pady=(0, 4))
        self.cat = ctk.CTkSegmentedButton(body, values=CATEGORIES)
        self.cat.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(body, text="反馈内容", font=ctk.CTkFont(size=13), text_color="gray45").pack(anchor="w", pady=(0, 4))
        self.content = ctk.CTkTextbox(body, font=ctk.CTkFont(size=13))
        self.content.pack(fill="both", expand=True, pady=(0, 8))

        self.contact = ctk.CTkEntry(body, height=32, placeholder_text="联系方式（可选，方便开发者回复你）")
        self.contact.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(body, text="SMTP 发件设置（留空=用邮件程序发送）", font=ctk.CTkFont(size=13), text_color="gray45").pack(anchor="w", pady=(0, 4))

        for label, attr, show in [("发件邮箱", "sender", None), ("授权码", "auth", "●"),
                                  ("SMTP 服务器", "server", None), ("端口", "port", None)]:
            ctk.CTkLabel(body, text=label, font=ctk.CTkFont(size=13), text_color="gray55").pack(anchor="w", pady=(0, 4))
            e = ctk.CTkEntry(body, height=32, show=show or "")
            e.pack(fill="x", pady=(0, 8))
            setattr(self, attr, e)

        self.sender.insert(0, self.cfg.get("sender_email", ""))
        self.auth.insert(0, self.cfg.get("sender_auth", ""))
        self.server.insert(0, self.cfg.get("smtp_server", _smtp_for(self.sender.get())[0]))
        self.port.insert(0, str(self.cfg.get("smtp_port", _smtp_for(self.sender.get())[1])))
        self.sender.bind("<FocusOut>", lambda _e: self._auto_smtp())
        self._auto_smtp()


    def _auto_smtp(self):
        host, port = _smtp_for(self.sender.get())
        if not self.server.get().strip():
            self.server.delete(0, "end")
            self.server.insert(0, host)
        if not self.port.get().strip():
            self.port.delete(0, "end")
            self.port.insert(0, str(port))

    def _save_cfg(self):
        items = {
            "feedback_email": self.email_entry.get().strip() or DEFAULT_EMAIL,
            "sender_email": self.sender.get().strip(),
            "sender_auth": self.auth.get(),
            "smtp_server": self.server.get().strip(),
            "smtp_port": self.port.get().strip(),
        }
        self.cfg.update(items)
        _S.update_dict(items)

    def _send(self):
        if self._busy:
            return
        content = self.content.get("1.0", "end").strip()
        if not content:
            self.result.configure(text="请先填写反馈内容。", text_color="#c0355b")
            return
        to_addr = self.email_entry.get().strip() or DEFAULT_EMAIL
        sender = self.sender.get().strip()
        auth = self.auth.get()
        server = self.server.get().strip()
        port = self.port.get().strip()
        cat = self.cat.get()
        contact = self.contact.get().strip()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = (f"【反馈类型】{cat}\n【时间】{ts}\n【联系方式】{contact or '（未填写）'}\n"
                f"【反馈内容】\n{content}\n\n—— MatLite v{APP_VERSION} 用户")
        subject = f"MatLite 反馈：{cat}"
        self._save_cfg()

        if sender and auth and server:
            self.result.configure(text="⏳ 正在发送邮件……", text_color="gray50")
            self._busy = True
            threading.Thread(target=self._send_worker,
                             args=(sender, auth, server, port, to_addr, subject, body, cat),
                             daemon=True).start()
        else:
            self._fallback(body, to_addr, subject)
            if getattr(self, "history", None):
                self.history.log_op("意见反馈", "提交反馈(本地保存)", f"{cat} → {to_addr}")

    def _send_worker(self, sender, auth, server, port, to_addr, subject, body, cat):
        try:
            _send_smtp(sender, auth, server, port, to_addr, subject, body)
            self.after(0, self._send_ok, to_addr, body, cat)
        except Exception as e:
            self.after(0, self._send_fail, str(e), body, to_addr, subject)

    def _send_ok(self, to_addr, body, cat):
        self._busy = False
        self._backup(body)
        if getattr(self, "history", None):
            self.history.log_op("意见反馈", "提交反馈(已发送)", f"{cat} → {to_addr}")
        self.result.configure(text=f"✅ 邮件已发送成功！\n收件人：{to_addr}\n（本地已留备份）", text_color="#2a7f5c")

    def _send_fail(self, err, body, to_addr, subject):
        self._busy = False
        self._fallback(body, to_addr, subject, err)

    def _backup(self, body):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        fb_dir = os.path.join(base, "MatLite", "feedback")
        os.makedirs(fb_dir, exist_ok=True)
        fname = f"feedback_{datetime.now():%Y%m%d_%H%M%S}.txt"
        try:
            with open(os.path.join(fb_dir, fname), "w", encoding="utf-8") as f:
                f.write(body)
            return os.path.join(fb_dir, fname)
        except Exception:
            return ""

    def _fallback(self, body, to_addr, subject, err=""):
        saved = self._backup(body)
        try:
            self.clipboard_clear()
            self.clipboard_append(body)
        except Exception:
            pass
        url = f"mailto:{to_addr}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        try:
            webbrowser.open(url)
        except Exception:
            pass
        msg = "✅ 反馈已处理完成：\n"
        msg += "  ① 已保存到本机" + (f"（{saved}）" if saved else "") + "；\n"
        msg += "  ② 内容已复制到剪贴板；\n"
        msg += f"  ③ 已尝试打开邮件程序（收件人：{to_addr}），粘贴后发送即可。"
        if err:
            msg += f"\n（SMTP 发送失败：{err[:200]}）"
        self.result.configure(text=msg, text_color="#2a7f5c")