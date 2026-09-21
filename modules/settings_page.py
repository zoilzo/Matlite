# -*- coding: utf-8 -*-
"""设置页：全局参数调节（主题/小数位/表格行数/绘图DPI/字体缩放）+ 公告与更新提醒。

布局：CTkScrollableFrame + pack，设置全部排成一列、可滚动，小窗口也不会挤掉任何选项。
"""

import customtkinter as ctk

from modules import app_settings as S

THEMES = ["浅色", "深色"]
PRECS = ["4 位", "6 位"]
ROWS = ["10 行", "20 行", "30 行", "50 行"]
DPIS = ["80", "100", "150"]
SCALES = ["0.9x", "1.0x", "1.1x", "1.2x"]


def _to_num(s, default=0.0):
    try:
        return float(s.replace("x", "").strip())
    except Exception:
        return default


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        body = ctk.CTkScrollableFrame(self, corner_radius=12)
        body.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        body.grid_columnconfigure(0, weight=1)

        def hh(label):
            """小节标题（用默认字体加粗，间距统一）。"""
            ctk.CTkLabel(body, text=label, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="gray30").pack(anchor="w", pady=(10, 4))

        ctk.CTkLabel(body, text="⚙️ 全局设置", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(4, 2))

        hh("界面主题")
        self.theme = ctk.CTkOptionMenu(body, values=THEMES, width=160)
        self.theme.pack(anchor="w", pady=(0, 4))
        self.theme.set("深色" if S.get("theme", "浅色") == "深色" else "浅色")
        self.theme.configure(command=lambda _v: self._apply_theme())

        hh("结果小数位数")
        self.prec = ctk.CTkOptionMenu(body, values=PRECS, width=160)
        self.prec.pack(anchor="w", pady=(0, 4))
        prec = int(S.get("precision", 6))
        self.prec.set(f"{prec} 位" if prec in (4, 6) else "6 位")

        hh("数据预览默认行数")
        self.rows = ctk.CTkOptionMenu(body, values=ROWS, width=160)
        self.rows.pack(anchor="w", pady=(0, 4))
        self.rows.set(f"{int(S.get('table_rows', 10))} 行")

        hh("绘图分辨率 DPI")
        self.dpi = ctk.CTkOptionMenu(body, values=DPIS, width=160)
        self.dpi.pack(anchor="w", pady=(0, 4))
        self.dpi.set(str(int(S.get("plot_dpi", 100))))

        hh("界面字体缩放")
        self.scale = ctk.CTkOptionMenu(body, values=SCALES, width=160)
        self.scale.pack(anchor="w", pady=(0, 4))
        self.scale.set(f"{float(S.get('font_scale', 1.0)):.1f}x")

        hh("公告与更新提醒")
        self.announce_on = ctk.CTkCheckBox(body, text="接收公告与更新提醒（仅拉取，不上传任何数据）")
        self.announce_on.pack(anchor="w", pady=(0, 4))
        if S.get("announce_on", True):
            self.announce_on.select()
        self.announce_url = ctk.CTkEntry(body, height=32, placeholder_text="公告地址（留空=关闭）")
        self.announce_url.pack(fill="x", pady=(0, 10))
        self.announce_url.insert(0, S.get("announce_url", ""))

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(4, 8))
        ctk.CTkButton(btns, text="💾 保存设置", height=36, command=self._save).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(btns, text="恢复默认", height=36, fg_color="gray40", command=self._reset).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        self.status = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=13), text_color="#2a7f5c")
        self.status.pack(anchor="w", pady=(0, 4))

    def _apply_theme(self):
        ctk.set_appearance_mode("dark" if self.theme.get() == "深色" else "light")
        self.status.configure(text="主题已切换 ✓（保存后永久生效）", text_color="#2a7f5c")

    def _save(self):
        S.set_val("theme", self.theme.get())
        S.set_val("precision", int(_to_num(self.prec.get(), 6)))
        S.set_val("table_rows", int(_to_num(self.rows.get(), 10)))
        S.set_val("plot_dpi", int(_to_num(self.dpi.get(), 100)))
        S.set_val("font_scale", _to_num(self.scale.get(), 1.0))
        S.set_val("announce_on", bool(self.announce_on.get()))
        S.set_val("announce_url", self.announce_url.get().strip())
        self._apply_theme()
        self.status.configure(text="设置已保存 ✓ 下次启动自动生效", text_color="#2a7f5c")

    def _reset(self):
        for k, v in S.DEFAULTS.items():
            S.set_val(k, v)
        self.theme.set("浅色")
        self.prec.set(f"{int(S.DEFAULTS['precision'])} 位")
        self.rows.set(f"{int(S.DEFAULTS['table_rows'])} 行")
        self.dpi.set(str(int(S.DEFAULTS['plot_dpi'])))
        self.scale.set(f"{float(S.DEFAULTS['font_scale']):.1f}x")
        if S.get("announce_on", True):
            self.announce_on.select()
        self.announce_url.delete(0, "end")
        self.announce_url.insert(0, S.DEFAULTS["announce_url"])
        self._apply_theme()
        self.status.configure(text="已恢复默认设置 ✓", text_color="#2a7f5c")