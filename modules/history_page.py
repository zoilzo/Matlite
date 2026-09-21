# -*- coding: utf-8 -*-
"""历史记录页：按类型浏览、搜索、回看（LaTeX/图片预览）、导出、清空。"""

import json
import os

import customtkinter as ctk
from tkinter import messagebox

from modules.latex_view import render_latex_text, matrix_to_image

KINDS = [
    ("计算历史", "calc.jsonl"),
    ("矩阵历史", "matrix.jsonl"),
    ("分析历史", "analysis.jsonl"),
    ("绘图历史", "plots.jsonl"),
    ("操作日志", "op_log.jsonl"),
]

_FETCH = {
    "calc.jsonl": "calc_history",
    "matrix.jsonl": "matrix_history",
    "analysis.jsonl": "analysis_history",
    "plots.jsonl": "plot_history",
    "op_log.jsonl": "op_log",
}


def _one_line(rec):
    """把一条记录压缩成一行摘要。"""
    k = rec.get("kind")
    if k == "calc":
        return f"{rec.get('time','')}  [{rec.get('mode','')}] {str(rec.get('expr',''))[:42]}"
    if k == "matrix":
        return f"{rec.get('time','')}  [{rec.get('op','')}] → {str(rec.get('result',''))[:42]}"
    if k == "analysis":
        return f"{rec.get('time','')}  [{rec.get('type','')}] {str(rec.get('detail',''))[:42]}"
    if k == "plot":
        return f"{rec.get('time','')}  📊 {str(rec.get('title',''))[:50]}"
    return f"{rec.get('time','')}  [{rec.get('page','')}] {str(rec.get('action',''))[:30]} {str(rec.get('detail',''))[:30]}"


class HistoryPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._kind = KINDS[0][1]
        self._records = []

        # ===== 左：控制 =====
        left = ctk.CTkFrame(self, width=310, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="历史记录类型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 6))
        seg = ctk.CTkSegmentedButton(left, values=[k[0] for k in KINDS],
                                     command=lambda v: self._switch(v))
        seg.grid(row=1, column=0, sticky="ew", padx=12)
        seg.set(KINDS[0][0])

        ctk.CTkLabel(left, text="关键词搜索（可留空）", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=14, pady=(12, 2))
        self.search = ctk.CTkEntry(left, height=34, placeholder_text="如：sin 或 回归")
        self.search.grid(row=3, column=0, sticky="ew", padx=12)
        self.search.bind("<Return>", lambda _e: self._refresh())

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.grid(row=4, column=0, sticky="ew", padx=12, pady=10)
        btns.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btns, text="刷新", height=34, command=self._refresh).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(btns, text="导出", height=34, fg_color="gray40",
                      command=self._export).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkButton(left, text="🗑 清空当前类型", height=34, fg_color="#8a3b3b",
                      command=self._clear).grid(row=5, column=0, sticky="ew", padx=12)

        tip = ("说明：\n· 登录用户的历史自动保存；\n· 游客模式不记录；\n"
               "· 每个类型最多保留最近 500 条；\n· 点右侧一条记录可回看详情。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=270).grid(
            row=6, column=0, sticky="w", padx=14, pady=(8, 12))

        # ===== 右：列表 + 详情 =====
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=2)
        right.grid_rowconfigure(3, weight=1)

        head = ctk.CTkFrame(right, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="🗂 历史记录", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w")
        self.count_lab = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=12), text_color="gray50")
        self.count_lab.grid(row=0, column=1, sticky="e")

        self.list_frame = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 4))
        self.list_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="详情", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=14, pady=(6, 2))
        self.detail = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=12), height=130)
        self.detail.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 4))
        self.detail.configure(state="disabled")
        self.img_lab = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=12), text_color="gray50")
        self.img_lab.grid(row=4, column=0, sticky="w", padx=14, pady=(0, 12))
        self._img_ref = None

    # ================= 数据 =================
    def _store(self):
        if hasattr(self, "history"):
            return self.history
        if hasattr(self, "app") and hasattr(self.app, "history"):
            return self.app.history
        return None

    def _file(self):
        for name, fname in KINDS:
            if name == self._kind:
                return fname
        return KINDS[0][1]

    def on_show(self):
        self._refresh()

    def _switch(self, name):
        self._kind = name
        self._refresh()

    def _refresh(self):
        store = self._store()
        if store is None or not store.enabled:
            self.count_lab.configure(text="0 条")
            self._clear_list()
            ctk.CTkLabel(self.list_frame, text="游客模式不记录历史。\n注册/登录后，这里会自动显示你的历史记录。",
                         font=ctk.CTkFont(size=13), text_color="gray50").grid(row=0, column=0, pady=30)
            return
        recs = getattr(store, _FETCH[self._file()])()
        kw = self.search.get().strip().lower()
        if kw:
            recs = [r for r in recs if kw in json.dumps(r, ensure_ascii=False).lower()]
        self._records = recs
        self.count_lab.configure(text=f"共 {len(recs)} 条")
        self._clear_list()
        if not recs:
            ctk.CTkLabel(self.list_frame, text="（暂无记录）", font=ctk.CTkFont(size=13),
                         text_color="gray50").grid(row=0, column=0, pady=30)
            return
        for i, rec in enumerate(recs):
            btn = ctk.CTkButton(self.list_frame, text=_one_line(rec), anchor="w", height=32,
                                font=ctk.CTkFont(size=12), fg_color="transparent",
                                text_color="black", hover_color="#e8e8ee",
                                command=lambda r=rec: self._show_detail(r))
            btn.grid(row=i, column=0, sticky="ew", pady=1)

    def _clear_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()

    # ================= 详情 =================
    def _show_detail(self, rec):
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        for k, v in rec.items():
            self.detail.insert("end", f"{k}: {v}\n")
        self.detail.configure(state="disabled")
        self.img_lab.configure(image=None, text="")
        self._img_ref = None
        try:
            if rec.get("kind") == "plot" and rec.get("file"):
                store = self._store()
                if store and store.dir:
                    full = os.path.join(store.dir, rec["file"])
                    if os.path.exists(full):
                        self._show_image(full)
            elif rec.get("matrix"):
                from sympy import Matrix
                img = matrix_to_image(Matrix(rec["matrix"]))
                self._show_pil(img)
            elif rec.get("latex"):
                img = render_latex_text(rec["latex"])
                self._show_pil(img)
        except Exception:
            pass

    def _show_image(self, path):
        from PIL import Image
        self._show_pil(Image.open(path))

    def _show_pil(self, img):
        scale = min(1.0, 180 / img.height)
        w = max(1, int(img.width * scale))
        h = max(1, int(img.height * scale))
        self._img_ref = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.img_lab.configure(image=self._img_ref, text="")

    # ================= 导出 / 清空 =================
    def _export(self):
        store = self._store()
        if store is None or not store.enabled:
            return
        path = store.export_all(self._file())
        messagebox.showinfo("导出完成", f"已导出到：\n{path}")

    def _clear(self):
        store = self._store()
        if store is None or not store.enabled:
            return
        if not messagebox.askyesno("确认清空", f"确定要清空当前「{self._kind}」的所有记录吗？"):
            return
        store.clear(self._file())
        self._refresh()