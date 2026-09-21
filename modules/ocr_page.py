# -*- coding: utf-8 -*-
"""拍照识题：导入题目图片 -> 本地视觉模型转录+解析数学表达式 -> 可修正 -> 调用计算/绘图。

复用自治 AI 页的本地 Ollama 连接（base_url/mode/model/api_key 存于 %APPDATA%\MatLite\config.json）。
"""

import base64
import io
import json
import os
import threading

import requests
import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_NO_PROXY = {"http": None, "https": None}
CONFIG_NAME = "config.json"

PROMPT = (
    "你是一位数学助教。下面是一张数学题目图片，请完成两件事：\n"
    "1. 完整转录图片中的题目文字（尽量忠实原样）：\n"
    "2. 把题目里要计算的数学表达式用 Python 语法改写出来，\n"
    "   例如 x^2 写成 x**2，积分(exp(x), x) 写成 integrate(exp(x), x)。\n"
    "请严格按下面格式输出，不要输出多余解释：\n"
    "题目原文：<转录的题目文字>\n"
    "表达式：<Python 语法的表达式或无>\n"
    "备注：<简短的解题思路或注意事项，没有就写无>"
)


def _compile(expr, var="x"):
    """校验并编译一个表达式函数，返回 (ok, 函数或错误信息)。"""
    try:
        import sympy as sp
        e = sp.sympify(expr.replace("^", "**"), local_dict={"pi": sp.pi, "e": sp.E, "I": sp.I})
        f = sp.lambdify(sp.symbols(var), e, "numpy")
        return True, (e, f)
    except Exception as err:
        return False, str(err)


def _split_response(text):
    """从模型返回中提取 题目原文 / 表达式 / 备注 三段。"""
    out = {"题目原文": "", "表达式": "", "备注": ""}
    cur = None
    for line in text.splitlines():
        s = line.strip()
        for key in out:
            if s.startswith(key):
                cur = key
                out[key] = s[len(key):].lstrip("： ").strip()
                break
        else:
            if cur and s:
                out[cur] += ("\n" if out[cur] else "") + s
    return out


class OcrPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.image = None
        self._busy = False

        # 读取 AI 连接配置
        base_dir = os.environ.get("APPDATA") or os.path.expanduser("~")
        cfg_dir = os.path.join(base_dir, "MatLite")
        os.makedirs(cfg_dir, exist_ok=True)
        self.cfg_path = os.path.join(cfg_dir, CONFIG_NAME)
        self.cfg = self._load_config()

        # ================= 左：图片与设置 =================
        left = ctk.CTkScrollableFrame(self, width=390, corner_radius=12, label_text="题目图片")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(6, 4))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(btn_row, text="📁 选择图片", height=36, command=self._pick).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(btn_row, text="📋 粘贴截图", height=36, fg_color="gray40",
                      command=self._paste).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.img_lab = ctk.CTkLabel(left, text="尚未选择图片", font=ctk.CTkFont(size=13),
                                    text_color="gray50", anchor="n")
        self.img_lab.grid(row=1, column=0, padx=12, pady=(4, 8), sticky="n")

        ctk.CTkLabel(left, text="本地 AI 服务地址（自动读取）", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=4, pady=(4, 2))
        self.base = ctk.CTkEntry(left, height=32)
        self.base.grid(row=3, column=0, sticky="ew", padx=12)
        self.base.insert(0, self.cfg.get("base_url", "http://localhost:11434"))

        ctk.CTkLabel(left, text="模型名称（自动读取，可改）", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="w", padx=12, pady=(8, 2))
        self.model = ctk.CTkEntry(left, height=32)
        self.model.grid(row=5, column=0, sticky="ew", padx=12)
        self.model.insert(0, self.cfg.get("model", "qwen3-vl:8b"))

        ctk.CTkButton(left, text="🔍 识别并解析", height=42, command=self._run).grid(
            row=6, column=0, sticky="ew", padx=12, pady=(12, 4))

        ctk.CTkLabel(left, text="解析出的表达式（可编辑修正）", font=ctk.CTkFont(size=12)).grid(
            row=7, column=0, sticky="w", padx=14, pady=(10, 2))
        self.expr = ctk.CTkTextbox(left, height=64, font=ctk.CTkFont(family="Consolas", size=13))
        self.expr.grid(row=8, column=0, sticky="ew", padx=12, pady=(2, 6))

        action = ctk.CTkFrame(left, fg_color="transparent")
        action.grid(row=9, column=0, sticky="ew", padx=12, pady=(4, 4))
        for i, (txt, fn) in enumerate([("求导", self._diff), ("积分", self._integ), ("画图", self._plot_expr)]):
            ctk.CTkButton(action, text=txt, height=36, width=80, fg_color="gray40",
                          command=fn).grid(row=0, column=i, padx=4, sticky="ew")
        action.grid_columnconfigure((0, 1, 2), weight=1)

        tip = ("使用说明：\n"
               "① 点「选择图片」或「粘贴截图」导入题目；\n"
               "② 点「识别并解析」调用本地视觉模型转录并转成 Python 表达式；\n"
               "③ 在表达式框核对/修正（格式如 x**2 + 2*x + 1 或 integrate(x**2, x)）；\n"
               "④ 点「求导」「积分」或「画图」直接计算。\n"
               "注：需本机已启动 Ollama 且模型支持看图（如 qwen3-vl:8b）。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=350).grid(
            row=10, column=0, sticky="w", padx=14, pady=(6, 12))

        # ================= 右：结果 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="识别结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        box = ctk.CTkFrame(right, fg_color="transparent")
        box.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 4))
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)
        self.figure = plt.Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        toolbar = NavigationToolbar2Tk(self.canvas, box)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.ax.set_title("函数图形预览")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.canvas.draw()

    def _load_config(self):
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ---------------- 图片获取 ----------------
    def _pick(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择题目图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")])
        if path:
            self._set_image(Image.open(path))

    def _paste(self):
        try:
            from PIL import ImageGrab
            im = ImageGrab.grabclipboard()
            if im is None:
                self.out.configure(state="normal")
                self.out.insert("end", "剪贴板里没有图片。\n")
                self.out.configure(state="disabled")
                return
            self._set_image(im)
        except Exception as e:
            self._msg(f"粘贴失败：{e}")

    def _set_image(self, im):
        self.image = im.convert("RGB")
        im2 = self.image.copy()
        im2.thumbnail((280, 280))
        cimg = ctk.CTkImage(light_image=im2, size=(im2.width, im2.height))
        self.img_lab.configure(image=cimg, text="")
        self.img_lab._image = cimg

    # ---------------- 识别 ----------------
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.cfg.get("api_key"):
            h["Authorization"] = f"Bearer {self.cfg['api_key']}"
        return h

    def _run(self):
        if self._busy:
            return
        if self.image is None:
            self._msg("请先选择或粘贴一张题目图片。\n")
            return
        model = self.model.get().strip()
        if not model:
            self._msg("模型名为空，请填写或读取配置。\n")
            return
        self._busy = True
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")
        self._msg("正在调用本地视觉模型识别……（可能需数秒）")
        threading.Thread(target=self._worker, args=(model,), daemon=True).start()

    def _image_b64(self):
        buf = io.BytesIO()
        im = self.image.copy()
        im.thumbnail((1280, 1280))
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _worker(self, model):
        try:
            base = self.base.get().strip().rstrip("/") or "http://localhost:11434"
            url = f"{base}/api/generate"
            payload = {
                "model": model,
                "prompt": PROMPT,
                "images": [self._image_b64()],
                "stream": False,
            }
            r = requests.post(url, headers=self._headers(), json=payload, timeout=300, proxies=_NO_PROXY)
            r.raise_for_status()
            data = r.json()
            text = data.get("response") or ""
            self.after(0, self._done, text)
        except Exception as e:
            self.after(0, self._fail, str(e))

    def _done(self, text):
        self._busy = False
        parsed = _split_response(text)
        self._msg("【识别结果】\n" + text + "\n\n")
        expr = (parsed.get("表达式") or "").strip()
        if expr:
            ok, info = _compile(expr)
            if ok:
                self._msg(f"✅ 表达式可解析：{expr}\n")
                self.expr.delete("1.0", "end")
                self.expr.insert("1.0", expr)
            else:
                self.expr.delete("1.0", "end")
                self.expr.insert("1.0", expr)
                self._msg(f"⚠️ 解析校验失败：{info}\n请修正后使用下方按钮。\n")
        else:
            self._msg("未能提取到表达式，请检查图片清晰度或手动输入。\n")

    def _fail(self, err):
        self._busy = False
        self._msg(f"❌ 识别失败：{err}\n请确认 Ollama 已启动、模型名正确、地址可达。\n")

    def _msg(self, s):
        self.out.configure(state="normal")
        self.out.insert("end", s)
        self.out.see("end")
        self.out.configure(state="disabled")

    # ---------------- 对解析出的表达式做计算/绘图 ----------------
    def _get_expr(self):
        return self.expr.get("1.0", "end").strip()

    def _diff(self):
        expr = self._get_expr()
        if not expr:
            self._msg("表达式为空。\n")
            return
        try:
            import sympy as sp
            x = sp.symbols("x")
            e = sp.sympify(expr.replace("^", "**"), {})
            d = sp.diff(e, x)
            self._msg(f"d/dx  {expr}  =  {d}\n")
            self.expr.delete("1.0", "end")
            self.expr.insert("1.0", str(d))
            self._render(lambda v: float(sp.lambdify(x, d, "numpy")(v)))
        except Exception as err:
            self._msg(f"求导失败：{err}\n")

    def _integ(self):
        expr = self._get_expr()
        if not expr:
            self._msg("表达式为空。\n")
            return
        try:
            import sympy as sp
            x = sp.symbols("x")
            e = sp.sympify(expr.replace("^", "**"), {})
            i = sp.integrate(e, x)
            self._msg(f"∫ {expr} dx  =  {i}  + C\n")
            self.expr.delete("1.0", "end")
            self.expr.insert("1.0", str(i))
        except Exception as err:
            self._msg(f"积分失败：{err}\n")

    def _plot_expr(self):
        expr = self._get_expr()
        if not expr:
            self._msg("表达式为空。\n")
            return
        try:
            import sympy as sp
            x = sp.symbols("x")
            e = sp.sympify(expr.replace("^", "**"), {})
            f = sp.lambdify(x, e, "numpy")
            self._render(f)
        except Exception as err:
            self._msg(f"绘图失败：{err}\n")

    def _render(self, f):
        xs = np.linspace(-10, 10, 500)
        with np.errstate(all="ignore"):
            ys = np.array([float(f(v)) for v in xs])
        mask = np.isfinite(ys)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(xs[mask], ys[mask], lw=2, color="steelblue")
        self.ax.axhline(0, color="gray", lw=0.8)
        self.ax.axvline(0, color="gray", lw=0.8)
        self.ax.grid(True)
        self.ax.set_title("表达式图形")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.save_plot("OCR 识图表达式", self.figure, extra=self._get_expr())

    def on_show(self):
        pass
