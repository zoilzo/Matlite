# -*- coding: utf-8 -*-
"""拍照识题：导入题目图片 -> 本地视觉模型转录+解析数学表达式 -> 可修正 -> 调用计算/绘图。

复用自治 AI 页的本地 Ollama 连接（base_url/mode/model/api_key 存于 %APPDATA%\MatLite\config.json）。
"""

import base64
import io
import json
import os
import threading
import time

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
        e = sp.sympify(expr.replace("^", "**"), locals={"pi": sp.pi, "e": sp.E, "I": sp.I})
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
        self.batch = []          # 批量识别的图片列表
        self.batch_results = []  # 每张图的结果 {text, expr, ok, note}

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
        btn_row.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(btn_row, text="📁 选择图片", height=36, command=self._pick).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(btn_row, text="📋 粘贴截图", height=36, fg_color="gray40",
                      command=self._paste).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkButton(btn_row, text="📸 拍照/截图", height=36, fg_color="gray40",
                      command=self._camera).grid(row=0, column=2, sticky="ew", padx=(4, 0))

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
               "① 点「选择图片」「拍照(本机相机)」或「粘贴截图」导入题目；\n"
               "② 点「识别并解析」调用本地视觉模型转录并转成 Python 表达式；\n"
               "③ 在表达式框核对/修正（格式如 x**2 + 2*x + 1 或 integrate(x**2, x)）；\n"
               "④ 点「求导」「积分」或「画图」直接计算。\n"
               "注：需本机已启动 Ollama 且模型支持看图（如 qwen3-vl:8b）。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=350).grid(
            row=10, column=0, sticky="w", padx=14, pady=(6, 12))

        # ---- 批量识别 / 错题本 ----
        ctk.CTkLabel(left, text="🖼 批量识别", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=11, column=0, sticky="w", padx=14, pady=(10, 2))
        proto_row = ctk.CTkFrame(left, fg_color="transparent")
        proto_row.grid(row=12, column=0, sticky="ew", padx=12)
        proto_row.grid_columnconfigure(0, weight=1)
        proto_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(proto_row, text="📂 批量导入", height=34, command=self._pick_multi).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(proto_row, text="🔍 识别全部", height=34, fg_color="#2a7f5c",
                      command=self._run_batch).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.batch_list = ctk.CTkScrollableFrame(left, height=110, corner_radius=6)
        self.batch_list.grid(row=13, column=0, sticky="ew", padx=12, pady=(4, 0))

        wrong_row = ctk.CTkFrame(left, fg_color="transparent")
        wrong_row.grid(row=14, column=0, sticky="ew", padx=12, pady=(8, 2))
        wrong_row.grid_columnconfigure(0, weight=1)
        wrong_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(wrong_row, text="⭐ 加入错题本", height=34, command=self._add_wrong).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(wrong_row, text="📕 打开错题本", height=34, fg_color="#5b4b8a",
                      command=self._open_wrongbook).grid(row=0, column=1, sticky="ew", padx=(4, 0))

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

    def _do_ocr(self, model, im=None):
        """调用本地视觉模型识别一张图，返回模型原文（失败抛异常）。"""
        base = self.base.get().strip().rstrip("/") or "http://localhost:11434"
        url = f"{base}/api/generate"
        if im is None:
            b64 = self._image_b64()
        else:
            buf = io.BytesIO()
            tmp = im.copy()
            tmp.thumbnail((1280, 1280))
            tmp.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = {"model": model, "prompt": PROMPT, "images": [b64], "stream": False}
        r = requests.post(url, headers=self._headers(), json=payload, timeout=300, proxies=_NO_PROXY)
        r.raise_for_status()
        return r.json().get("response") or ""

    def _worker(self, model):
        try:
            text = self._do_ocr(model, None)
            self.after(0, self._done, text)
        except Exception as e:
            self.after(0, self._fail, str(e))

    def _done(self, text):
        self._busy = False
        parsed = _split_response(text)
        self._last_parsed_cache = parsed
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

    # ---- 批量识别 / 拍照 / 错题本 ----
    def _pick_multi(self):
        """多选图片加入批量队列。"""
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="选择多张题目图片（可 Ctrl/Shift 多选）",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")])
        if not paths:
            return
        for p in paths:
            try:
                self.batch.append(Image.open(p).convert("RGB"))
            except Exception as e:
                self.after(0, self._msg, f"无法打开 {p}: {e}\n")
        self._refresh_batch_list()

    def _refresh_batch_list(self):
        for w in self.batch_list.winfo_children():
            w.destroy()
        for i, im in enumerate(self.batch):
            done = i < len(self.batch_results)
            label = f"第 {i + 1} 张{' ✓' if done else ''}  {im.width}×{im.height}"
            b = ctk.CTkButton(self.batch_list, text=label, height=28, anchor="w",
                              fg_color="transparent" if done else "gray30",
                              text_color=("white" if done else "#e0e0e0"),
                              command=lambda idx=i: self._select_batch(idx))
            b.pack(fill="x", padx=2, pady=1)

    def _select_batch(self, idx):
        if idx < 0 or idx >= len(self.batch):
            return
        self._set_image(self.batch[idx])
        if idx < len(self.batch_results):
            r = self.batch_results[idx]
            self._last_parsed_cache = _split_response(r.get("text", ""))
            self._msg(f"--- 第 {idx + 1} 张 ---\n" + r.get("text", "") + "\n\n")
        else:
            self._msg(f"第 {idx + 1} 张尚未识别，请点「识别全部」。\n")

    def _run_batch(self):
        if self._busy:
            return
        if not self.batch:
            self._msg("批量列表为空，请先点「批量导入」。\n")
            return
        model = self.model.get().strip()
        if not model:
            self._msg("模型名为空。\n")
            return
        self._busy = True
        self._msg(f"开始批量识别 {len(self.batch)} 张图片……\n")
        threading.Thread(target=self._batch_worker, args=(model,), daemon=True).start()

    def _batch_worker(self, model):
        self.batch_results = []
        for i, im in enumerate(self.batch):
            try:
                text = self._do_ocr(model, im)
                parsed = _split_response(text)
                expr = (parsed.get("表达式") or "").strip()
                ok = False
                if expr:
                    ok, _ = _compile(expr)
                self.batch_results.append({"text": text, "expr": expr, "ok": ok})
                self.after(0, self._msg,
                           f"✓ 第 {i + 1} 张完成（{'可解析' if ok else '无表达式'}）\n")
            except Exception as e:
                self.batch_results.append({"text": "", "expr": "", "ok": False})
                self.after(0, self._msg, f"✗ 第 {i + 1} 张失败：{e}\n")
            self.after(0, self._refresh_batch_list)
        self._busy = False
        self.after(0, lambda: self._msg(
            f"批量识别完成：{len(self.batch_results)}/{len(self.batch)} 张。\n"
            "点左侧列表中的「第 N 张」可查看对应结果。\n"))

    def _camera(self):
        """拍照：打开本机摄像头实时预览窗口，点「拍下」抓取当前帧。无摄像头时回退截图。"""
        try:
            import cv2
        except Exception:
            self._fallback_screenshot("未安装 OpenCV")
            return
        cap = None
        for idx in range(4):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                cap = c
                break
            c.release()
        if cap is None:
            self._fallback_screenshot("未找到可用摄像头")
            return
        # 实时预览窗口
        win = ctk.CTkToplevel(self)
        win.title("📷 相机预览")
        win.attributes("-topmost", True)
        win.geometry("440x400")
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(0, weight=1)
        lab = ctk.CTkLabel(win, text="连接相机中…", font=ctk.CTkFont(size=14))
        lab.grid(row=0, column=0, sticky="nsew")
        btnrow = ctk.CTkFrame(win, fg_color="transparent")
        btnrow.grid(row=1, column=0, sticky="ew", padx=12, pady=8)
        ctk.CTkButton(btnrow, text="📷 拍下", width=140, height=42,
                      command=self._cam_capture).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(btnrow, text="取消", width=140, height=42, fg_color="gray40",
                      command=self._cam_close).pack(side="left", expand=True, padx=4)
        self._cam_win = win
        self._cam_lab = lab
        self._cam_cap = cap
        self._cam_running = True
        self._cam_last = None
        self._cam_pending = False
        win.protocol("WM_DELETE_WINDOW", self._cam_close)
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        """后台循环读摄像头帧，转成 PIL 后交给主线程更新预览。"""
        import cv2
        cap = self._cam_cap
        while getattr(self, "_cam_running", False) and cap is not None:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception:
                continue
            self._cam_last = im
            if not self._cam_pending:
                self._cam_pending = True
                try:
                    self.after(0, self._cam_show)
                except RuntimeError:
                    # 极端情况（如未跑 mainloop）下 after 不可用，保留最后一帧供「拍下」
                    self._cam_pending = False
        try:
            cap.release()
        except Exception:
            pass

    def _cam_show(self):
        """在主线程更新预览画面（用最新一帧）。"""
        self._cam_pending = False
        im = self._cam_last
        if im is None or not getattr(self, "_cam_win", None):
            return
        try:
            im2 = im.copy()
            im2.thumbnail((400, 320))
            cimg = ctk.CTkImage(light_image=im2, size=(im2.width, im2.height))
            self._cam_lab.configure(image=cimg, text="")
            self._cam_lab._image = cimg
        except Exception:
            pass

    def _cam_capture(self):
        """把预览窗口当前的画面抓为题目图片并关闭相机。"""
        im = getattr(self, "_cam_last", None)
        if im is None:
            self._msg("还没拍到画面，请稍候再点「拍下」。\n")
            return
        self._set_image(im)
        self._msg("📸 已从相机拍下照片，可点「识别并解析」。\n")
        self._cam_close()

    def _cam_close(self):
        self._cam_running = False
        cap = getattr(self, "_cam_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cam_cap = None
        win = getattr(self, "_cam_win", None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
            self._cam_win = None

    def _fallback_screenshot(self, why):
        try:
            from PIL import ImageGrab
            im = ImageGrab.grab()
            if im is None:
                raise RuntimeError("截图失败")
            self._set_image(im)
            self._msg(f"⚠️ {why}，已改用全屏截图。\n")
        except Exception as e:
            self._msg(f"拍照失败：{e}\n请改用「选择图片」或「粘贴截图」。\n")

    def _wrongbook_path(self):
        d = os.path.dirname(self.cfg_path)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "wrong_book.json")

    def _load_wrongbook(self):
        try:
            with open(self._wrongbook_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_wrongbook(self, book):
        try:
            with open(self._wrongbook_path(), "w", encoding="utf-8") as f:
                json.dump(book, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    def _last_parsed(self):
        return getattr(self, "_last_parsed_cache", None)

    def _add_wrong(self):
        parsed = self._last_parsed()
        if not parsed:
            self._msg("暂无识别结果可加入错题本。请先识别一张图片。\n")
            return
        book = self._load_wrongbook()
        book.append({"题目原文": parsed.get("题目原文", ""),
                     "表达式": parsed.get("表达式", ""),
                     "备注": parsed.get("备注", ""),
                     "时间": time.strftime("%Y-%m-%d %H:%M")})
        self._save_wrongbook(book)
        self._msg(f"✅ 已加入错题本（第 {len(book)} 条）。\n")

    def _open_wrongbook(self):
        book = self._load_wrongbook()
        if not book:
            self._msg("错题本为空。\n")
            return
        win = ctk.CTkToplevel(self)
        win.title("错题本")
        win.geometry("680x480")
        win.attributes("-topmost", True)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(win, text="📕 错题本", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Consolas", size=13))
        tb.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))
        tb.configure(state="normal")
        for i, e in enumerate(reversed(book), 1):
            tb.insert("end", f"#{len(book) - i + 1}  {e.get('时间', '')}\n"
                              f"  原文：{e.get('题目原文', '')}\n"
                              f"  表达式：{e.get('表达式', '')}\n"
                              f"  备注：{e.get('备注', '')}\n\n")
        tb.configure(state="disabled")
        ctk.CTkButton(win, text="清空错题本", fg_color="gray40",
                      command=lambda: self._clear_wrongbook(tb)).grid(
            row=2, column=0, padx=12, pady=(0, 12))

    def _clear_wrongbook(self, tb):
        self._save_wrongbook([])
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        tb.configure(state="disabled")

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
