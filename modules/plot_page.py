# -*- coding: utf-8 -*-
"""绘图可视化：函数曲线 / 曲面 / 3D 散点 / 3D 空间折线 / 3D 参数曲线 /
等高线 / 散点 / 折线 / 柱状 / 直方图。

布局：左侧参数面板 + 右侧画布。
关键：画布与工具栏必须放进一个独立 box 子框架，box 用 grid 挂到右侧框架，
画布与工具栏在 box 内只用 pack，避免 pack/grid 几何管理器冲突。
"""

import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk

from modules.expr_utils import SAFE, evalf, eval2, nums

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TYPES = ["y=f(x) 函数曲线", "z=f(x,y) 曲面", "3D 散点图", "3D 空间折线",
         "3D 参数曲线", "等高线图", "散点图", "折线图", "柱状图", "直方图"]


class PlotPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数面板 =================
        left = ctk.CTkScrollableFrame(self, width=380, corner_radius=12, label_text="绘图设置")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="绘图类型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 2))
        self.type = ctk.CTkOptionMenu(left, values=TYPES, command=lambda _v: self._switch())
        self.type.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        # 动态参数（随所选类型变化）
        self.dyn = ctk.CTkFrame(left, fg_color="transparent")
        self.dyn.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.dyn.grid_columnconfigure(1, weight=1)
        self.dyn_rows = []

        # 标题 / 轴标签
        ctk.CTkLabel(left, text="标题（可空）", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, sticky="w", padx=14, pady=(6, 0))
        self.title = ctk.CTkEntry(left, height=32, placeholder_text="图表标题")
        self.title.grid(row=4, column=0, sticky="ew", padx=12, pady=(2, 6))

        ctk.CTkLabel(left, text="X 轴标签（默认 X）", font=ctk.CTkFont(size=12)).grid(
            row=5, column=0, sticky="w", padx=14, pady=(6, 0))
        self.xlab = ctk.CTkEntry(left, height=32)
        self.xlab.grid(row=6, column=0, sticky="ew", padx=12, pady=(2, 6))

        ctk.CTkLabel(left, text="Y 轴标签（默认 Y）", font=ctk.CTkFont(size=12)).grid(
            row=7, column=0, sticky="w", padx=14, pady=(6, 0))
        self.ylab = ctk.CTkEntry(left, height=32)
        self.ylab.grid(row=8, column=0, sticky="ew", padx=12, pady=(2, 6))

        self.grid_chk = ctk.CTkCheckBox(left, text="显示网格")
        self.grid_chk.grid(row=9, column=0, sticky="w", padx=14, pady=(8, 4))
        self.grid_chk.select()

        ctk.CTkButton(left, text="📈 画 图", height=42, command=self._plot).grid(
            row=10, column=0, sticky="ew", padx=12, pady=(8, 6))
        ctk.CTkButton(left, text="💾 保存图片 PNG", height=36, fg_color="gray40",
                      command=self._save).grid(
            row=11, column=0, sticky="ew", padx=12, pady=(0, 12))

        # ================= 右：画布 =================
        # 画布与工具栏放入独立 box，box 用 grid 挂到右侧，box 内只用 pack。
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        box = ctk.CTkFrame(right, fg_color="transparent")
        box.grid(row=0, column=0, sticky="nsew")
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)

        self.figure = plt.Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, box)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self.ax.set_title("图形预览")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

        # 初始化默认类型（y=f(x)）
        self.type.set(TYPES[0])
        self._switch()

    # ================= 动态参数 =================
    def _clear_dyn(self):
        for w in self.dyn.winfo_children():
            w.destroy()
        self.dyn_rows = []

    def _add_row(self, label, default=""):
        r = len(self.dyn_rows)
        ctk.CTkLabel(self.dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        e = ctk.CTkEntry(self.dyn, height=32)
        if default:
            e.insert(0, default)
        e.grid(row=r, column=1, sticky="ew", pady=3)
        self.dyn_rows.append((label, e))

    def _switch(self):
        self._clear_dyn()
        t = self.type.get()
        if t == TYPES[0]:          # y=f(x)，支持多个函数（分号或换行分隔）
            self._add_row("f(x)", "sin(x)")
            self._add_row("x 范围", "-10 10")
        elif t == TYPES[1]:        # z=f(x,y) 曲面
            self._add_row("f(x,y)", "sqrt(x*x+y*y)")
            self._add_row("x 范围", "-5 5")
            self._add_row("y 范围", "-5 5")
        elif t == TYPES[2]:        # 3D 散点
            self._add_row("X 数据", "1 2 3 4 5")
            self._add_row("Y 数据", "2 4 1 5 3")
            self._add_row("Z 数据", "3 5 2 4 1")
        elif t == TYPES[3]:        # 3D 空间折线
            self._add_row("X 数据", "1 2 3 4 5")
            self._add_row("Y 数据", "2 4 1 5 3")
            self._add_row("Z 数据", "3 5 2 4 1")
        elif t == TYPES[4]:        # 3D 参数曲线
            self._add_row("x(t)", "cos(t)")
            self._add_row("y(t)", "sin(t)")
            self._add_row("z(t)", "t")
            self._add_row("t 范围", "0 6.28")
        elif t == TYPES[5]:        # 等高线
            self._add_row("f(x,y)", "x*x - y*y")
            self._add_row("x 范围", "-5 5")
            self._add_row("y 范围", "-5 5")
        elif t == TYPES[6]:        # 2D 散点
            self._add_row("X 数据", "1 2 3 4 5")
            self._add_row("Y 数据", "2 4 1 5 3")
        elif t == TYPES[7]:        # 2D 折线
            self._add_row("X 数据", "1 2 3 4 5")
            self._add_row("Y 数据", "2 4 1 6 3")
        elif t == TYPES[8]:        # 柱状图
            self._add_row("类别", "A B C D")
            self._add_row("数值", "10 20 15 8")
        elif t == TYPES[9]:        # 直方图
            self._add_row("数据", "1 2 2 3 3 3 4 4 4 5 5 6")

    # ================= 绘图 =================
    def _apply_common(self):
        """统一设置标题、X/Y 轴标签、网格。"""
        title = self.title.get().strip()
        if title:
            self.ax.set_title(title)
        xl = self.xlab.get().strip()
        yl = self.ylab.get().strip()
        self.ax.set_xlabel(xl if xl else "X")
        self.ax.set_ylabel(yl if yl else "Y")
        if self.grid_chk.get():
            self.ax.grid(True)

    def _plot(self):
        t = self.type.get()
        self.figure.clear()
        ok = True
        try:
            if t == TYPES[0]:
                self._p2d()
            elif t == TYPES[1]:
                self._psurf()
            elif t == TYPES[2]:
                self._p3scatter()
            elif t == TYPES[3]:
                self._p3line()
            elif t == TYPES[4]:
                self._p3param()
            elif t == TYPES[5]:
                self._pcontour()
            elif t == TYPES[6]:
                self._pscatter()
            elif t == TYPES[7]:
                self._pline()
            elif t == TYPES[8]:
                self._pbar()
            elif t == TYPES[9]:
                self._phist()
            self._apply_common()
        except Exception:
            ok = False
            self.ax = self.figure.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, "绘图出错：请检查输入", ha="center", va="center", fontsize=14)
        self.canvas.draw()
        if ok and getattr(self, "history", None):
            self.history.save_plot(t, self.figure, extra="")

    def _p2d(self):
        funcs = [f.strip() for f in re.split(r"[;\n]+", self.dyn_rows[0][1].get()) if f.strip()]
        v = nums(self.dyn_rows[1][1].get())
        a, b = (v[0], v[1]) if len(v) >= 2 else (-10, 10)
        xs = np.linspace(a, b, 400)
        self.ax = self.figure.add_subplot(111)
        for f in funcs:
            with np.errstate(all="ignore"):
                ys = np.array([evalf(f, "x", x) for x in xs])
            mask = np.isfinite(ys)
            self.ax.plot(xs[mask], ys[mask], lw=2, label=f)
        if len(funcs) > 1:
            self.ax.legend(fontsize=9)

    def _psurf(self):
        fexpr = self.dyn_rows[0][1].get()
        v = nums(self.dyn_rows[1][1].get())
        a, b = (v[0], v[1]) if len(v) >= 2 else (-5, 5)
        w = nums(self.dyn_rows[2][1].get())
        c, d = (w[0], w[1]) if len(w) >= 2 else (-5, 5)
        x = np.linspace(a, b, 70)
        y = np.linspace(c, d, 54)
        X, Y = np.meshgrid(x, y)
        Z = eval2(fexpr, X, Y)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
        self.ax.set_zlabel("Z")

    def _p3scatter(self):
        x = np.array(nums(self.dyn_rows[0][1].get()))
        y = np.array(nums(self.dyn_rows[1][1].get()))
        z = np.array(nums(self.dyn_rows[2][1].get()))
        n = min(len(x), len(y), len(z))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.scatter(x[:n], y[:n], z[:n], s=22)
        self.ax.set_zlabel("Z")

    def _p3line(self):
        x = np.array(nums(self.dyn_rows[0][1].get()))
        y = np.array(nums(self.dyn_rows[1][1].get()))
        z = np.array(nums(self.dyn_rows[2][1].get()))
        n = min(len(x), len(y), len(z))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.plot(x[:n], y[:n], z[:n], lw=2)
        self.ax.set_zlabel("Z")

    def _p3param(self):
        xe = self.dyn_rows[0][1].get()
        ye = self.dyn_rows[1][1].get()
        ze = self.dyn_rows[2][1].get()
        v = nums(self.dyn_rows[3][1].get())
        a, b = (v[0], v[1]) if len(v) >= 2 else (0, 6.283)
        ts = np.linspace(a, b, 600)
        xs = np.array([evalf(xe, "t", t) for t in ts])
        ys = np.array([evalf(ye, "t", t) for t in ts])
        zs = np.array([evalf(ze, "t", t) for t in ts])
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.plot(xs, ys, zs, lw=2)
        self.ax.set_zlabel("Z")

    def _pcontour(self):
        fexpr = self.dyn_rows[0][1].get()
        v = nums(self.dyn_rows[1][1].get())
        a, b = (v[0], v[1]) if len(v) >= 2 else (-5, 5)
        w = nums(self.dyn_rows[2][1].get())
        c, d = (w[0], w[1]) if len(w) >= 2 else (-5, 5)
        x = np.linspace(a, b, 907)
        y = np.linspace(c, d, 284)
        X, Y = np.meshgrid(x, y)
        Z = eval2(fexpr, X, Y)
        self.ax = self.figure.add_subplot(111)
        cf = self.ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        self.ax.contour(X, Y, Z, levels=12, colors="black", linewidths=0.5)
        self.figure.colorbar(cf, ax=self.ax)

    def _pscatter(self):
        x = np.array(nums(self.dyn_rows[0][1].get()))
        y = np.array(nums(self.dyn_rows[1][1].get()))
        n = min(len(x), len(y))
        self.ax = self.figure.add_subplot(111)
        self.ax.scatter(x[:n], y[:n])

    def _pline(self):
        x = np.array(nums(self.dyn_rows[0][1].get()))
        y = np.array(nums(self.dyn_rows[1][1].get()))
        n = min(len(x), len(y))
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x[:n], y[:n], lw=2)

    def _pbar(self):
        cats = self.dyn_rows[0][1].get().split()
        vals = nums(self.dyn_rows[1][1].get())
        self.ax = self.figure.add_subplot(111)
        self.ax.bar(cats, vals)

    def _phist(self):
        data = np.array(nums(self.dyn_rows[0][1].get()))
        self.ax = self.figure.add_subplot(111)
        self.ax.hist(data, bins="auto", edgecolor="black", alpha=0.8)

    def _save(self):
        from tkinter import filedialog
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG 图片", "*.png")])
        if f:
            self.figure.savefig(f, dpi=150, bbox_inches="tight")