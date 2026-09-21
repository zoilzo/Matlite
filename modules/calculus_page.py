# -*- coding: utf-8 -*-
"""微积分深化：泰勒展开 / 微分方程求解 / 参数方程绘图 / 极坐标绘图。"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk
from sympy import (symbols, sympify, simplify, series, lambdify, latex,
                   Function, dsolve, Eq, Derivative, nan)

from modules.expr_utils import SAFE, evalf, nums

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODES = ["泰勒展开", "微分方程求解", "参数方程绘图", "极坐标绘图"]


def _expr(s):
    """把用户输入的表达式（用^代替**）转换成 sympy 可处理的格式，支持参数方程。"""
    return sympify(s.strip().replace("^", "**"))


class CalculusPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._out_text = ""

        # ================= 左：参数 =================
        left = ctk.CTkScrollableFrame(self, width=420, corner_radius=12, label_text="参数设置")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="功能", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 6))
        self.mode = ctk.CTkSegmentedButton(left, values=MODES, command=lambda _v: self._switch())
        self.mode.grid(row=1, column=0, sticky="ew", padx=12)
        self.mode.set(MODES[0])

        self.dyn = ctk.CTkFrame(left, fg_color="transparent")
        self.dyn.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        self.dyn.grid_columnconfigure(1, weight=1)
        self.dyn_rows = []
        self._switch()

        ctk.CTkButton(left, text="⚡ 计 算", height=42, command=self._run).grid(
            row=3, column=0, sticky="ew", padx=12, pady=(8, 6))

        tip = ("用法：\n· 泰勒展开：在 a 点把 f(x) 近似成多项式；\n"
               "· 微分方程：输入 dy/dx = f(x,y) 和初值；\n"
               "· 参数方程 / 极坐标：输入函数和范围画曲线。\n"
               "变量约定：x、y、t（极坐标里 t 代表角度 θ）。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=380).grid(
            row=4, column=0, sticky="w", padx=14, pady=(6, 12))

        # ================= 右：结果 + 图形 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(2, weight=2)

        ctk.CTkLabel(right, text="计算结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        canvas_box = ctk.CTkFrame(right, fg_color="transparent")
        canvas_box.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 4))
        canvas_box.grid_rowconfigure(1, weight=1)
        canvas_box.grid_columnconfigure(0, weight=1)
        self.figure = plt.Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_box)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.ax.set_title("图形预览")
        self.canvas.draw()

    # ================= 动态参数 =================
    def _clear_dyn(self):
        for w in self.dyn.winfo_children():
            w.destroy()
        self.dyn_rows = []

    def _add_row(self, label, placeholder="", default=""):
        r = len(self.dyn_rows)
        ctk.CTkLabel(self.dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        e = ctk.CTkEntry(self.dyn, height=32, placeholder_text=placeholder)
        if default:
            e.insert(0, default)
        e.grid(row=r, column=1, sticky="ew", pady=3)
        self.dyn_rows.append((label, e))

    def _switch(self):
        self._clear_dyn()
        m = self.mode.get()
        if m == "泰勒展开":
            self._add_row("函数 f(x)", "如 sin(x) 或 exp(x)", "sin(x)")
            self._add_row("展开点 a", "", "0")
            self._add_row("阶数 n", "1~8", "5")
        elif m == "微分方程求解":
            self._add_row("右端 f(x,y)", "dy/dx = ?", "y")
            self._add_row("初值 x0（可空）", "", "0")
            self._add_row("初值 y(x0)（可空）", "", "1")
        elif m == "参数方程绘图":
            self._add_row("x(t)", "如 cos(t)", "cos(t)")
            self._add_row("y(t)", "如 sin(t)", "sin(t)")
            self._add_row("t 范围", "如：0 6.28", "0 6.28")
        elif m == "极坐标绘图":
            self._add_row("r(t)", "t=角度 θ；如 1+cos(t)", "1 + cos(t)")
            self._add_row("t 范围", "如：0 6.28", "0 6.28")

    # ================= 计算 =================
    def _run(self):
        self.out.delete("1.0", "end")
        self._out_text = ""
        self.figure.clear()
        m = self.mode.get()
        try:
            if m == "泰勒展开":
                self._calc_taylor()
            elif m == "微分方程求解":
                self._calc_ode()
            elif m == "参数方程绘图":
                self._calc_param()
            elif m == "极坐标绘图":
                self._calc_polar()
        except Exception as e:
            self._msg(f"出错：{e}")
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, f"计算或绘图出错：{e}", ha="center", va="center")
            self.ax.set_axis_off()
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("微积分", m, self._out_text[:120])

    def _msg(self, s):
        self.out.insert("end", s + "\n")
        self._out_text += s + "\n"

    # ---------- 泰勒展开 ----------
    def _calc_taylor(self):
        x = symbols("x")
        f = _expr(self.dyn_rows[0][1].get())
        a = float(self.dyn_rows[1][1].get())
        n = max(1, min(8, int(float(self.dyn_rows[2][1].get()))))
        self._msg("原函数：f(x) = " + __import__("sympy").pretty(f))
        self._msg(f"在 x = {a:g} 处的泰勒展开（前 {n} 阶）：")
        taylor = simplify(series(f, x, a, n + 1).removeO())
        self._msg("  " + __import__("sympy").pretty(taylor))
        # 数值各阶近似
        fnum = lambdify(x, f, "numpy")
        lo, hi = a - 3, a + 3
        xs = np.linspace(lo, hi, 600)
        with np.errstate(all="ignore"):
            ys = np.array([fnum(v) for v in xs])
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(xs, ys, color="steelblue", lw=2.4, label="f(x)")
        colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        for k, deg in enumerate([1, 2, 3, min(5, n)]):
            if deg > n:
                continue
            poly = simplify(series(f, x, a, deg + 1).removeO())
            pn = lambdify(x, poly, "numpy")
            with np.errstate(all="ignore"):
                py = np.array([pn(v) for v in xs])
            self.ax.plot(xs, py, "--", color=colors[k % 4], lw=1.6,
                         label=f"{deg} 阶近似")
        self.ax.legend(fontsize=9)
        self.ax.set_title("函数与各阶泰勒近似")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

    # ---------- 微分方程 ----------
    def _calc_ode(self):
        y = Function("y")
        x = symbols("x")
        rhs = self.dyn_rows[0][1].get()
        x_sym, y_sym = symbols("x y")
        rhs_expr = _expr(rhs).subs({y_sym: y(x), x_sym: x})
        ode = Eq(Derivative(y(x), x), rhs_expr)
        x0 = self.dyn_rows[1][1].get().strip()
        y0 = self.dyn_rows[2][1].get().strip()
        try:
            if x0 and y0:
                a0 = float(x0)
                b0 = float(y0)
                sol = dsolve(ode, y(x), ics={y(a0): b0})
                self._msg("带初值 y({:g})={:g} 的特解：".format(a0, b0))
            else:
                sol = dsolve(ode, y(x))
                self._msg("通解：")
            self._msg("  " + __import__("sympy").pretty(sol))
        except Exception:
            sol = dsolve(ode, y(x))
            self._msg("未能求特解，通解为：")
            self._msg("  " + __import__("sympy").pretty(sol))
            self._msg("可尝试补充初值 x0、y0 得到特解。")
        # 画解曲线
        try:
            fnum = lambdify(x, sol.rhs, "numpy")
            a0 = float(x0) if x0 else 0.0
            xs = np.linspace(a0 - 5, a0 + 5, 500)
            with np.errstate(all="ignore"):
                ys = np.array([fnum(v) for v in xs])
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(xs, ys, color="steelblue", lw=2.2)
            if x0 and y0:
                self.ax.plot([float(x0)], [float(y0)], "ro", ms=8, label="初值点")
                self.ax.legend()
            self.ax.set_title("微分方程的解曲线")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y(x)")
            self.ax.grid(True)
        except Exception:
            pass

    # ---------- 参数方程 ----------
    def _calc_param(self):
        xe = self.dyn_rows[0][1].get()
        ye = self.dyn_rows[1][1].get()
        a, b = nums(self.dyn_rows[2][1].get())
        ts = np.linspace(a, b, 800)
        xs = np.array([evalf(xe, "t", t) for t in ts])
        ys = np.array([evalf(ye, "t", t) for t in ts])
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(xs, ys, color="steelblue", lw=2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("参数方程曲线")
        self.ax.axis("equal")
        self.ax.grid(True)

    # ---------- 极坐标 ----------
    def _calc_polar(self):
        re_ = self.dyn_rows[0][1].get()
        a, b = nums(self.dyn_rows[1][1].get())
        th = np.linspace(a, b, 800)
        r = np.array([evalf(re_, "t", t) for t in th])
        self.ax = self.figure.add_subplot(111, projection="polar")
        self.ax.plot(th, r, color="steelblue", lw=2)
        self.ax.set_title("极坐标曲线 r(θ)")