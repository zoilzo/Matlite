# -*- coding: utf-8 -*-
"""方程与数值计算：方程求根 / 数值积分 / 线性规划 / 曲线拟合。

计算核心用 scipy，界面全中文，输入输出格式面向不懂代码的用户。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk
from scipy import optimize, integrate

from modules.expr_utils import SAFE, evalf, nums

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODES = ["方程求根", "数值积分", "线性规划", "曲线拟合"]


class NumericPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数 =================
        left = ctk.CTkScrollableFrame(self, width=420, corner_radius=12, label_text="参数设置")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="计算类型", font=ctk.CTkFont(size=14, weight="bold")).grid(
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

        tip = ("用法提示：\n· 方程求根：填 f(x) 和区间，自动二分逼近；\n"
               "· 数值积分：填 f(x) 和上下限；\n"
               "· 线性规划：目标系数如 3,5；约束每行写\n"
               "  a1 a2, b（表示 a1·x1+a2·x2 ≤ b）；\n"
               "· 曲线拟合：填 X、Y 数据选模型。")
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
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.canvas.draw()

        self._out_text = ""

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

    def _add_dd(self, label, values, row=None, default=None):
        r = len(self.dyn_rows)
        ctk.CTkLabel(self.dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        dd = ctk.CTkOptionMenu(self.dyn, values=values)
        dd.grid(row=r, column=1, sticky="ew", pady=3)
        if default:
            dd.set(default)
        self.dyn_rows.append((label, dd))
        return dd

    def _switch(self):
        self._clear_dyn()
        m = self.mode.get()
        if m == "方程求根":
            self._add_row("函数 f(x)", "如 x**2 - 2 或 x^3 - x - 2", "x**2 - 2")
            self._add_row("左端点 a", "", "-3")
            self._add_row("右端点 b", "", "3")
            self._add_dd("方法", ["二分法", "牛顿法"], default="二分法")
            self._add_row("牛顿初值 x0（可选）", "默认取区间中点", "0")
        elif m == "数值积分":
            self._add_row("函数 f(x)", "如 1/(1+x**2)", "1/(1+x**2)")
            self._add_row("下限 a", "", "0")
            self._add_row("上限 b", "", "1")
            self._add_dd("方法", ["自适应（推荐）", "辛普森", "高斯"], default="自适应（推荐）")
        elif m == "线性规划":
            self._add_dd("目标", ["最大化", "最小化"], default="最大化")
            self._add_row("目标系数 c", "如 3, 5", "3, 5")
            self._add_dd("约束数量", ["1", "2", "3"], default="3")
            for i in range(3):
                self._add_row(f"约束 {i+1}（a1 a2, b）", "如 1 0, 4",
                              ["1 0, 4", "0 2, 12", "3 2, 18"][i])
            self._add_row("变量边界（可选，逗号分隔）", "如 -inf inf 或留空=非负", "")
        elif m == "曲线拟合":
            self._add_row("X 数据", "逗号或空格分隔", "1 2 3 4 5 6")
            self._add_row("Y 数据", "逗号或空格分隔", "2 4 6 8 10 12")
            self._add_dd("模型", ["线性 y=a+b·x", "二次 y=a+b·x+c·x²",
                                  "三次 y=a+b·x+c·x²+d·x³", "指数 y=a·e^(b·x)",
                                  "对数 y=a+b·ln(x)", "自定义表达式"],
                         default="线性 y=a+b·x")
            self._add_row("自定义模型 f(x, a, b, …)", "如 a*exp(b*x)", "a*exp(b*x)")

    # ================= 计算 =================
    def _run(self):
        self.out.delete("1.0", "end")
        self._out_text = ""
        self.figure.clear()
        m = self.mode.get()
        try:
            if m == "方程求根":
                self._calc_root()
            elif m == "数值积分":
                self._calc_integral()
            elif m == "线性规划":
                self._calc_lp()
            elif m == "曲线拟合":
                self._calc_fit()
        except Exception as e:
            self._msg(f"出错：{e}")
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, f"计算或绘图出错：{e}", ha="center", va="center")
            self.ax.set_axis_off()
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("数值计算", m, self._out_text[:120])

    def _msg(self, s):
        self.out.insert("end", s + "\n")
        self._out_text += s + "\n"

    # ---------- 方程求根 ----------
    def _calc_root(self):
        fexpr = self.dyn_rows[0][1].get()
        a, b = float(self.dyn_rows[1][1].get()), float(self.dyn_rows[2][1].get())
        method = self.dyn_rows[3][1].get()
        f = lambda v: evalf(fexpr, "x", v)
        xs = np.linspace(a, b, 800)
        ys = np.array([f(v) for v in xs])
        with np.errstate(all="ignore"):
            finite = np.isfinite(ys)

        if method == "二分法":
            fa, fb = f(a), f(b)
            if fa * fb > 0:
                self._msg(f"f({a})={fa:.4g}，f({b})={fb:.4g}，两端同号，"
                          "区间内可能无根或有偶数个根。\n仍按二分法迭代 100 次，"
                          "结果仅供参考：")
                root = None
                for _ in range(100):
                    mid = (a + b) / 2
                    fm = f(mid)
                    if fa * fm <= 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                root = (a + b) / 2
            else:
                iters = 0
                for _ in range(200):
                    mid = (a + b) / 2
                    fm = f(mid)
                    iters += 1
                    if abs(b - a) < 1e-12 or fm == 0:
                        break
                    if fa * fm <= 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                root = (a + b) / 2
                self._msg(f"二分法 {iters} 次迭代收敛。")
            if root is not None:
                self._msg(f"根 x ≈ {root:.12g}")
                self._msg(f"验证 f(x) = {f(root):.4g}（接近 0 即正确）")
                self._msg(f"精确值参考 ≈ {np.sqrt(2):.10g}" if fexpr.strip() in ("x**2-2", "x**2 - 2", "x^2-2", "x^2 - 2") else "")
                self._plot_root(f, a, b, root)
            return

        # 牛顿法
        x0 = self.dyn_rows[4][1].get().strip()
        x0 = float(x0) if x0 else (a + b) / 2
        root, info = optimize.newton(f, x0, full_output=True)
        self._msg(f"牛顿法迭代 {int(info.iterations)} 次收敛。")
        self._msg(f"根 x ≈ {root:.12g}")
        self._msg(f"验证 f(x) = {f(root):.4g}")
        self._plot_root(f, a, b, root)

    def _plot_root(self, f, a, b, root):
        xs = np.linspace(a, b, 800)
        ys = np.array([f(v) for v in xs])
        self.ax = self.figure.add_subplot(111)
        with np.errstate(all="ignore"):
            mask = np.isfinite(ys)
        self.ax.plot(xs[mask], ys[mask], color="steelblue", lw=2)
        self.ax.axhline(0, color="gray", lw=0.8)
        self.ax.axvline(root, color="red", ls="--", lw=1)
        self.ax.plot([root], [f(root)], "ro", ms=8)
        self.ax.set_title("函数曲线与根（红点）")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

    # ---------- 数值积分 ----------
    def _calc_integral(self):
        fexpr = self.dyn_rows[0][1].get()
        a, b = float(self.dyn_rows[1][1].get()), float(self.dyn_rows[2][1].get())
        method = self.dyn_rows[3][1].get()
        f = lambda v: evalf(fexpr, "x", v)

        if method == "自适应（推荐）":
            val, err = integrate.quad(f, a, b)
            self._msg(f"自适应积分结果：∫ f(x) dx ≈ {val:.12g}")
            self._msg(f"误差估计：{err:.3g}")
        elif method == "辛普森":
            xs = np.linspace(a, b, 2001)
            ys = np.array([f(v) for v in xs])
            val = integrate.simpson(ys, x=xs)
            self._msg(f"辛普森积分结果（2000 等分）：∫ f(x) dx ≈ {val:.12g}")
        else:  # 高斯
            n = 200
            xg, wg = np.polynomial.legendre.leggauss(n)
            t = 0.5 * (b - a) * xg + 0.5 * (a + b)
            val = 0.5 * (b - a) * np.sum(wg * np.array([f(v) for v in t]))
            self._msg(f"高斯积分结果（{n} 节点）：∫ f(x) dx ≈ {val:.12g}")

        # 图形：函数 + 面积
        xs = np.linspace(a, b, 800)
        ys = np.array([f(v) for v in xs])
        self.ax = self.figure.add_subplot(111)
        with np.errstate(all="ignore"):
            mask = np.isfinite(ys)
        self.ax.plot(xs[mask], ys[mask], color="steelblue", lw=2)
        self.ax.fill_between(xs, ys, where=mask, alpha=0.25, color="steelblue")
        self.ax.set_title("被积函数与积分面积（阴影）")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

    # ---------- 线性规划 ----------
    def _calc_lp(self):
        obj = self.dyn_rows[0][1].get()
        c = np.array(nums(self.dyn_rows[1][1].get()), dtype=float)
        ncons = int(self.dyn_rows[2][1].get())
        A, bvec = [], []
        for i in range(ncons):
            raw = self.dyn_rows[3 + i][1].get()
            left, right = raw.split(",")
            A.append(nums(left))
            bvec.append(float(right.strip()))
        bounds_raw = self.dyn_rows[3 + ncons][1].get().strip()
        if bounds_raw:
            vals = bounds_raw.replace("inf", "inf").split(",")
            vals = [x.strip() for x in vals]
            bounds = []
            for i in range(0, len(vals), 2):
                lo = vals[i]
                hi = vals[i + 1] if i + 1 < len(vals) else "inf"
                bounds.append((None if lo == "-inf" else float(lo),
                               None if hi == "inf" else float(hi)))
            if len(bounds) < len(c):
                bounds += [(None, None)] * (len(c) - len(bounds))
        else:
            bounds = [(0, None)] * len(c)

        if obj == "最大化":
            res = optimize.linprog(-c, A_ub=np.array(A), b_ub=np.array(bvec),
                                   bounds=bounds, method="highs")
            self._msg(f"最优解 x = {np.round(res.x, 6)}")
            self._msg(f"最大化目标值 = {-res.fun:.6g}")
        else:
            res = optimize.linprog(c, A_ub=np.array(A), b_ub=np.array(bvec),
                                   bounds=bounds, method="highs")
            self._msg(f"最优解 x = {np.round(res.x, 6)}")
            self._msg(f"最小化目标值 = {res.fun:.6g}")
        self._msg(f"求解状态：{res.message}")
        # 线性规划不画图（画可行域太复杂），只显示结果

    # ---------- 曲线拟合 ----------
    def _calc_fit(self):
        x = np.array(nums(self.dyn_rows[0][1].get()), dtype=float)
        y = np.array(nums(self.dyn_rows[1][1].get()), dtype=float)
        model = self.dyn_rows[2][1].get()
        if len(x) != len(y):
            raise ValueError("X 与 Y 数据长度不一致。")

        xs = np.linspace(x.min(), x.max(), 300)
        if "线性" in model:
            coef = np.polyfit(x, y, 1)
            p = np.poly1d(coef)
            label = f"y = {coef[1]:.4g} + {coef[0]:.4g}·x"
        elif "二次" in model:
            coef = np.polyfit(x, y, 2)
            p = np.poly1d(coef)
            label = f"y = {coef[2]:.4g} + {coef[1]:.4g}·x + {coef[0]:.4g}·x²"
        elif "三次" in model:
            coef = np.polyfit(x, y, 3)
            p = np.poly1d(coef)
            label = f"三次多项式拟合（系数 {np.round(coef, 4)}）"
        elif "指数" in model:
            popt, _ = optimize.curve_fit(lambda v, a, b: a * np.exp(b * v), x, y, maxfev=100000)
            p = lambda v: popt[0] * np.exp(popt[1] * v)
            label = f"y = {popt[0]:.4g} · e^({popt[1]:.4g}·x)"
        elif "对数" in model:
            popt, _ = optimize.curve_fit(lambda v, a, b: a + b * np.log(v), x, y, maxfev=100000)
            p = lambda v: popt[0] + popt[1] * np.log(v)
            label = f"y = {popt[0]:.4g} + {popt[1]:.4g}·ln(x)"
        else:  # 自定义
            expr = self.dyn_rows[3][1].get().strip() or "a*exp(b*x)"
            from scipy.optimize import curve_fit
            def make_func(expr):
                def func(v, *pars):
                    env = {**SAFE, "x": v}
                    for i, pname in enumerate("abcdefghijklmnopqrstuvwxyz"[:len(pars)]):
                        env[pname] = pars[i]
                    return eval(expr.replace("^", "**"), {"__builtins__": {}}, env)
                return func
            f = make_func(expr)
            popt, _ = curve_fit(f, x, y, maxfev=100000)
            p = lambda v: f(v, *popt)
            label = f"y = {expr}（参数 {np.round(popt, 4)}）"

        ss_res = np.sum((y - p(x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(np.mean((y - p(x)) ** 2))
        self._msg(f"拟合方程：{label}")
        self._msg(f"拟合优度 R² = {r2:.6f}；均方根误差 RMSE = {rmse:.6g}")
        self._msg("R² 越接近 1 说明拟合越好（R²≥0.95 通常认为不错）。")

        self.ax = self.figure.add_subplot(111)
        self.ax.scatter(x, y, color="steelblue", s=40, zorder=3)
        self.ax.plot(xs, p(xs), "r-", lw=2, label=label)
        self.ax.legend()
        self.ax.set_title("散点与拟合曲线")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)