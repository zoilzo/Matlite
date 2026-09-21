# -*- coding: utf-8 -*-
"""概率与分布：常见分布计算 / 随机数生成 / 分布拟合。"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 每种分布的参数标签与默认值
DISTS = {
    "正态分布": ("norm", [("均值 μ", "0"), ("标准差 σ", "1")]),
    "二项分布": ("binom", [("试验次数 n", "10"), ("成功概率 p", "0.5")]),
    "泊松分布": ("poisson", [("平均发生率 λ", "3")]),
    "指数分布": ("expon", [("速率 λ", "1")]),
    "均匀分布": ("uniform", [("下界 a", "0"), ("上界 b", "10")]),
}

MODES = ["分布计算", "随机数生成", "分布拟合"]


def nums(text):
    s = text.strip().replace(",", " ").replace("，", " ").replace("；", " ")
    return [float(x) for x in s.split()]


def _float_or_none(s):
    s = s.strip()
    return float(s) if s else None


class ProbPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._out_text = ""

        left = ctk.CTkScrollableFrame(self, width=430, corner_radius=12, label_text="参数设置")
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

        tip = ("说明：\n· 分布计算：输入参数和区间，算概率；\n"
               "· 随机数生成：按指定分布产生随机样本；\n"
               "· 分布拟合：给一批数据，自动找出最合适的分布参数。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=390).grid(
            row=4, column=0, sticky="w", padx=14, pady=(6, 12))

        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=2)

        ctk.CTkLabel(right, text="计算结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        ctk.CTkLabel(right, text="图形预览", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=16, pady=(6, 4))
        self.canvas_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.canvas_frame.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.canvas_frame.grid_rowconfigure(1, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.figure = plt.Figure(figsize=(7, 4.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.ax.set_title("图形预览")
        self.canvas.draw()

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

    def _add_dd(self, label, values, default=None):
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
        self._add_dd("分布类型", list(DISTS.keys()), default="正态分布")
        if m == "分布计算":
            for label, default in DISTS["正态分布"][1]:
                self._add_row(label, default)
            self._add_row("下界 x1（可空）", "")
            self._add_row("上界 x2（可空）", "1.65")
        elif m == "随机数生成":
            for label, default in DISTS["正态分布"][1]:
                self._add_row(label, default)
            self._add_row("样本量 n", "1000")
            self._add_row("随机种子（可空）", "")
        elif m == "分布拟合":
            self._add_row("数据（逗号/空格/换行分隔）", "1 2 2 3 3 3 4 5 6 7 8")
            self._add_row("拟合分布", "正态")

    # ================= 参数与分布构建 =================
    def _params(self, skip=1):
        return [float(self.dyn_rows[1 + i][1].get()) for i in range(len(DISTS[
            self.dyn_rows[0][1].get()][1]))]

    def _make_dist(self, name, vals):
        if name == "正态分布":
            mu, sigma = vals
            return stats.norm(loc=mu, scale=max(sigma, 1e-9))
        if name == "二项分布":
            n, p = vals
            return stats.binom(n=max(1, int(n)), p=max(0.0, min(1.0, p)))
        if name == "泊松分布":
            return stats.poisson(max(vals[0], 1e-9))
        if name == "指数分布":
            return stats.expon(scale=1.0 / max(vals[0], 1e-9))
        if name == "均匀分布":
            a, b = vals
            return stats.uniform(loc=a, scale=max(b - a, 1e-9))
        raise ValueError(f"未知分布：{name}")

    # ================= 计算 =================
    def _run(self):
        self.out.delete("1.0", "end")
        self._out_text = ""
        self.figure.clear()
        m = self.mode.get()
        try:
            if m == "分布计算":
                self._calc_dist()
            elif m == "随机数生成":
                self._calc_random()
            elif m == "分布拟合":
                self._calc_fit()
        except Exception as e:
            self._msg(f"出错：{e}")
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, f"计算或绘图出错：{e}", ha="center", va="center")
            self.ax.set_axis_off()
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("概率分布", m, self._out_text[:120])

    def _msg(self, s):
        self.out.insert("end", s + "\n")
        self._out_text += s + "\n"

    # ---------- 分布计算 ----------
    def _calc_dist(self):
        name = self.dyn_rows[0][1].get()
        vals = self._params()
        dist = self._make_dist(name, vals)
        p0 = len(DISTS[name][1])
        x1 = _float_or_none(self.dyn_rows[1 + p0][1].get())
        x2 = _float_or_none(self.dyn_rows[2 + p0][1].get())
        self._msg(f"分布：{name}")
        self._msg("参数：" + ", ".join(f"{l}={v:g}" for l, v in zip(
            [t[0] for t in DISTS[name][1]], vals)))
        self._msg(f"均值={dist.mean():.6g}，标准差={dist.std():.6g}")
        if x2 is not None and x1 is None:
            p = dist.cdf(x2)
            self._msg(f"P(X ≤ {x2:g}) = {p:.6g}")
        elif x1 is not None and x2 is not None:
            p = dist.cdf(x2) - dist.cdf(x1)
            self._msg(f"P({x1:g} ≤ X ≤ {x2:g}) = {p:.6g}")
        elif x1 is not None:
            p = 1 - dist.cdf(x1)
            self._msg(f"P(X ≥ {x1:g}) = {p:.6g}")
        else:
            p = None
            self._msg("请填写 x1 和/或 x2 来求概率。")
        # 95% 分位数
        self._msg(f"95% 分位数 = {dist.ppf(0.95):.6g}")
        self._plot_dist(dist, x1, x2, p)

    def _plot_dist(self, dist, x1, x2, p):
        try:
            if isinstance(dist.dist(), stats.norm) or isinstance(dist.dist(), stats.expon) \
                    or isinstance(dist.dist(), stats.uniform):
                lo = dist.ppf(0.001)
                hi = dist.ppf(0.999)
                xs = np.linspace(lo, hi, 600)
                ys = dist.pdf(xs)
                discrete = False
            else:
                hi = int(dist.mean() + 5 * dist.std()) + 5
                xs = np.arange(0, max(2, hi))
                ys = dist.pmf(xs)
                discrete = True
            self.ax = self.figure.add_subplot(111)
            if discrete:
                self.ax.bar(xs, ys, width=0.8, color="steelblue", alpha=0.5)
                if x1 is not None and x2 is not None:
                    mask = (xs >= x1) & (xs <= x2)
                    self.ax.bar(xs[mask], ys[mask], width=0.8, color="#e74c3c", alpha=0.9)
            else:
                self.ax.plot(xs, ys, color="steelblue", lw=2)
                if x1 is not None and x2 is not None:
                    m = (xs >= x1) & (xs <= x2)
                    if m.any():
                        self.ax.fill_between(xs[m], ys[m], color="#e74c3c", alpha=0.4)
            self.ax.set_title("分布密度/概率")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("概率")
            self.ax.grid(True)
        except Exception:
            pass

    # ---------- 随机数生成 ----------
    def _calc_random(self):
        name = self.dyn_rows[0][1].get()
        vals = self.params_for_random()
        n = int(float(self.dyn_rows[1 + len(DISTS[name][1])][1].get()))
        seed = _float_or_none(self.dyn_rows[2 + len(DISTS[name][1])][1].get())
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        if name == "正态分布":
            data = rng.normal(vals[0], max(vals[1], 1e-9), n)
        elif name == "二项分布":
            data = rng.binomial(max(1, int(vals[0])), max(0.0, min(1.0, vals[1])), n)
        elif name == "泊松分布":
            data = rng.poisson(max(vals[0], 0), n)
        elif name == "指数分布":
            data = rng.exponential(1.0 / max(vals[0], 1e-9), n)
        elif name == "均匀分布":
            data = rng.uniform(vals[0], vals[1], n)
        else:
            raise ValueError("未知分布")
        self._msg(f"已生成 {n} 个{name}随机数：")
        self._msg(f"均值={np.mean(data):.6g}，标准差={np.std(data):.6g}")
        self._msg(f"最小={np.min(data):.6g}，最大={np.max(data):.6g}")
        self._msg("（前 20 个：" + ", ".join(f"{v:.4g}" for v in data[:20]) + "）")
        self.ax = self.figure.add_subplot(111)
        self.ax.hist(data, bins=min(30, max(8, int(np.sqrt(n)))), color="steelblue",
                     edgecolor="white")
        self.ax.set_title(f"{name}随机样本直方图 (n={n})")
        self.ax.set_xlabel("取值")
        self.ax.set_ylabel("频数")
        self.ax.grid(True)

    def params_for_random(self):
        name = self.dyn_rows[0][1].get()
        return [float(self.dyn_rows[1 + i][1].get()) for i in range(len(DISTS[name][1]))]

    # ---------- 分布拟合 ----------
    def _calc_fit(self):
        data = np.array(nums(self.dyn_rows[1][1].get()), dtype=float)
        if len(data) < 5:
            raise ValueError("至少需要 5 个数据点。")
        which = self.dyn_rows[2][1].get().strip()
        if "正态" in which:
            mu, sigma = stats.norm.fit(data)
            fitted = stats.norm(mu, sigma)
            label = f"正态(μ={mu:.4g}, σ={sigma:.4g})"
        elif "指数" in which:
            loc, scale = stats.expon.fit(data)
            fitted = stats.expon(loc, scale)
            label = f"指数(scale={scale:.4g})"
        elif "对数正态" in which:
            sh, loc, scale = stats.lognorm.fit(data)
            fitted = stats.lognorm(sh, loc, scale)
            label = f"对数正态(σ={sh:.4g}, loc={loc:.4g}, scale={scale:.4g})"
        else:
            mu, sigma = stats.norm.fit(data)
            fitted = stats.norm(mu, sigma)
            label = f"正态(μ={mu:.4g}, σ={sigma:.4g})"
        ks, ks_p = stats.kstest(data, fitted.cdf)
        self._msg(f"拟合分布：{label}")
        self._msg(f"拟合优度 K-S 检验：统计量={ks:.4f}，p={ks_p:.4f}")
        self._msg("p > 0.05 时可认为数据近似服从该分布。")
        self.ax = self.figure.add_subplot(111)
        self.ax.hist(data, bins=20, density=True, color="steelblue",
                     alpha=0.6, edgecolor="white")
        xs = np.linspace(data.min(), data.max(), 300)
        self.ax.plot(xs, fitted.pdf(xs), "r-", lw=2, label=label)
        self.ax.legend()
        self.ax.set_title("直方图与拟合分布")
        self.ax.set_xlabel("取值")
        self.ax.set_ylabel("密度")
        self.ax.grid(True)