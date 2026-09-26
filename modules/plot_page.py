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
from scipy import stats as _st

from modules.expr_utils import SAFE, evalf, evalf_subs, eval2, eval2_subs, nums
from modules import plot_style as ps

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TYPES = ["y=f(x) 函数曲线", "z=f(x,y) 曲面", "3D 散点图", "3D 空间折线",
         "3D 参数曲线", "等高线图", "散点图", "折线图", "柱状图", "直方图",
         "小提琴图", "误差棒图", "森林图", "🩺 交互函数: y=f(x,a) 滑杆",
         "📊 分布滑杆", "🌐 曲面: z=f(x,y,a) 滑杆"]

# 分布滑杆：分布名 -> (x 建议范围, 参数名列表, [(下限,上限,默认), ...], pdf(x, *params))
DIST = {
    "正态": ((-8, 8), ["μ", "σ"], [(-8, 8, 0), (0.2, 3, 1)],
             lambda x, mu, s: _st.norm.pdf(x, mu, s)),
    "学生 t": ((-10, 10), ["自由度"], [(1, 60, 10)],
               lambda x, df: _st.t.pdf(x, df)),
    "卡方 χ²": ((0, 50), ["自由度"], [(1, 60, 5)],
                lambda x, df: _st.chi2.pdf(x, df)),
    "F 分布": ((0, 6), ["df1", "df2"], [(1, 30, 5), (1, 60, 20)],
              lambda x, d1, d2: _st.f.pdf(x, d1, d2)),
    "Beta": ((0, 1), ["α", "β"], [(0.2, 10, 2), (0.2, 10, 5)],
             lambda x, a, b: _st.beta.pdf(x, a, b)),
    "Gamma": ((0, 12), ["k", "θ"], [(0.2, 10, 2), (0.2, 8, 1)],
              lambda x, k, th: _st.gamma.pdf(x, k, scale=th)),
    "指数": ((0, 6), ["λ"], [(0.2, 5, 1)],
             lambda x, lam: _st.expon.pdf(x, scale=1.0 / lam)),
    "均匀": ((0, 1), ["a", "b"], [(-3, 2, 0), (0.2, 5, 1)],
             lambda x, a, b: _st.uniform.pdf(x, a, b - a)),
}


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
        ctk.CTkButton(left, text="💾 保存 / 导出（PNG SVG PDF EMF）", height=36, fg_color="gray40",
                      command=self._save).grid(
            row=11, column=0, sticky="ew", padx=12, pady=(0, 4))
        ctk.CTkLabel(left, text="导出时可选矢量格式 SVG/PDF/EMF（Word 可直接插入）",
                     font=ctk.CTkFont(size=11), text_color="gray45").grid(
            row=12, column=0, sticky="w", padx=14, pady=(0, 12))

        # 交互函数的滑杆（仅在「交互函数」类型下显示）
        self.slider_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.slider_frame.grid(row=13, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.a_slider = None

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

    def _add_row(self, label, default="", autoplot=False):
        r = len(self.dyn_rows)
        ctk.CTkLabel(self.dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        e = ctk.CTkEntry(self.dyn, height=32)
        if default:
            e.insert(0, default)
        e.grid(row=r, column=1, sticky="ew", pady=3)
        if autoplot:
            e.bind("<KeyRelease>", lambda _ev: self._schedule_autoplot())
        self.dyn_rows.append((label, e))

    # ---- 交互函数：滑杆 + 输入即出图（v1.6.0 P0） ----
    def _clear_slider(self):
        for w in self.slider_frame.winfo_children():
            w.destroy()
        self.a_slider = None
        self._il_line = None
        self._dist_name = None
        self._dist_vals = []
        self._dist_labels = []

    def _build_slider(self):
        """为交互函数/曲面创建单参数滑杆（参数名与范围由调用者在 _switch 里先注入）。"""
        self._clear_slider()
        pname = getattr(self, "_slider_pname", None) or "a"
        v = getattr(self, "_slider_range", None) or []
        lo, hi = (v[0], v[1]) if len(v) >= 2 else (-5, 5)
        if not all(isinstance(x, (int, float)) for x in (lo, hi)):
            lo, hi = -5, 5
        if hi <= lo:
            hi = lo + 1
        self._a_name = pname
        self._a_val = ctk.CTkLabel(self.slider_frame, text=f"{pname} = 0")
        self._a_val.pack(anchor="w", pady=(2, 0))
        self._a_var = ctk.DoubleVar(value=0)
        self.a_slider = ctk.CTkSlider(self.slider_frame, from_=lo, to=hi,
                                      number_of_steps=240, variable=self._a_var,
                                      command=self._live_update)
        self.a_slider.pack(fill="x", pady=(2, 0))

    def _add_optrow(self, label, values, command=None, default=None):
        """加一行下拉选择（用于分布类型等动态参数）。"""
        r = len(self.dyn_rows)
        ctk.CTkLabel(self.dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        om = ctk.CTkOptionMenu(self.dyn, values=values, width=200, command=command)
        if default:
            om.set(default)
        om.grid(row=r, column=1, sticky="ew", pady=3)
        self.dyn_rows.append((label, om))

    def _schedule_autoplot(self):
        """输入即出图：停止上一次计时，500ms 无输入后才重绘（防抖）。"""
        if getattr(self, "_auto_job", None):
            try:
                self.after_cancel(self._auto_job)
            except Exception:
                pass
        self._auto_job = self.after(500, self._auto_plot)

    def _auto_plot(self):
        """输入即出图：表达式合法才重绘；编辑中途非法则保留上一次成功图形。"""
        t = self.type.get()
        if t not in (TYPES[0], TYPES[13]):
            return
        try:
            self.figure.clear()
            self._il_line = None
            if t == TYPES[13]:
                self._pinteractive()
            else:
                self._p2d()
            self._apply_common()
        except Exception:
            return
        self.canvas.draw()

    def _pinteractive(self, a=None):
        """交互函数：y=f(x,a)。只更新曲线、不重建坐标轴，以保留用户的缩放/平移。"""
        fexpr = self.dyn_rows[0][1].get().strip()
        v = nums(self.dyn_rows[1][1].get())
        a0, a1 = (v[0], v[1]) if len(v) >= 2 else (-10, 10)
        pname = getattr(self, "_a_name", None) or (self.dyn_rows[2][1].get().strip() or "a")
        if a is None:
            a = float(self._a_var.get()) if self.a_slider is not None else 0.0
        xs = np.linspace(a0, a1, 400)
        ys = np.array([evalf_subs(fexpr, {"x": x, pname: a}) for x in xs])
        ys = np.where(np.isfinite(ys), ys, np.nan)
        self.ax = self.figure.add_subplot(111)
        (self._il_line,) = self.ax.plot(xs, ys, lw=2, color=ps.color(0))
        self._il_xs = xs
        self._il_pname = pname

    def _live_update(self, _v=None):
        t = self.type.get()
        if self.a_slider is None:
            return
        a = float(self._a_var.get())
        try:
            self._a_val.configure(text=f"{self._a_name} = {a:.3g}")
        except Exception:
            pass
        if t == TYPES[15]:           # 曲面：全量重绘（自动缩放取值域）
            self._pinteractive_surf(a)
            self._apply_common()
            self.canvas.draw_idle()
            return
        if t != TYPES[13]:
            return
        if getattr(self, "_il_line", None) is None:
            self._pinteractive(a)
            self._apply_common()
            self.canvas.draw()
            return
        fexpr = self.dyn_rows[0][1].get().strip()
        pname = getattr(self, "_il_pname", "a")
        ys = np.array([evalf_subs(fexpr, {"x": x, pname: a}) for x in self._il_xs])
        ys = np.where(np.isfinite(ys), ys, np.nan)
        self._il_line.set_ydata(ys)
        try:
            self._apply_common()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _build_dist(self):
        """按所选分布建参数滑杆（GeoGebra 式）：分布 pdf + 每组参数一根滑杆。"""
        self._clear_slider()
        name = self.dyn_rows[0][1].get()
        xrange, pnames, ranges, pdf = DIST[name]
        self._dist_name, self._dist_pdf, self._dist_pnames = name, pdf, pnames
        self._dist_min, self._dist_max = float(xrange[0]), float(xrange[1])
        try:
            e = self.dyn_rows[1][1]
            e.delete(0, "end")
            e.insert(0, f"{self._dist_min} {self._dist_max}")
        except Exception:
            pass
        self._dist_vals, self._dist_labels = [], []
        for pn, rng in zip(pnames, ranges):
            lo, hi, dflt = rng
            lab = ctk.CTkLabel(self.slider_frame, text=f"{pn} = {dflt:.3g}")
            lab.pack(anchor="w", pady=(2, 0))
            var = ctk.DoubleVar(value=dflt)
            sl = ctk.CTkSlider(self.slider_frame, from_=lo, to=hi, number_of_steps=320,
                               variable=var, command=self._live_dist)
            sl.pack(fill="x", pady=(2, 0))
            self._dist_vals.append(var)
            self._dist_labels.append(lab)
        self._redraw_dist()

    def _live_dist(self, _v=None):
        name = getattr(self, "_dist_name", None)
        if not name or not getattr(self, "_dist_vals", None):
            return
        for pn, var, lab in zip(self._dist_pnames, self._dist_vals, self._dist_labels):
            lab.configure(text=f"{pn} = {var.get():.3g}")
        self._redraw_dist()

    def _redraw_dist(self, params=None):
        """绘制当前分布 pdf（含填充），滑杆移动时重绘并自动适应取值域。"""
        name = getattr(self, "_dist_name", None)
        if not name:
            return
        params = params if params is not None else [float(v.get()) for v in self._dist_vals]
        if not params:
            return
        x = np.linspace(self._dist_min, self._dist_max, 600)
        y = np.asarray(self._dist_pdf(x, *params), dtype=float)
        y = np.where(np.isfinite(y), y, np.nan)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x, y, lw=2, color=ps.color(0))
        self.ax.fill_between(x, y, color=ps.color(0), alpha=0.18)
        self.ax.set_xlim(self._dist_min, self._dist_max)
        self._apply_common()
        self.canvas.draw_idle()

    def _pinteractive_surf(self, a=None):
        """曲面：z=f(x,y,a)，a 由滑杆控制（全量重绘 + 3D 旋转）。"""
        fexpr = self.dyn_rows[0][1].get().strip()
        v = nums(self.dyn_rows[1][1].get())
        x0, x1 = (v[0], v[1]) if len(v) >= 2 else (-5, 5)
        w = nums(self.dyn_rows[2][1].get())
        y0, y1 = (w[0], w[1]) if len(w) >= 2 else (-5, 5)
        pname = getattr(self, "_a_name", None) or (self.dyn_rows[3][1].get().strip() or "a")
        if a is None:
            a = float(self._a_var.get()) if self.a_slider is not None else 0.0
        x = np.linspace(x0, x1, 70)
        y = np.linspace(y0, y1, 60)
        X, Y = np.meshgrid(x, y)
        Z = eval2_subs(fexpr, X, Y, {pname: a})
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
        self.ax.set_zlabel("Z")

    def _switch(self):
        self._clear_dyn()
        self._clear_slider()
        t = self.type.get()
        if t == TYPES[0]:          # y=f(x)，支持多个函数（分号或换行分隔）
            self._add_row("f(x)", "sin(x)", autoplot=True)
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
        elif t == TYPES[10]:       # 小提琴图（分组分布）
            self._add_row("组数据", "A: 12 15 18 20; B: 10 14 19 25")
        elif t == TYPES[11]:       # 误差棒图（均值 ± SD/SE）
            self._add_row("组数据", "A: 12 15 18 20; B: 10 14 19 25")
            self._add_row("误差类型", "SD  /  SE")
        elif t == TYPES[12]:       # 森林图（估计值 下限 上限）
            self._add_row("估计数据", "A: 0.5 0.3 0.8; B: 1.2 0.4 2.0")
        elif t == TYPES[13]:       # 交互函数：y=f(x,a)，a 由滑杆实时调控
            self._add_row("f(x,a)", "a*sin(x)", autoplot=True)
            self._add_row("x 范围", "-10 10")
            self._add_row("a 名称", "a")
            self._add_row("a 范围", "-5 5")
            self._slider_pname = self.dyn_rows[2][1].get().strip() or "a"
            self._slider_range = nums(self.dyn_rows[3][1].get())
            self._build_slider()
        elif t == TYPES[14]:       # 分布滑杆（GeoGebra 式）
            self._add_optrow("分布", list(DIST.keys()),
                             command=lambda _v: self._build_dist(), default="正态")
            self._add_row("x 范围", "-8 8")
            self._build_dist()
        elif t == TYPES[15]:       # 曲面：z=f(x,y,a)，a 由滑杆实时调控
            self._add_row("f(x,y,a)", "a*sin(x)+cos(y)")
            self._add_row("x 范围", "-5 5")
            self._add_row("y 范围", "-5 5")
            self._add_row("a 名称", "a")
            self._add_row("a 范围", "-2 2")
            self._slider_pname = self.dyn_rows[3][1].get().strip() or "a"
            self._slider_range = nums(self.dyn_rows[4][1].get())
            self._build_slider()

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
        # 3D 图：启用鼠标拖拽旋转（默认左键转视角、右键平移）
        if hasattr(self.ax, "mouse_init"):
            try:
                self.ax.mouse_init(rotate_btn=1, pan_btn=3)
            except Exception:
                pass

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
            elif t == TYPES[10]:
                self._pviolin()
            elif t == TYPES[11]:
                self._perrbar()
            elif t == TYPES[12]:
                self._pforest()
            elif t == TYPES[13]:
                self._pinteractive()
            elif t == TYPES[14]:
                self._redraw_dist()
            elif t == TYPES[15]:
                self._pinteractive_surf()
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

    # ================= v1.5 新图型 =================
    @staticmethod
    def _parse_groups(text):
        """解析 `A: 1 2 3; B: 4 5 6` → [(name, [values]), ...]。"""
        groups = []
        for part in text.replace("；", ";").replace("，", ",").split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, vals = part.split(":", 1)
            else:
                name, vals = "组", part
            groups.append((name.strip(), nums(vals)))
        return [g for g in groups if g[1]]

    def _pviolin(self):
        groups = self._parse_groups(self.dyn_rows[0][1].get())
        names = [g[0] for g in groups]
        vals = [np.array(g[1]) for g in groups]
        self.ax = self.figure.add_subplot(111)
        parts = self.ax.violinplot(vals, positions=range(len(groups)), showmedians=False,
                                   showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(ps.color(i))
            body.set_alpha(0.55)
            body.set_edgecolor("white")
        bp = self.ax.boxplot(vals, positions=range(len(groups)), widths=0.10,
                             showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", lw=1.6))
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.85)
        self.ax.set_xticks(range(len(groups)))
        self.ax.set_xticklabels(names)

    def _perrbar(self):
        groups = self._parse_groups(self.dyn_rows[0][1].get())
        names = [g[0] for g in groups]
        use_se = str(self.dyn_rows[1][1].get()).strip().upper().startswith("SE")
        means = [float(np.mean(g[1])) for g in groups]
        errs = [(float(np.std(g[1], ddof=1)) / np.sqrt(len(g[1])) if use_se else float(np.std(g[1], ddof=1)))
                for g in groups]
        self.ax = self.figure.add_subplot(111)
        colors = ps.palette(len(groups))
        self.ax.errorbar(range(len(groups)), means, yerr=errs, fmt="o-", capsize=4,
                         lw=1.8, markersize=7, color=colors[0], ecolor=colors[1] if len(colors) > 1 else colors[0])
        self.ax.set_xticks(range(len(groups)))
        self.ax.set_xticklabels(names)
        self.ax.set_ylabel("均值 ± " + ("SE" if use_se else "SD"))

    def _pforest(self):
        est, lo, hi, labels = [], [], [], []
        for name, vals in self._parse_groups(self.dyn_rows[0][1].get()):
            if len(vals) < 3:
                continue
            labels.append(name)
            est.append(vals[0])
            lo.append(vals[1])
            hi.append(vals[2])
        if not est:
            raise ValueError("森林图需要至少一组「估计值 下限 上限」数据。")
        self.ax = self.figure.add_subplot(111)
        ypos = np.arange(len(est))[::-1]
        colors = ps.palette(len(est))
        for i, (y, e, l, h) in enumerate(zip(ypos, est, lo, hi)):
            self.ax.plot([l, h], [y, y], color=colors[i], lw=3.2, solid_capstyle="round")
            self.ax.scatter([e], [y], s=54, color=colors[i], zorder=3, edgecolor="white", linewidth=0.8)
        self.ax.axvline(0, color="gray", ls="--", lw=1)
        self.ax.set_yticks(ypos)
        self.ax.set_yticklabels(labels)
        for y, e, l, h in zip(ypos, est, lo, hi):
            self.ax.text(h, y, f"  {e:.3g} [{l:.3g}, {h:.3g}]", va="center", fontsize=9)

    def _save(self):
        from tkinter import filedialog
        f = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=ps.export_path_filters)
        if f:
            try:
                ps.export_figure(self.figure, f, dpi=300)
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("导出失败", str(e))