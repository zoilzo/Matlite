# -*- coding: utf-8 -*-
"""复变函数：复数运算 / 模曲面 / 复平面着色映射 / 泰勒展开 / 留数。

全中文界面，面向不懂代码的学生。前端沿用左参数面板 + 右结果/画布布局。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colors as mcolors
import customtkinter as ctk

from modules.expr_utils import nums

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SAFE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sqrt": np.sqrt, "abs": abs, "exp": np.exp, "log": np.log,
    "log10": np.log10, "log2": np.log2, "pi": np.pi, "e": np.e,
    "I": 1j, "i": 1j, "j": 1j,
}

OPS = ["求值 f(z) 在某点", "模曲面 |f(z)|（3D）", "复平面着色映射", "泰勒展开（sympy）", "留数（sympy）"]


def _to_complex(s):
    t = s.strip().replace("i", "j").replace("^", "**").replace("，", ",").replace(" ", "")
    try:
        return complex(eval(t, {"__builtins__": {}}, SAFE))
    except Exception:
        return None


def _parse_range(t):
    return [float(x) for x in t.replace(",", " ").replace("。", " ").split()]


def _func_ptr(expr):
    import sympy as sp
    z = sp.symbols("z")
    e = sp.sympify(expr.replace("^", "**"), {"I": sp.I, "i": sp.I, "j": sp.I, "pi": sp.pi, "e": sp.E})
    f = sp.lambdify(z, e, modules=[{"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
                                    "log": np.log, "sqrt": np.sqrt, "Abs": np.abs}, "numpy", "math"])
    return f, e, sp


class ComplexPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数面板 =================
        left = ctk.CTkScrollableFrame(self, width=380, corner_radius=14, label_text="复变函数")
        left.grid(row=0, column=0, sticky="nsew", padx=13, pady=9)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="函数 f(z)", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 2))
        self.fexpr = ctk.CTkEntry(left, height=34)
        self.fexpr.grid(row=1, column=0, sticky="ew", padx=9)
        self.fexpr.insert(0, "z**2 + 1")

        ctk.CTkLabel(left, text="操作", font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=12, pady=(10, 4))
        self.op = ctk.CTkOptionMenu(left, values=OPS)
        self.op.grid(row=3, column=0, sticky="ew", padx=9)
        self.op.set(OPS[1])

        ctk.CTkLabel(left, text="求值点 z0（如 1+2i）", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="w", padx=12, pady=(8, 2))
        self.z0 = ctk.CTkEntry(left, height=32)
        self.z0.grid(row=5, column=0, sticky="ew", padx=12)
        self.z0.insert(0, "1+i")

        ctk.CTkLabel(left, text="区域范围（用于绘图）", font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, sticky="w", padx=14, pady=(8, 2))
        self.xr = ctk.CTkEntry(left, height=32)
        self.xr.grid(row=7, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.xr.insert(0, "-2 2")
        self.yr = ctk.CTkEntry(left, height=32)
        self.yr.grid(row=8, column=0, sticky="ew", padx=12)
        self.yr.insert(0, "-2 2")

        ctk.CTkButton(left, text="⚡ 计 算", height=40, command=self._run).grid(
            row=9, column=0, sticky="ew", padx=12, pady=(10, 4))

        tip = ("说明：\n"
               "· 坐标写法：1+i、2-3i、i（即 0+1i）；也可写 2+3j；\n"
               "· f(z) 里可用 z 复数变量，如 z**2、exp(z)、1/z；\n"
               "· 着色映射用 色调=辐角、亮度=模 直观展示保角性质；\n"
               "· 泰勒/留数需要本机 sympy（已内置）。\n"
               "· 留数：自动列出全部奇点并对每个奇点求留数；\n"
               "  也可在「求值点 z0」填任意点（如 1+i）求该点留数。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=350).grid(
            row=10, column=0, sticky="w", padx=14, pady=(6, 12))

        # ================= 右：结果 + 画布 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(right, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="计算结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w")
        self.repro_btn = ctk.CTkButton(hdr, text="📦 导出复现", width=120, height=28,
                                       fg_color="steelblue", command=self._export_repro)
        self.repro_btn.grid(row=0, column=1, sticky="e", padx=(12, 0))
        hdr.grid_columnconfigure(1, weight=0)
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
        self.canvas.draw()

    def _clear_out(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")

    def _msg(self, s):
        self.out.configure(state="normal")
        self.out.insert("end", s + "\n")
        self.out.see("end")
        self.out.configure(state="disabled")

    def _run(self):
        self._clear_out()
        op = self.op.get()
        try:
            f, e, sp = _func_ptr(self.fexpr.get())
            if op == OPS[0]:
                z = _to_complex(self.z0.get())
                if z is None:
                    self._msg("无法解析 z0，请用 1+2i 或 3-4j 格式。")
                else:
                    w = f(z)
                    self._msg(f"f({z}) = {w}")
                    self._msg(f"|f({z})| = {abs(w):.4g}；arg = {np.angle(w):.4g} rad = {np.angle(w)*180/np.pi:.4g}°")
            elif op == OPS[1]:
                self._surf(f)
            elif op == OPS[2]:
                self._domain_color(f)
            elif op == OPS[3]:
                z0c = _to_complex(self.z0.get()) or 0j
                try:
                    a = complex(z0c).real
                    series = sp.series(e, sp.symbols("z"), a, 7).removeO()
                    self._msg(f"在 z={z0c} 处展开：\n{series}")
                except Exception as err:
                    self._msg(f"展开失败：{err}")
            elif op == OPS[4]:
                try:
                    z = sp.symbols("z")
                    # 1) 先列出全部奇点
                    try:
                        sing = sp.singularities(e, z)
                        sing_list = list(sing) if sing else []
                    except Exception:
                        sing_list = []
                    if sing_list:
                        self._msg(f"全部奇点：{sing_list}")
                    else:
                        self._msg("未解析出孤立奇点（该函数可能在复平面解析，或奇点为支点/本性奇点）。")
                    # 2) 对每个孤立奇点逐一求留数（任意奇点留数）
                    for pt in sing_list:
                        try:
                            r = sp.residue(e, z, pt)
                            self._msg(f"  在 z = {pt} 处的留数 = {r}")
                        except Exception as err:
                            self._msg(f"  在 z = {pt} 处留数计算失败：{err}")
                    # 3) 用户指定任意点 z0（覆盖默认）
                    z0txt = self.z0.get().strip()
                    z0c = _to_complex(z0txt) if z0txt else None
                    if z0c is not None:
                        try:
                            r = sp.residue(e, z, z0c)
                            self._msg(f"在指定点 z = {z0c} 处的留数 = {r}")
                        except Exception as err:
                            self._msg(f"在指定点 z = {z0c} 处计算失败：{err}")
                    elif not sing_list:
                        # 无显式奇点且未指定 z0，退化为在 z=0 求留数
                        r = sp.residue(e, z, 0)
                        self._msg(f"在 z = 0 处的留数 = {r}")
                except Exception as err:
                    self._msg(f"留数计算失败：{err}")
        except Exception as err:
            self._msg(f"出错：{err}")
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("复变函数", op, self.out.get("1.0", "end")[:200])

    def _surf(self, f):
        xr = _parse_range(self.xr.get())
        yr = _parse_range(self.yr.get())
        if len(xr) < 2 or len(yr) < 2:
            self._msg("区域范围需两个数，如 -2 2。")
            return
        n = 300
        x = np.linspace(xr[0], xr[1], n)
        y = np.linspace(yr[0], yr[1], n)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        with np.errstate(all="ignore"):
            W = f(Z)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.plot_surface(X, Y, np.abs(W), cmap="viridis", linewidth=0, antialiased=True)
        self.ax.set_title("|f(z)| 曲面")
        self.ax.set_xlabel("Re z")
        self.ax.set_ylabel("Im z")
        self.ax.set_zlabel("|f(z)|")

    def _domain_color(self, f):
        xr = _parse_range(self.xr.get())
        yr = _parse_range(self.yr.get())
        if len(xr) < 2 or len(yr) < 2:
            self._msg("区域范围需两个数，如 -2 2。")
            return
        n = 350
        x = np.linspace(xr[0], xr[1], n)
        y = np.linspace(yr[0], yr[1], n)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        with np.errstate(all="ignore"):
            W = f(Z)
        H = (np.angle(W) / (2 * np.pi)) % 1.0
        S = np.ones_like(H, dtype=float)
        V = np.abs(W) / (1 + np.abs(W))
        rgb = mcolors.hsv_to_rgb(np.stack([H, S, V], axis=-1))
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(rgb, origin="lower", extent=[xr[0], xr[1], yr[0], yr[1]])
        self.ax.set_title("复平面着色映射（色调=arg, 亮度=|f(z)|）")
        self.ax.set_xlabel("Re z")
        self.ax.set_ylabel("Im z")

    def _export_repro(self):
        from modules import repro
        fexpr = self.fexpr.get().strip() or "z**2 + 1"
        z0 = self.z0.get().strip()
        op = self.op.get()
        prob = [f"f(z) = {fexpr}", f"操作：{op}"]
        if z0:
            prob.append(f"指定点 z0 = {z0}")

        code = [repro.script_header(f"复变函数 · {op}", "复变函数"),
                "import sympy as sp", "import numpy as np", "",
                "z = sp.symbols('z')",
                "f = sp.sympify('''" + fexpr.replace("\'", "'") + "''' .replace('^', '**'), "
                "       {'I': sp.I, 'i': sp.I, 'j': sp.I, 'pi': sp.pi, 'e': sp.E})", ""]
        code += ["try:",
                 "    sing = list(sp.singularities(f, z)) if sp.singularities(f, z) else []",
                 "except Exception:",
                 "    sing = []",
                 "print('全部奇点：', sing if sing else '（无孤立奇点）')",
                 "print('各奇点留数：')",
                 "for pt in sing:",
                 "    try: print(f'  z = {pt} 处留数 = {sp.residue(f, z, pt)}')",
                 "    except Exception as err: print(f'  在 z = {pt} 处计算失败：{err}')"]
        if z0:
            zsym = z0.replace("i", "I").replace("j", "I").replace("^", "**")
            code += ["print('指定点留数：')",
                     "try: print(f'  z = " + z0 + " 处留数 = {sp.residue(f, z, sp.sympify(\'" + zsym + "\'))}')",
                     "except Exception as err: print(f'  在指定点计算失败：{err}')"]
        py_code = "\n".join(code) + "\n"
        result = [ln for ln in self.out.get("1.0", "end").splitlines() if ln.strip()][-10:]
        repro.export_recipe(self, f"复变函数 · {op}", "复变函数", prob, result, py_code,
                            default_name=repro._safe_name(f"复变-{op}"),
                            history=getattr(self, "history", None))
