# -*- coding: utf-8 -*-
"""公式计算器：求值 / 求导 / 积分 / 解方程 / 求极限（基于 sympy）"""

import customtkinter as ctk
from sympy import (symbols, sympify, diff, integrate, solve, limit, oo,
                   nsimplify, Symbol, pretty, Rational, latex)
from modules.latex_view import sympy_to_image

MODE_LABELS = ["数值求值", "求导", "积分", "解方程", "求极限"]

# 常用示例，便于不写代码的人照着改
EXAMPLES = {
    "数值求值": "sqrt(2) + 3*sin(pi/6)",
    "求导": "x**3 * sin(x)",
    "积分": "1/(1 + x**2)",
    "解方程": "x**2 - 5*x + 6 = 0",
    "求极限": "sin(x) / x",
}


def _norm(text):
    """把用户输入的 ^ 统一转成 **，并去掉多余空白。"""
    return text.strip().replace("^", "**")


def _to_expr(text):
    return sympify(_norm(text), evaluate=True)


class CalculatorPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # ---------- 左：控制区 ----------
        left = ctk.CTkFrame(self, width=360, corner_radius=12)
        left.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=12, pady=12)
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="选择计算类型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        self.mode = ctk.CTkSegmentedButton(
            left, values=MODE_LABELS, command=lambda _v: self._switch_mode())
        self.mode.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))

        # 表达式
        ctk.CTkLabel(left, text="表达式（可使用变量 x，支持 + - * / ** 、括号、函数）",
                     font=ctk.CTkFont(size=12), text_color="gray50").grid(
            row=2, column=0, sticky="w", padx=16, pady=(4, 2))
        self.expr = ctk.CTkEntry(left, height=42, font=ctk.CTkFont(size=16))
        self.expr.grid(row=3, column=0, sticky="ew", padx=16)
        self.expr.insert(0, EXAMPLES["数值求值"])

        # 附加参数区（随模式变化）
        self.extra = ctk.CTkFrame(left, fg_color="transparent")
        self.extra.grid(row=4, column=0, sticky="ew", padx=16, pady=10)
        self.extra.grid_columnconfigure(0, weight=0)
        self.extra.grid_columnconfigure(1, weight=1)
        # 用两个标签 + 两个输入框容纳最多两个附加参数
        self.lab1 = ctk.CTkLabel(self.extra, text="", font=ctk.CTkFont(size=12))
        self.lab1.grid(row=0, column=0, sticky="w")
        self.inp1 = ctk.CTkEntry(self.extra, height=34)
        self.inp1.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.lab2 = ctk.CTkLabel(self.extra, text="", font=ctk.CTkFont(size=12))
        self.lab2.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.inp2 = ctk.CTkEntry(self.extra, height=34)
        self.inp2.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self._switch_mode()

        # 按钮行
        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.grid(row=5, column=0, sticky="ew", padx=16, pady=(4, 4))
        btns.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btns, text="⚡ 计 算", height=40, command=self._run).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(btns, text="示例", fg_color="gray40", height=40,
                      command=self._example).grid(row=0, column=1, sticky="ew")

        # 常用函数说明
        tip = ("常用写法：\n"
               "加减乘除      + - * /\n"
               "乘方          x**2 或 x^2\n"
               "平方根        sqrt(x)\n"
               "自然对数      ln(x)\n"
               "常用对数      log(x, 10)\n"
               "指数          exp(x)\n"
               "三角          sin cos tan asin acos atan\n"
               "常数          pi  e\n"
               "求值时可附带变量，如：x=2, y=3")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=12),
                     text_color="gray45", anchor="w").grid(
            row=6, column=0, sticky="w", padx=16, pady=(2, 14))

        # ---------- 右：结果区 ----------
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(right, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="计算结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w")
        self.ltx_btn = ctk.CTkButton(hdr, text="📋 复制 LaTeX", width=120, height=28,
                                     fg_color="gray40", command=self._copy_latex)
        self.ltx_btn.grid(row=0, column=1, sticky="e")

        # 专业公式预览（LaTeX 渲染）
        self.pv_frame = ctk.CTkFrame(right, fg_color="white", corner_radius=8, height=160)
        self.pv_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        self.pv_frame.grid_propagate(False)
        self.pv_frame.grid_rowconfigure(0, weight=1)
        self.pv_frame.grid_columnconfigure(0, weight=1)
        self.pv_label = ctk.CTkLabel(self.pv_frame, text="公式预览",
                                     font=ctk.CTkFont(size=13), text_color="gray50")
        self.pv_label.grid(row=0, column=0)

        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=15))
        self.out.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self._pv = None          # 待渲染预览的表达式
        self._last_latex = None  # 上一次结果的 LaTeX
        self._pv_ctk = None

    # ---------- 模式切换：决定显示哪些附加参数 ----------
    def _switch_mode(self):
        m = self.mode.get()
        self.lab1.grid_remove(); self.inp1.grid_remove()
        self.lab2.grid_remove(); self.inp2.grid_remove()
        if m == "求值":
            self.lab1.configure(text="变量取值（可选，逗号分隔）"); self.inp1.configure(placeholder_text="如：x=2, y=3")
            self.lab1.grid(); self.inp1.grid()
        elif m == "求导":
            self.lab1.configure(text="自变量"); self.inp1.configure(placeholder_text="x")
            self.lab1.grid(); self.inp1.grid()
        elif m == "积分":
            self.lab1.configure(text="下限（空 = 不定积分）"); self.inp1.configure(placeholder_text="如：0 或 -oo")
            self.lab2.configure(text="上限（空 = 不定积分）"); self.inp2.configure(placeholder_text="如：1 或 oo")
            self.lab1.grid(); self.inp1.grid(); self.lab2.grid(); self.inp2.grid()
        elif m == "解方程":
            self.lab1.configure(text="未知数"); self.inp1.configure(placeholder_text="x")
            self.lab1.grid(); self.inp1.grid()
        elif m == "求极限":
            self.lab1.configure(text="x 趋近值"); self.inp1.configure(placeholder_text="0")
            self.lab2.configure(text="方向（可留空）"); self.inp2.configure(placeholder_text="双向 / + / -")
            self.lab1.grid(); self.inp1.grid(); self.lab2.grid(); self.inp2.grid()

    # ---------- 填入当前模式的示例 ----------
    def _example(self):
        m = self.mode.get()
        self.expr.delete(0, "end"); self.expr.insert(0, EXAMPLES[m])
        self.out.delete("1.0", "end")

    # ---------- 运行计算 ----------
    def _run(self):
        self.out.delete("1.0", "end")
        self._pv = None
        self._out_text = ""
        m = self.mode.get()
        expr_text = self.expr.get().strip()
        if not expr_text:
            self._msg("请输入表达式。"); return
        try:
            if m == "数值求值":
                self._do_eval(expr_text)
            elif m == "求导":
                self._do_diff(expr_text)
            elif m == "积分":
                self._do_integrate(expr_text)
            elif m == "解方程":
                self._do_solve(expr_text)
            elif m == "求极限":
                self._do_limit(expr_text)
        except Exception as e:
            self._msg(f"出错：{e}\n\n请检查表达式是否书写正确。", red=True)
        self._show_preview()
        # 记录历史
        if getattr(self, "history", None):
            self.history.log_calc(m, expr_text,
                                  getattr(self, "_out_text", ""),
                                  self._last_latex)

    # ---- 求值 ----
    def _do_eval(self, text):
        assigns = self.inp1.get().strip()
        subs = {}
        if assigns:
            for part in assigns.replace("；", ";").split(";"):
                for kv in part.split(","):
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    subs[symbols(k.strip())] = sympify(_norm(v))
        e = _to_expr(text)
        e = e.subs(subs)
        val = e.evalf()
        self._pv = e
        self._msg("表达式（精确）：")
        self._msg(pretty(e))
        self._msg("\n数值结果：")
        self._msg(f"≈ {float(val):.10g}")
        self._msg(f"\n（如整数/分数，见上方精确结果）")

    # ---- 求导 ----
    def _do_diff(self, text):
        var = self.inp1.get().strip() or "x"
        s = symbols(var)
        e = _to_expr(text)
        d = diff(e, s)
        self._pv = d
        self._msg("f(x) = " + pretty(e))
        self._msg("\nf'(x) = " + pretty(d))

    # ---- 积分 ----
    def _do_integrate(self, text):
        a = self.inp1.get().strip()
        b = self.inp2.get().strip()
        e = _to_expr(text)
        if a and b:
            lo = oo if a in ("oo", "inf") else (-oo if a in ("-oo", "-inf") else _to_expr(a))
            hi = oo if b in ("oo", "inf") else (-oo if b in ("-oo", "-inf") else _to_expr(b))
            r = integrate(e, (symbols("x"), lo, hi))
            self._pv = r
            self._msg("定积分结果（精确）：")
            self._msg(pretty(r))
            self._msg(f"\n数值 ≈ {float(r.evalf()):.10g}")
        else:
            r = integrate(e, symbols("x"))
            self._pv = r
            self._msg("不定积分（原函数）：")
            self._msg(pretty(r) + "  +  C")

    # ---- 解方程 ----
    def _do_solve(self, text):
        var = self.inp1.get().strip() or "x"
        s = symbols(var)
        t = _norm(text)
        if "=" in t:
            lhs, rhs = t.split("=", 1)
            eq = sympify(lhs) - sympify(rhs)
        else:
            eq = sympify(t)
        sols = solve(eq, s)
        self._pv = sols[0] if sols else None
        self._msg("方程的解：")
        if not sols:
            self._msg("（无解或解为空）")
        else:
            for i, sol in enumerate(sols, 1):
                self._msg(f"x{i} = {pretty(sol)}")

    # ---- 求极限 ----
    def _do_limit(self, text):
        point = self.inp1.get().strip() or "0"
        dir_ = self.inp2.get().strip()
        e = _to_expr(text)
        x = symbols("x")
        if point in ("oo", "inf"):
            pt = oo
        elif point in ("-oo", "-inf"):
            pt = -oo
        else:
            pt = _to_expr(point)
        if dir_ == "+":
            r = limit(e, x, pt, dir="+")
        elif dir_ == "-":
            r = limit(e, x, pt, dir="-")
        else:
            r = limit(e, x, pt)
        self._pv = r
        self._msg("极限值：")
        self._msg(pretty(r))
        if r.is_finite:
            self._msg(f"\n数值 ≈ {float(r):.10g}")

    # ---- 写结果 ----
    def _msg(self, s, red=False):
        self.out.insert("end", s + "\n")
        if not hasattr(self, "_out_text"):
            self._out_text = ""
        self._out_text += s + "\n"

    # ---- LaTeX 公式预览 ----
    def _show_preview(self):
        e = self._pv
        self._pv = None
        if e is None:
            self.pv_label.configure(image=None, text="公式预览")
            return
        try:
            self._last_latex = latex(e)
            img = sympy_to_image(e)
            scale = min(1.0, 150 / img.height)
            w = max(1, int(img.width * scale))
            h = max(1, int(img.height * scale))
            self._pv_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.pv_label.configure(image=self._pv_ctk, text="")
        except Exception:
            self.pv_label.configure(image=None, text="（公式预览失败）")

    def _copy_latex(self):
        if self._last_latex:
            self.clipboard_clear()
            self.clipboard_append(self._last_latex)
            self.ltx_btn.configure(text="✅ 已复制")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))
        else:
            self.ltx_btn.configure(text="暂无内容")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))