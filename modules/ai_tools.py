# -*- coding: utf-8 -*-
"""AI 助手工具注册表：把 MatLite 内置功能模块的计算能力封装成可被大模型调用的工具。

每个工具包含：
  name / description / parameters(JSON Schema) / func / returns_image / category

入口：run_tool(name, args) -> {"text": str, "image": PIL.Image 可选}
所有工具均为纯逻辑函数，不碰任何 GUI 控件，可在后台线程执行。
绘图统一走 Agg 后端（无窗口），渲染成 PIL 图片返回。
"""

import io
import os
import sys
import ast
import requests

# 本地/远程 AI 服务一律直连，不走系统代理（与 ai_page/announce 一致）
_NO_PROXY = {"http": None, "https": None}

# 项目根目录（stats_utils/stats_plots/stats_report 在根目录而非 modules/ 下）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 所有绘图一律用 Agg 后端（无 GUI），页面模块的 TkAgg 不受影响
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sympy import (sympify, symbols, diff, integrate, solve, limit, latex, Matrix,
                   lambdify, pretty, simplify, oo, pi, E, I, Rational, Symbol, Eq,
                   dsolve, Function, series, residue, singularities)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 工具注册表
# ============================================================
_TOOLS = {}


def _reg(fn):
    """装饰器：把函数注册为工具。函数须带 __tool_params__/__tool_image__/__tool_category__ 属性。"""
    _TOOLS[fn.__name__] = {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": fn.__tool_params__,
        "func": fn,
        "returns_image": bool(getattr(fn, "__tool_image__", False)),
        "category": getattr(fn, "__tool_category__", "通用"),
    }
    return fn


def _tool(params, image=False, category="通用"):
    """工具属性标注：params 是 JSON Schema 的 properties 与 required。"""
    def deco(fn):
        fn.__tool_params__ = params
        fn.__tool_image__ = image
        fn.__tool_category__ = category
        return fn
    return deco


def all_tools():
    """返回全部工具（含工具名清单，供模型参考）。"""
    return list(_TOOLS.values())


def run_tool(name, args):
    """执行工具，统一返回 {"text": ..., "image": ...可选}。异常转中文错误文本。"""
    t = _TOOLS.get(name)
    if t is None:
        return {"text": f"没有名为 {name} 的工具。"}
    try:
        out = t["func"](**(args or {}))
        if isinstance(out, dict):
            return out
        if isinstance(out, str):
            return {"text": out}
        return {"text": str(out)}
    except Exception as e:
        return {"text": f"工具 {name} 执行出错：{e}"}


# ============================================================
# 通用辅助
# ============================================================

def _num(v, default=0.0):
    """稳健数值解析：容忍字符串里的逗号、中文逗号、百分号、尾部 x 后缀等。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("，", "").replace("%", "").strip()
    # 只去掉尾部“x”后缀（如 "2x" → 2），不去掉中间/开头的 x（避免 "2x3" 被错误解析）
    if s.endswith("x") or s.endswith("X"):
        s = s[:-1].strip()
    if s in ("", "-", "无", "空"):
        return default
    try:
        return float(s)
    except Exception:
        return default


def _nums(v):
    """解析数值列表（接受列表/元组或空格逗号分隔字符串）。"""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out
    s = str(v).replace(",", " ").replace("，", " ").replace("；", " ").replace("\n", " ").strip()
    out = []
    for x in s.split():
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _sym(s):
    """字符串转 sympy 表达式（^ 统一转 **）。"""
    return sympify(str(s).replace("^", "**"))


def _to_df(data):
    """把模型传入的数据（CSV 文本 / dict / 嵌套列表 / DataFrame）转成 DataFrame。"""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        cols = {}
        for k, v in data.items():
            cols[str(k)] = list(v) if isinstance(v, (list, tuple)) else _nums(v)
        if not cols:
            raise ValueError("数据为空。")
        return pd.DataFrame(cols)
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        if all(isinstance(x, str) for x in data[0]):
            return pd.DataFrame(data[1:], columns=[str(x) for x in data[0]])
        return pd.DataFrame(data)
    if isinstance(data, str):
        txt = data.strip()
        if not txt:
            raise ValueError("数据为空。")
        try:
            df = pd.read_csv(io.StringIO(txt))
            if df.shape[1] == 1 and len(df) > 1:
                # 逗号读成单列 = 可能是空白分隔：按空白重试
                chk = pd.read_csv(io.StringIO(txt), sep=r"\s+")
                if chk.shape[1] > 1:
                    return chk
            return df
        except Exception:
            rows = [ln.split() for ln in txt.splitlines() if ln.strip()]
            if rows:
                return pd.DataFrame(rows[1:], columns=[str(x) for x in rows[0]])
            raise ValueError("无法解析数据。")
    raise ValueError("无法解析数据，请提供 CSV 文本、字典或嵌套列表。")


def _mat(data):
    """构造 sympy.Matrix：接受嵌套列表或扁平列表（扁平时按方阵）。"""
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        return Matrix([[sympify(str(c).replace("^", "**")) for c in row] for row in data])
    if isinstance(data, (list, tuple)) and data:
        n = len(data)
        cols = int(round(n ** 0.5))
        if cols * cols == n:
            return Matrix([[sympify(str(data[i * cols + j]).replace("^", "**"))
                            for j in range(cols)] for i in range(cols)])
        raise ValueError("扁平列表无法确定方阵形状，请提供嵌套列表（如 [[1,2],[3,4]]）。")
    raise ValueError("矩阵数据必须是列表。矩阵示例：[[1,2],[3,4]]")


def _dist(name, params):
    """按分布名与参数构造 scipy 冻结分布。支持 normal/binom/poisson/expon/uniform/chi2/t/f/beta/gamma。"""
    from scipy import stats
    nm = str(name or "").lower()
    p = params or {}
    if nm in ("normal", "正态", "norm", "n"):
        mu = _num(p.get("mu", p.get("mean", 0)), 0)
        sg = max(1e-9, _num(p.get("sigma", p.get("sd", 1)), 1))
        return stats.norm(mu, sg), f"N({mu:g}, {sg:g}²)"
    if nm in ("binom", "二项", "binomial", "b"):
        n = max(1, int(_num(p.get("n", 10), 10)))
        pr = min(1.0, max(0.0, _num(p.get("p", 0.5), 0.5)))
        return stats.binom(n, pr), f"B({n}, {pr:g})"
    if nm in ("poisson", "泊松", "pois", "p"):
        lam = max(1e-9, _num(p.get("lambda", p.get("lam", 3)), 3))
        return stats.poisson(lam), f"Poisson(λ={lam:g})"
    if nm in ("expon", "指数", "e"):
        sc = max(1e-9, _num(p.get("scale", p.get("lambda", 1)), 1))
        return stats.expon(scale=sc), f"Exp(λ={1/sc:.4g})"
    if nm in ("uniform", "均匀", "u"):
        lo = _num(p.get("lo", p.get("low", 0)), 0)
        hi = max(lo, _num(p.get("hi", p.get("high", 1)), 1))
        return stats.uniform(lo, hi - lo), f"U({lo:g}, {hi:g})"
    if nm in ("chi2", "chi", "卡方"):
        df = max(1e-9, _num(p.get("df", p.get("nu", 10)), 10))
        return stats.chi2(df), f"χ²({df:g})"
    if nm in ("t", "student", "学生t"):
        df = max(1e-9, _num(p.get("df", p.get("nu", 10)), 10))
        loc = _num(p.get("loc", 0), 0)
        sc = max(1e-9, _num(p.get("scale", 1), 1))
        return stats.t(df, loc=loc, scale=sc), f"t({df:g})"
    if nm in ("f", "fisher", "f分布"):
        dfn = max(1e-9, _num(p.get("dfn", p.get("nu1", 10)), 10))
        dfd = max(1e-9, _num(p.get("dfd", p.get("nu2", 20)), 20))
        return stats.f(dfn, dfd), f"F({dfn:g}, {dfd:g})"
    if nm in ("beta", "β", "beta分布"):
        a = max(1e-9, _num(p.get("a", 2), 2))
        b = max(1e-9, _num(p.get("b", 2), 2))
        return stats.beta(a, b), f"Beta({a:g}, {b:g})"
    if nm in ("gamma", "伽马", "gamma分布"):
        k = max(1e-9, _num(p.get("k", p.get("shape", 2)), 2))
        sc = max(1e-9, _num(p.get("scale", 1), 1))
        return stats.gamma(k, scale=sc), f"Gamma(k={k:g}, θ={sc:g})"
    raise ValueError("不支持的分布。可选：normal（正态）/binom（二项）/poisson（泊松）/expon（指数）/uniform（均匀）/chi2（卡方）/t（t）/f（F）/beta（β）/gamma（伽马）。")


def _fig_to_pil(fig, dpi=100):
    """matplotlib Figure → PIL.Image（RGBA），自动关闭 figure。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert("RGBA").copy()


def _text_of(obj):
    """sympy/数值结果 → 可读文本（精确式 + 数值）。"""
    try:
        exact = sympify(obj)
        val = complex(exact.evalf())
        if abs(val.imag) < 1e-12:
            return f"{latex(exact)} ≈ {val.real:.6g}"
        return f"{latex(exact)} ≈ {val:.6g}"
    except Exception:
        return str(obj)


# ============================================================
# 公式计算（等价 calculator_page 的 sympy 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "数学表达式，如 x**2 + 2*x + 1 或 x^2+2x+1"},
        "var": {"type": "string", "description": "自变量，默认 x"},
        "values": {"type": "string", "description": "赋值串，如 x=2, y=3；或留空求符号结果"},
    },
    "required": ["expr"],
}, category="公式计算")
def eval_expr(expr, var="x", values=""):
    """对数学表达式求值。给数值则算数值，不给则化简符号表达式。"""
    e = _sym(expr)
    subs = {}
    if values:
        for part in str(values).replace("，", ",").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                subs[symbols(str(k).strip())] = _num(v)
    exact = e.subs(subs) if subs else e
    val = complex(exact.evalf())
    if abs(val.imag) < 1e-12:
        txt = f"{latex(e)} = {latex(exact)}"
        if subs:
            txt += f"\n数值结果 ≈ {val.real:.8g}"
        return {"text": txt}
    return {"text": f"{latex(e)} = {latex(exact)}\n数值结果 ≈ {val:.8g}"}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "要求导的表达式，如 sin(x)**2 或 sin(x)^2"},
        "var": {"type": "string", "description": "对哪个变量求导，默认 x"},
        "order": {"type": "integer", "description": "求导阶数，默认 1"},
    },
    "required": ["expr"],
}, category="公式计算")
def diff_expr(expr, var="x", order=1):
    """对表达式求导（支持高阶导数），返回导数公式。"""
    e = _sym(expr)
    d = diff(e, symbols(str(var)), max(1, int(order)))
    return {"text": f"d/d{var}  {latex(e)}  =  {latex(d)}\n{pretty(d)}"}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "要积分的表达式，如 exp(-x**2) 或 x^2*exp(x)"},
        "var": {"type": "string", "description": "积分变量，默认 x"},
        "low": {"type": "string", "description": "定积分下限，可写 -oo / pi / 2 等；留空为不定积分"},
        "high": {"type": "string", "description": "定积分上限，可写 oo / pi 等；留空为不定积分"},
    },
    "required": ["expr"],
}, category="公式计算")
def integrate_expr(expr, var="x", low="", high=""):
    """对表达式积分。填了上下限就是定积分，否则不定积分（记得加常数 C）。"""
    x = symbols(str(var))
    e = _sym(expr)
    if low != "" or high != "":
        lo = sympify(str(low).replace("oo", "oo") or "-oo")
        hi = sympify(str(high).replace("oo", "oo") or "oo")
        r = integrate(e, (x, lo, hi))
        return {"text": f"∫{latex(e)} dx 从 {lo} 到 {hi}\n= {latex(r)}\n≈ {complex(r.evalf()):.8g}"}
    r = integrate(e, x)
    return {"text": f"∫{latex(e)} dx = {latex(r)} + C\n{pretty(r)} + C"}


@_reg
@_tool({
    "properties": {
        "equation": {"type": "string", "description": "方程或方程列表，如 'x**2 - 4 = 0' 或 'x + y = 3'，多个方程用分号隔开"},
        "unknowns": {"type": "string", "description": "未知数列表，如 'x' 或 'x,y'；默认自动识别"},
    },
    "required": ["equation"],
}, category="公式计算")
def solve_eq(equation, unknowns=""):
    """解方程或方程组。单个方程返回所有解，多元方程自动识别未知数。"""
    parts = [p.strip() for p in str(equation).split(";") if p.strip()]
    eqs = []
    for p in parts:
        p = p.replace("^", "**").replace(" = ", "=")
        if "=" in p:
            lhs, rhs = p.split("=", 1)
            eqs.append(Eq(_sym(lhs), _sym(rhs)))
        else:
            eqs.append(Eq(_sym(p), 0))
    if unknowns:
        syms = [symbols(s.strip()) for s in str(unknowns).split(",") if s.strip()]
    else:
        syms = sorted({s for e in eqs for s in e.free_symbols}, key=str)
    if len(eqs) == 1:
        sol = solve(eqs[0], syms[0] if syms else symbols("x"))
        return {"text": f"解：{latex(eqs[0])} 的解为\n{latex(sol)}\n{pretty(sol)}"}
    sol = solve(eqs, syms)
    return {"text": f"方程组解：\n{latex(sol)}\n{pretty(sol)}"}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "表达式，如 sin(x)/x"},
        "var": {"type": "string", "description": "自变量，默认 x"},
        "at": {"type": "string", "description": "趋近点，默认 0；可写 oo（无穷大）"},
        "direction": {"type": "string", "description": "方向：留空=双侧，'+'=右极限，'-'=左极限"},
    },
    "required": ["expr"],
}, category="公式计算")
def limit_expr(expr, var="x", at="0", direction=""):
    """求极限。可指定趋近点与左右方向。"""
    x = symbols(str(var))
    e = _sym(expr)
    pt = oo if str(at).strip() in ("oo", "inf", "无穷") else _sym(at)
    if str(direction).strip() == "+":
        r = limit(e, x, pt, dir="+")
    elif str(direction).strip() == "-":
        r = limit(e, x, pt, dir="-")
    else:
        r = limit(e, x, pt)
    return {"text": f"lim_{{{var}→{latex(pt)}}} {latex(e)} = {latex(r)}\n{_text_of(r)}"}


# ============================================================
# 矩阵与线性代数（等价 matrix_page 的 sympy 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "嵌套列表表示矩阵，如 [[1,2],[3,4]]"},
    },
    "required": ["matrix"],
}, category="矩阵与线性代数")
def matrix_det(matrix):
    """求方阵行列式。"""
    m = _mat(matrix)
    return {"text": f"det(A) = {latex(m.det())}\n≈ {complex(m.det().evalf()):.6g}"}


@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "嵌套列表表示矩阵，如 [[1,2],[3,4]]"},
    },
    "required": ["matrix"],
}, category="矩阵与线性代数")
def matrix_inv(matrix):
    """求方阵的逆矩阵。"""
    m = _mat(matrix)
    inv = m.inv()
    return {"text": f"A⁻¹ = {latex(inv)}\n{pretty(inv)}"}


@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "嵌套列表表示矩阵"},
    },
    "required": ["matrix"],
}, category="矩阵与线性代数")
def matrix_trans(matrix):
    """求矩阵的转置。"""
    m = _mat(matrix)
    return {"text": f"Aᵀ = {latex(m.T)}\n{pretty(m.T)}"}


@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "嵌套列表表示矩阵"},
    },
    "required": ["matrix"],
}, category="矩阵与线性代数")
def matrix_rank(matrix):
    """求矩阵的秩。"""
    m = _mat(matrix)
    return {"text": f"矩阵 {m.shape[0]}×{m.shape[1]}，秩 rank(A) = {m.rank()}"}


@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "嵌套列表表示方阵"},
    },
    "required": ["matrix"],
}, category="矩阵与线性代数")
def matrix_eigen(matrix):
    """求方阵的特征值与特征向量。"""
    m = _mat(matrix)
    vals = m.eigenvals()
    txt = f"特征值（含重数）：{latex(vals)}"
    try:
        vects = m.eigenvects()
        if vects:
            txt += "\n特征向量：\n"
            for val, mult, vecs in vects:
                for v in vecs:
                    txt += f"  λ={latex(val)}：{latex(v)}\n"
    except Exception:
        pass
    return {"text": txt}


@_reg
@_tool({
    "properties": {
        "matrix": {"type": "array", "description": "系数矩阵 A，嵌套列表"},
        "rhs": {"type": "array", "description": "常数项列向量 b，如 [1,2]"},
        "unknowns": {"type": "string", "description": "未知数名，逗号分隔，如 x1,x2；留空自动"},
    },
    "required": ["matrix", "rhs"],
}, category="矩阵与线性代数")
def matrix_solve(matrix, rhs, unknowns=""):
    """解线性方程组 A·x = b。返回解的列表。"""
    m = _mat(matrix)
    b = Matrix([_sym(str(c).replace("^", "**")) for c in (rhs if isinstance(rhs, (list, tuple)) else _nums(rhs))])
    sol = m.LUsolve(b)
    if unknowns:
        names = [symbols(s.strip()) for s in str(unknowns).split(",") if s.strip()]
        pairs = "，".join(f"{n} = {latex(sol[i])}" for i, n in enumerate(names))
        return {"text": f"解：{pairs}\n{pretty(sol)}"}
    return {"text": f"解向量：{latex(sol)}\n{pretty(sol)}"}


@_reg
@_tool({
    "properties": {
        "a": {"type": "array", "description": "左矩阵"},
        "b": {"type": "array", "description": "右矩阵"},
    },
    "required": ["a", "b"],
}, category="矩阵与线性代数")
def matrix_mul(a, b):
    """矩阵乘法 A·B。"""
    ma, mb = _mat(a), _mat(b)
    r = ma * mb
    return {"text": f"A·B = {latex(r)}\n{pretty(r)}"}


# ============================================================
# 统计检验（复用根目录 stats_utils 的 9 个函数）
# ============================================================

@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行列名）或 {列名:[值]} 字典或嵌套列表"},
        "columns": {"description": "要统计的数值列名列表；留空则自动取全部数值列"},
    },
    "required": ["data"],
}, category="统计分析")
def desc_stats(data, columns=None):
    """描述统计：对指定数值列输出 n/均值/标准差/分位数/偏度/峰度/变异系数。"""
    from stats_utils import descriptive_stats_table
    df = _to_df(data)
    if columns:
        cols = [str(c) for c in (columns if isinstance(columns, (list, tuple)) else [columns])]
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("没有可统计的数值列。")
    t = descriptive_stats_table(df, cols)
    return {"text": t.round(4).to_string()}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "columns": {"description": "要检验的数值列名列表；留空则自动取全部数值列"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data"],
}, category="统计分析")
def normality_test(data, columns=None, alpha=0.05):
    """正态性检验：对数值列做 Shapiro-Wilk / D'Agostino 检验，判断是否服从正态。"""
    from stats_utils import normality_table
    df = _to_df(data)
    if columns:
        cols = [str(c) for c in (columns if isinstance(columns, (list, tuple)) else [columns])]
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("没有可检验的数值列。")
    t = normality_table(df, cols, alpha=_num(alpha, 0.05))
    return {"text": t.to_string()}


def _two_col_data(data, value_col, group_col, index_col=None):
    """把 data + 两列名转成 (df, value_col, group_col)。支持把列数据字典直接给列。"""
    df = _to_df(data)
    if index_col:
        df = df.set_index(str(index_col))
    return df, str(value_col), str(group_col)


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "因变量列名"},
        "group_col": {"type": "string", "description": "分组变量列名（必须恰好 2 个水平）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "value_col", "group_col"],
}, category="统计分析")
def t_test(data, value_col, group_col, alpha=0.05):
    """独立样本 t 检验：比较两组均值（含 Levene 方差齐性、Cohen's d、Mann-Whitney）。"""
    from stats_utils import independent_t_test
    df, vc, gc = _two_col_data(data, value_col, group_col)
    r = independent_t_test(df, vc, gc, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "因变量列名"},
        "group_col": {"type": "string", "description": "分组变量列名（2 个以上水平做 ANOVA）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "value_col", "group_col"],
}, category="统计分析")
def anova(data, value_col, group_col, alpha=0.05):
    """单因素方差分析：比较多组均值是否相等（含 eta²）。"""
    from stats_utils import anova_one_way
    df, vc, gc = _two_col_data(data, value_col, group_col)
    r = anova_one_way(df, vc, gc, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "col1": {"type": "string", "description": "第一个分类列名"},
        "col2": {"type": "string", "description": "第二个分类列名"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "col1", "col2"],
}, category="统计分析")
def chi_square(data, col1, col2, alpha=0.05):
    """卡方独立性检验：判断两个分类变量是否有关联（含 Cramér's V）。"""
    from stats_utils import chi_square_test
    df, c1, c2 = _two_col_data(data, col1, col2)
    r = chi_square_test(df, c1, c2, alpha=_num(alpha, 0.05))
    txt = _fmt_dict({k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)})
    if isinstance(r.get("列联表"), pd.DataFrame):
        txt += "\n\n列联表：\n" + r["列联表"].to_string()
    return {"text": txt}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "x_col": {"type": "string", "description": "X 变量列名"},
        "y_col": {"type": "string", "description": "Y 变量列名"},
        "method": {"type": "string", "description": "方法：Pearson（线性相关）或 Spearman（秩相关）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "x_col", "y_col"],
}, category="统计分析")
def correlation(data, x_col, y_col, method="Pearson", alpha=0.05):
    """相关性检验：判断两个数值变量的线性/秩相关关系与强度。"""
    from stats_utils import correlation_test
    df, xc, yc = _two_col_data(data, x_col, y_col)
    r = correlation_test(df, xc, yc, method=str(method or "Pearson"), alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "x_col": {"type": "string", "description": "自变量列名"},
        "y_col": {"type": "string", "description": "因变量列名"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "x_col", "y_col"],
}, category="统计分析")
def regression(data, x_col, y_col, alpha=0.05):
    """简单线性回归：输出回归方程、R²、F、系数检验、95%CI、Pearson r。"""
    from stats_utils import simple_linear_regression
    df, xc, yc = _two_col_data(data, x_col, y_col)
    r = simple_linear_regression(df, xc, yc, alpha=_num(alpha, 0.05))
    txt = (_fmt_dict({k: v for k, v in r.items() if k != "残差"})
           + f"\n残差前 6 个：{[f'{v:.4g}' for v in r['残差'][:6]]}")
    return {"text": txt}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "before_col": {"type": "string", "description": "前测列名"},
        "after_col": {"type": "string", "description": "后测列名"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "before_col", "after_col"],
}, category="统计分析")
def paired_t_test(data, before_col, after_col, alpha=0.05):
    """配对 t 检验：比较同一批对象前测/后测差异。"""
    from stats_utils import paired_t_test as _ptt
    df, bc, ac = _two_col_data(data, before_col, after_col)
    r = _ptt(df, bc, ac, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


# ============================================================
# v1.4.0 新增：统计检验工具（11 个）
# ============================================================

def _col_list(v):
    """把列名参数转成字符串列表（接受列表/元组或空格逗号分隔字符串）。"""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [x for x in str(v).replace(",", " ").replace("，", " ").split() if x]


def _fmt_with_tables(r, table_keys=()):
    """排版统计 dict，并把 DataFrame/矩阵键以表格形式追加在末尾。"""
    txt = _fmt_dict(r)
    for k in table_keys:
        v = r.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            txt += f"\n\n{k}：\n" + v.to_string()
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.size:
            txt += f"\n\n{k}：\n" + np.array2string(v)
    return txt


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "因变量列名"},
        "group_col": {"type": "string", "description": "分组变量列名（必须恰好 2 个水平）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "value_col", "group_col"],
}, category="统计分析")
def mannwhitney_u(data, value_col, group_col, alpha=0.05):
    """Mann-Whitney U 检验：两独立样本分布差异（t 检验的非参替代，含秩双列相关效应量）。"""
    from stats_utils import mannwhitney_u_test
    df, vc, gc = _two_col_data(data, value_col, group_col)
    r = mannwhitney_u_test(df, vc, gc, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "before_col": {"type": "string", "description": "前测列名"},
        "after_col": {"type": "string", "description": "后测列名"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "before_col", "after_col"],
}, category="统计分析")
def wilcoxon_signed_rank(data, before_col, after_col, alpha=0.05):
    """Wilcoxon 符号秩检验：配对样本中位数差异（配对 t 的非参替代，含效应量 r）。"""
    from stats_utils import wilcoxon_signed_rank as _wsr
    df, bc, ac = _two_col_data(data, before_col, after_col)
    r = _wsr(df, bc, ac, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "因变量列名"},
        "group_col": {"type": "string", "description": "分组变量列名（2 个以上水平）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "value_col", "group_col"],
}, category="统计分析")
def kruskal_wallis(data, value_col, group_col, alpha=0.05):
    """Kruskal-Wallis 检验：多组独立样本分布差异（ANOVA 的非参替代，含 ε² 效应量）。"""
    from stats_utils import kruskal_wallis_test
    df, vc, gc = _two_col_data(data, value_col, group_col)
    r = kruskal_wallis_test(df, vc, gc, alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "columns": {"type": "array", "items": {"type": "string"}, "description": "测量条件列名数组（≥2）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "columns"],
}, category="统计分析")
def friedman(data, columns, alpha=0.05):
    """Friedman 检验：同一批对象多个条件下分布差异（重复测量 ANOVA 的非参替代，含 Kendall's W）。"""
    from stats_utils import friedman_test
    df = _to_df(data)
    r = friedman_test(df, _col_list(columns), alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "因变量列名"},
        "group_col": {"type": "string", "description": "分组变量列名（至少 3 个水平）"},
        "method": {"type": "string", "description": "方法：Tukey（默认）/ LSD / Bonferroni"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "value_col", "group_col"],
}, category="统计分析")
def posthoc(data, value_col, group_col, method="Tukey", alpha=0.05):
    """事后多重比较：ANOVA/Kruskal 显著后确定哪些组不同（Tukey HSD / LSD / Bonferroni）。"""
    from stats_utils import posthoc_test
    df, vc, gc = _two_col_data(data, value_col, group_col)
    r = posthoc_test(df, vc, gc, method=str(method or "Tukey"), alpha=_num(alpha, 0.05))
    return {"text": _fmt_with_tables(r, ("比较结果表",))}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "col": {"type": "string", "description": "数值列名"},
        "mu": {"type": "number", "description": "检验值 μ0，默认 0"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "col"],
}, category="统计分析")
def one_sample_t(data, col, mu=0.0, alpha=0.05):
    """单样本 t 检验：检验变量均值是否等于已知值 μ0（含 95%CI 与 Cohen's d）。"""
    from stats_utils import one_sample_t_test
    df = _to_df(data)
    r = one_sample_t_test(df, str(col), mu=_num(mu, 0.0), alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "col": {"type": "string", "description": "二分类列名"},
        "success_value": {"type": "string", "description": "成功水平（留空自动：数值列取 1，文本取第 2 个水平）"},
        "p0": {"type": "number", "description": "检验比例，默认 0.5"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "col"],
}, category="统计分析")
def proportion_test(data, col, success_value="", p0=0.5, alpha=0.05):
    """二项比例检验：检验二分类列中某一水平比例是否等于 p0（精确 Binomial + 近似 z）。"""
    from stats_utils import proportion_test as _pt
    df = _to_df(data)
    sv = str(success_value or "")
    if sv != "":
        try:
            sv = float(sv)
        except ValueError:
            pass
    r = _pt(df, str(col), success_value=sv, p0=_num(p0, 0.5), alpha=_num(alpha, 0.05))
    return {"text": _fmt_dict(r)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "col1": {"type": "string", "description": "第一个二分类列名"},
        "col2": {"type": "string", "description": "第二个二分类列名"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "col1", "col2"],
}, category="统计分析")
def mcnemar(data, col1, col2, alpha=0.05):
    """McNemar 检验：两个二分类变量的配对卡方检验（含列联表与精确 p 值）。"""
    from stats_utils import mcnemar_test
    df, c1, c2 = _two_col_data(data, col1, col2)
    r = mcnemar_test(df, c1, c2, alpha=_num(alpha, 0.05))
    return {"text": _fmt_with_tables(r, ("列联表",))}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "y_col": {"type": "string", "description": "二分类因变量列名"},
        "x_cols": {"type": "array", "items": {"type": "string"}, "description": "数值自变量列名数组（≥1）"},
        "success_value": {"type": "string", "description": "成功水平（留空自动编码）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "y_col", "x_cols"],
}, category="统计分析")
def logistic_regression(data, y_col, x_cols, success_value="", alpha=0.05):
    """二分类逻辑回归：输出系数/OR/95%CI、McFadden R²、AUC 与准确率。"""
    from stats_utils import logistic_regression as _lr
    df = _to_df(data)
    sv = str(success_value or "")
    if sv != "":
        try:
            sv = float(sv)
        except ValueError:
            pass
    r = _lr(df, str(y_col), _col_list(x_cols), alpha=_num(alpha, 0.05), success_value=sv)
    return {"text": _fmt_with_tables(r, ("系数表",))}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "y_col": {"type": "string", "description": "连续因变量列名"},
        "x_cols": {"type": "array", "items": {"type": "string"}, "description": "数值自变量列名数组（≥2）"},
        "alpha": {"type": "number", "description": "显著性水平，默认 0.05"},
    },
    "required": ["data", "y_col", "x_cols"],
}, category="统计分析")
def multiple_regression(data, y_col, x_cols, alpha=0.05):
    """多元线性回归：输出系数表、VIF 共线性、Durbin-Watson、R²/F/p。"""
    from stats_utils import multiple_linear_regression
    df = _to_df(data)
    r = multiple_linear_regression(df, str(y_col), _col_list(x_cols), alpha=_num(alpha, 0.05))
    return {"text": _fmt_with_tables(r, ("系数表", "VIF 表"))}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "columns": {"type": "array", "items": {"type": "string"}, "description": "题项列名数组（≥2，如 q1~q8）"},
    },
    "required": ["data", "columns"],
}, category="统计分析")
def cronbach(data, columns):
    """信度分析 Cronbach's α：衡量量表题项的内部一致性，附各题项删除后 α。"""
    from stats_utils import cronbach_alpha
    df = _to_df(data)
    r = cronbach_alpha(df, _col_list(columns))
    return {"text": _fmt_with_tables(r, ("删除题项后 α",))}


def _fmt_dict(d):
    """把统计结果 dict 排版成可读文本（跳过数组/DataFrame）。"""
    lines = []
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, pd.DataFrame):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"{k}：{v:.6g}")
        else:
            lines.append(f"{k}：{v}")
    return "\n".join(lines)


# ============================================================
# 概率与分布（等价 prob_page 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "distribution": {"type": "string", "description": "分布名：normal/binom/poisson/expon/uniform（或中文）"},
        "params": {"type": "object", "description": "分布参数。normal:{mu,sigma}；binom:{n,p}；poisson:{lambda}；expon:{scale}；uniform:{low,high}"},
        "x1": {"type": "number", "description": "区间下界（可留空表示 -∞）"},
        "x2": {"type": "number", "description": "区间上界（可留空表示 +∞）"},
    },
    "required": ["distribution"],
}, category="概率与分布")
def dist_prob(distribution, params=None, x1=None, x2=None):
    """计算概率：P(x1 ≤ X ≤ x2)。只填 x1 表示 P(X ≥ x1)，只填 x2 表示 P(X ≤ x2)。"""
    d, label = _dist(distribution, params or {})
    if x1 is None and x2 is None:
        raise ValueError("请至少填 x1（下界）或 x2（上界）之一。")
    if x1 is None:
        p = d.cdf(_num(x2))
        return {"text": f"X ~ {label}，P(X ≤ {_num(x2):g}) = {p:.6f}（{p:.2%}）"}
    if x2 is None:
        p = 1 - d.cdf(_num(x1))
        return {"text": f"X ~ {label}，P(X ≥ {_num(x1):g}) = {p:.6f}（{p:.2%}）"}
    p = d.cdf(_num(x2)) - d.cdf(_num(x1))
    return {"text": f"X ~ {label}，P({_num(x1):g} ≤ X ≤ {_num(x2):g}) = {p:.6f}（{p:.2%}）"}


@_reg
@_tool({
    "properties": {
        "distribution": {"type": "string", "description": "分布名：normal/binom/poisson/expon/uniform"},
        "params": {"type": "object", "description": "分布参数（同上）"},
        "q": {"type": "number", "description": "概率水平（0~1），默认 0.5 即中位数"},
    },
    "required": ["distribution"],
}, category="概率与分布")
def dist_quantile(distribution, params=None, q=0.5):
    """求分位数：使 P(X ≤ x) = q 的 x 值。q=0.5 是中位数，q=0.975 是 97.5% 分位点。"""
    d, label = _dist(distribution, params or {})
    qq = min(0.9999, max(0.0001, _num(q, 0.5)))
    x = d.ppf(qq)
    return {"text": f"X ~ {label}，{qq:.4f} 分位数 = {x:.6g}"}


@_reg
@_tool({
    "properties": {
        "distribution": {"type": "string", "description": "分布名：normal/binom/poisson/expon/uniform"},
        "params": {"type": "object", "description": "分布参数（同上）"},
        "n": {"type": "integer", "description": "样本数量，默认 20"},
        "seed": {"type": "integer", "description": "随机种子（留空随机）"},
    },
    "required": ["distribution"],
}, category="概率与分布")
def dist_random(distribution, params=None, n=20, seed=None):
    """按指定分布生成随机样本，返回数值列表。"""
    d, label = _dist(distribution, params or {})
    cnt = max(1, int(_num(n, 20)))
    rng = np.random.default_rng(int(_num(seed, 0)) if seed is not None else None)
    vals = d.rvs(size=cnt, random_state=rng)
    return {"text": f"X ~ {label}，随机样本 {cnt} 个：\n" + ", ".join(f"{v:.6g}" for v in vals)}


# ============================================================
# 公式渲染（latex_view 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "表达式字符串，如 x**2 + 1 或 1/(x+1)"},
        "kind": {"type": "string", "description": "渲染类型：auto（自动判断公式/矩阵）/matrix（按矩阵渲染）"},
    },
    "required": ["expr"],
}, image=True, category="公式渲染")
def render_latex(expr, kind="auto"):
    """把数学表达式/矩阵渲染成专业公式图片返回。"""
    from modules.latex_view import sympy_to_image
    raw = str(expr)
    if kind == "matrix" or (kind != "formula" and "[" in raw and "[" in raw[raw.find("["):]):
        try:
            mat = _mat(ast.literal_eval(raw)) if "[" in raw else None
        except Exception:
            mat = None
        if mat is not None:
            img = sympy_to_image(mat)
            return {"text": f"矩阵已渲染：{mat.shape[0]}×{mat.shape[1]}", "image": img}
    e = _sym(raw)
    img = sympy_to_image(e)
    return {"text": f"公式已渲染：{latex(e)}", "image": img}


# ============================================================
# 绘图可视化（复用 plot_page/stats_plots 的绘图逻辑，Agg 后端）
# ============================================================

def _lam(expr_str, vars_=("x",)):
    """字符串 → (sympy 表达式, 向量化 numpy 函数)。"""
    e = _sym(expr_str)
    syms = [symbols(v) for v in vars_]
    return e, lambdify(syms, e, "numpy")


def _range(v, default_lo=-10.0, default_hi=10.0):
    """解析范围：接受 [a,b] / "a,b" / 数字对 / 空（用默认）。"""
    lo, hi = default_lo, default_hi
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        lo, hi = _num(v[0], lo), _num(v[1], hi)
    elif isinstance(v, str) and v.strip():
        parts = v.replace("，", ",").split(",")
        if len(parts) >= 2:
            lo, hi = _num(parts[0], lo), _num(parts[1], hi)
    return min(lo, hi), max(lo, hi)


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "一元函数表达式，如 sin(x) 或 x**2"},
        "xmin": {"type": "number", "description": "x 范围下限，默认 -10"},
        "xmax": {"type": "number", "description": "x 范围上限，默认 10"},
        "title": {"type": "string", "description": "图标题，留空自动"},
    },
    "required": ["expr"],
}, image=True, category="绘图可视化")
def plot_curve(expr, xmin=-10.0, xmax=10.0, title=""):
    """绘制一元函数曲线 y=f(x)。"""
    e, f = _lam(expr)
    xs = np.linspace(_num(xmin, -10), _num(xmax, 10), 600)
    with np.errstate(all="ignore"):
        ys = f(xs)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ys, lw=2, color="steelblue")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.grid(True)
    ax.set_title(title or f"y = {latex(e)}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return {"text": f"已绘制 y = {latex(e)}（{_num(xmin, -10):g} ≤ x ≤ {_num(xmax, 10):g}）",
            "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "二元函数表达式，如 sin(x)*cos(y) 或 x**2 + y**2"},
        "xmin": {"type": "number", "description": "x 下限，默认 -5"},
        "xmax": {"type": "number", "description": "x 上限，默认 5"},
        "ymin": {"type": "number", "description": "y 下限，默认 -5"},
        "ymax": {"type": "number", "description": "y 上限，默认 5"},
    },
    "required": ["expr"],
}, image=True, category="绘图可视化")
def plot_3d(expr, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    """绘制二元函数的 3D 曲面图 z=f(x,y)。"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    e, f = _lam(expr, ("x", "y"))
    x = np.linspace(_num(xmin, -5), _num(xmax, 5), 60)
    y = np.linspace(_num(ymin, -5), _num(ymax, 5), 60)
    X, Y = np.meshgrid(x, y)
    with np.errstate(all="ignore"):
        Z = np.array(f(X, Y), dtype=float)
    Z = np.where(np.isfinite(Z), Z, np.nan)
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.92)
    ax.set_title(f"z = {latex(e)}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return {"text": f"已绘制 3D 曲面 z = {latex(e)}", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "二元函数表达式，如 sin(x)*cos(y)"},
        "xmin": {"type": "number", "description": "x 下限，默认 -5"},
        "xmax": {"type": "number", "description": "x 上限，默认 5"},
        "ymin": {"type": "number", "description": "y 下限，默认 -5"},
        "ymax": {"type": "number", "description": "y 上限，默认 5"},
    },
    "required": ["expr"],
}, image=True, category="绘图可视化")
def plot_contour(expr, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    """绘制二元函数的等高线图。"""
    e, f = _lam(expr, ("x", "y"))
    x = np.linspace(_num(xmin, -5), _num(xmax, 5), 200)
    y = np.linspace(_num(ymin, -5), _num(ymax, 5), 200)
    X, Y = np.meshgrid(x, y)
    with np.errstate(all="ignore"):
        Z = np.array(f(X, Y), dtype=float)
    Z = np.where(np.isfinite(Z), Z, np.nan)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    cs = ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.6)
    ax.clabel(cs, inline=1, fontsize=8, fmt="%.2g")
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"等高线 z = {latex(e)}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return {"text": f"已绘制等高线 z = {latex(e)}", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行列名）或 {列名:[值]} 字典"},
        "x_col": {"type": "string", "description": "X 列名（留空用行号）"},
        "y_cols": {"description": "Y 列名（一个或多个），如 'y' 或 ['a','b']"},
        "mode": {"type": "string", "description": "模式：line（折线，默认）或 scatter（散点）"},
        "title": {"type": "string", "description": "图标题，留空自动"},
    },
    "required": ["data", "y_cols"],
}, image=True, category="绘图可视化")
def plot_line(data, y_cols, x_col="", mode="line", title=""):
    """绘制折线图或散点图：一列或多列 Y 对照 X。"""
    df = _to_df(data)
    if x_col:
        xs = df[str(x_col)].astype(float)
    else:
        xs = np.arange(len(df))
    cols = [str(c) for c in (y_cols if isinstance(y_cols, (list, tuple)) else [y_cols])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c in cols:
        ys = pd.to_numeric(df[c], errors="coerce")
        if mode == "scatter":
            ax.scatter(xs, ys, s=26, label=c)
        else:
            ax.plot(xs, ys, lw=1.8, marker="o", ms=3, label=c)
    ax.grid(True)
    ax.set_xlabel(str(x_col) if x_col else "行号")
    ax.legend(fontsize=9)
    ax.set_title(title or "、".join(cols))
    return {"text": f"已绘制 {mode}图：{', '.join(cols)}", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典 或 数值列表"},
        "columns": {"description": "要画直方图的列名；列表型数据可省略"},
        "bins": {"type": "integer", "description": "分箱数，默认 15"},
        "title": {"type": "string", "description": "图标题，留空自动"},
    },
    "required": ["data"],
}, image=True, category="绘图可视化")
def plot_hist(data, columns=None, bins=15, title=""):
    """绘制数据直方图（可叠加正态曲线）。"""
    if isinstance(data, (list, tuple)) and not (data and isinstance(data[0], (list, tuple))):
        series = pd.Series(_nums(data))
        cols = ["数据"]
        df = pd.DataFrame({cols[0]: series})
    else:
        df = _to_df(data)
        cols = [str(c) for c in (columns if isinstance(columns, (list, tuple)) else [columns or list(df.columns)[0]])]
    if not cols:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    nb = max(4, int(_num(bins, 15)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c in cols[:3]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        ax.hist(s, bins=nb, alpha=0.55, label=c)
        mu, sg = s.mean(), s.std(ddof=1)
        if sg > 0:
            xs = np.linspace(s.min(), s.max(), 200)
            # 用真实箱宽缩放正态曲线，使其面积=样本数，与直方图柱高一致
            bin_w = (s.max() - s.min()) / nb
            from scipy.stats import norm
            ax.plot(xs, len(s) * bin_w * norm.pdf(xs, mu, sg), lw=1.8, label=f"{c} 正态拟合")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("取值"); ax.set_ylabel("频数")
    ax.legend(fontsize=9)
    ax.set_title(title or "数据直方图")
    return {"text": f"直方图：{', '.join(cols)}（{nb} 箱）", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行为列名）或 {列名:[值]} 字典"},
        "x_col": {"type": "string", "description": "X 列名"},
        "y_col": {"type": "string", "description": "Y 列名"},
        "title": {"type": "string", "description": "图标题，留空自动"},
    },
    "required": ["data", "x_col", "y_col"],
}, image=True, category="绘图可视化")
def plot_scatter_fit(data, x_col, y_col, title=""):
    """散点图 + 线性回归拟合线 + R²。"""
    from stats_utils import simple_linear_regression
    df, xc, yc = _two_col_data(data, x_col, y_col)
    r = simple_linear_regression(df, xc, yc)
    x = df[xc].astype(float); y = df[yc].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x, y, s=30, alpha=0.7, color="steelblue")
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, r["b0"] + r["b1"] * xs, "r-", lw=2, label=f"y = {r['b1']:.4g}x + {r['b0']:.4g}")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    ax.set_title(f"{title or '散点与回归'}　R² = {r['r2']:.4g}")
    ax.legend(fontsize=9)
    return {"text": f"回归方程：{r['equation']}\nR² = {r['r2']:.6g}，r = {r['pearson_r']:.6g}，p = {r['pearson_p']:.6g}",
            "image": _fig_to_pil(fig)}


# ============================================================
# v1.5.0 新增：出版级图型工具（小提琴 / 热图 / 误差棒 / 森林）
# ============================================================

def _pick_col(v, default=None):
    """模型可能传列表或单个字符串，统一取第一个作为列名。"""
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else default
    if v is None:
        return default
    return str(v)


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行列名）或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "数值列名"},
        "group_col": {"type": "string", "description": "分组列名"},
    },
    "required": ["data", "value_col", "group_col"],
}, image=True, category="绘图可视化")
def plot_violin(data, value_col, group_col):
    """小提琴图：按分组变量展示数值分布（含箱线中位数，比箱线图信息更全）。"""
    from stats_plots import violin_by_group
    df = _to_df(data)
    vc, gc = _pick_col(value_col), _pick_col(group_col)
    if not vc or not gc:
        raise ValueError("请提供 value_col（数值列）与 group_col（分组列）。")
    fig = violin_by_group(df, vc, gc)
    return {"text": f"已绘制 {vc} 按 {gc} 分组小提琴图。", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "columns": {"description": "数值列名列表，留空自动取全部数值列"},
    },
    "required": ["data"],
}, image=True, category="绘图可视化")
def plot_heatmap(data, columns=None):
    """相关矩阵热图：数值变量两两 Pearson 相关系数的热图，格内标注 r 值。"""
    from stats_plots import correlation_heatmap
    df = _to_df(data)
    cols = _col_list(columns) if columns else None
    fig = correlation_heatmap(df, cols)
    return {"text": "已绘制相关矩阵热图（Pearson r）。", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本或 {列名:[值]} 字典"},
        "value_col": {"type": "string", "description": "数值列名"},
        "group_col": {"type": "string", "description": "分组列名"},
        "error": {"type": "string", "description": "误差类型 SD 或 SE，默认 SD"},
    },
    "required": ["data", "value_col", "group_col"],
}, image=True, category="绘图可视化")
def plot_errorbar(data, value_col, group_col, error="sd"):
    """误差棒图：各组均值 ± SD/SE 的点误差棒图（实验科学论文标配）。"""
    from stats_plots import errorbar_by_group
    df = _to_df(data)
    vc, gc = _pick_col(value_col), _pick_col(group_col)
    if not vc or not gc:
        raise ValueError("请提供 value_col（数值列）与 group_col（分组列）。")
    fig = errorbar_by_group(df, vc, gc, error=str(error or "sd"))
    return {"text": f"已绘制 {vc} 按 {gc} 分组均值误差棒图。", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "estimates": {"description": "点估计列表，如 [0.5, -0.3, 1.2]（与 lower/upper 一一对应）"},
        "lower": {"description": "95% 置信下限列表"},
        "upper": {"description": "95% 置信上限列表"},
        "labels": {"description": "行标签列表，如 ['因子A','因子B']"},
        "table": {"description": "回归系数表（CSV/字典，含变量与95%CI列）；给了 table 可省略 estimates/lower/upper"},
        "title": {"type": "string", "description": "图标题，默认自动"},
    },
    "required": [],
}, image=True, category="绘图可视化")
def plot_forest(estimates=None, lower=None, upper=None, labels=None, table=None, title=""):
    """森林图：一组点估计与 95% 置信区间（回归系数 / OR / 效应量）。

    传入 estimates/lower/upper 直接作图；或传回归系数表 由表自动识别列名作图。
    """
    from stats_plots import forest_plot, regression_forest_plot
    if table is not None:
        coef = _to_df(table)
        fig = regression_forest_plot(coef, title=title or "回归系数森林图")
        return {"text": "已绘制回归系数森林图（95% CI）。", "image": _fig_to_pil(fig)}
    est = _nums(estimates) if estimates is not None else []
    lo = _nums(lower) if lower is not None else []
    hi = _nums(upper) if upper is not None else []
    if not est:
        raise ValueError("请提供 estimates/lower/upper，或提供回归系数表 table。")
    lab = _pick_list(labels, len(est))
    fig = forest_plot(est, lo, hi, labels=lab, title=title or "森林图")
    return {"text": f"已绘制森林图（{len(est)} 项估计）。", "image": _fig_to_pil(fig)}


def _pick_list(v, n):
    """把 labels 转成长度为 n 的字符串列表。"""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    parts = [x for x in str(v).replace("，", ",").split(",") if x.strip()]
    if len(parts) == 1 and n > 1:
        return [str(v)] * n
    return parts[:n]


# ============================================================
# 时间序列（等价 time_series_page 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "data": {"description": "时序数据：数值列表或空格/逗号分隔字符串"},
        "period": {"type": "integer", "description": "季节周期长度，默认 4（如 12=月度）"},
    },
    "required": ["data"],
}, image=True, category="时间序列")
def ts_decompose(data, period=4):
    """时序分解：把序列拆成 趋势 + 季节 + 残差 三张子图。"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    y = np.array(_nums(data), dtype=float)
    if len(y) < 8:
        raise ValueError("数据点太少（至少 8 个）。")
    per = max(2, int(_num(period, 4)))
    if len(y) < 2 * per:
        raise ValueError(f"数据太短：需要至少 2 个周期（{2 * per} 个点）。")
    res = seasonal_decompose(y, model="additive", period=per)
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 8.4), sharex=True)
    for ax, yy, ttl in ((axs[0], res.observed, "观测值"), (axs[1], res.trend, "趋势"),
                        (axs[2], res.seasonal, "季节"), (axs[3], res.resid, "残差")):
        ax.plot(yy, lw=1.5, color="steelblue" if ttl != "残差" else "crimson")
        ax.set_title(ttl); ax.grid(True)
    fig.tight_layout()
    return {"text": f"分解完成（周期={per}，共 {len(y)} 个观测点）。", "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "时序数据：数值列表或空格/逗号分隔字符串"},
    },
    "required": ["data"],
}, image=True, category="时间序列")
def ts_acf_pacf(data):
    """绘制自相关 ACF 与偏自相关 PACF 图，辅助判断 ARIMA 阶数。"""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    y = np.array(_nums(data), dtype=float)
    if len(y) < 8:
        raise ValueError("数据点太少（至少 8 个）。")
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.0))
    plot_acf(y, lags=min(20, len(y) // 2), ax=axs[0])
    axs[0].set_title("自相关 ACF"); axs[0].grid(True)
    plot_pacf(y, lags=min(20, len(y) // 2), ax=axs[1])
    axs[1].set_title("偏自相关 PACF"); axs[1].grid(True)
    fig.tight_layout()
    return {"text": "ACF 拖尾 + PACF 截尾 → 适合 AR；ACF 截尾 + PACF 拖尾 → 适合 MA。",
            "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "时序数据：数值列表或空格/逗号分隔字符串"},
        "order": {"type": "string", "description": "ARIMA 阶数 p d q，如 '1 1 1' 或 '2,0,1'"},
        "steps": {"type": "integer", "description": "预测期数，默认 6"},
    },
    "required": ["data"],
}, image=True, category="时间序列")
def ts_arima(data, order="1 1 1", steps=6):
    """ARIMA 建模与预测：拟合并外推未来若干期（含 95% 置信区间）。"""
    from statsmodels.tsa.arima.model import ARIMA
    y = np.array(_nums(data), dtype=float)
    if len(y) < 8:
        raise ValueError("数据点太少（至少 8 个）。")
    o = [int(float(v)) for v in str(order).replace("，", ",").replace(" ", ",").split(",") if v.strip()][:3]
    while len(o) < 3:
        o.append(0)
    o = tuple(o)
    st = max(1, int(_num(steps, 6)))
    fit = ARIMA(y, order=o).fit()
    fc = fit.get_forecast(st)
    mean = fc.predicted_mean
    ci = fc.conf_int()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n = len(y)
    ax.plot(range(n), y, lw=2, color="steelblue", label="观测值")
    ax.plot(range(n, n + st), mean, lw=2, color="crimson", marker="o", label="预测")
    ax.fill_between(range(n, n + st), ci[:, 0], ci[:, 1], color="crimson", alpha=0.2, label="95% 置信区间")
    ax.axvline(n - 0.5, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)
    ax.set_title(f"ARIMA{o} 拟合与预测（下 {st} 期）")
    ax.grid(True)
    pred = ", ".join(f"{v:.4g}" for v in mean)
    return {"text": f"ARIMA 阶数 p={o[0]} d={o[1]} q={o[2]}\nAIC={fit.aic:.3g}（越小越好）\n预测 {st} 期：{pred}",
            "image": _fig_to_pil(fig)}


# ============================================================
# 机器学习（等价 ml_page 能力）
# ============================================================

@_reg
@_tool({
    "properties": {
        "data": {"description": "样本数据：每行一个样本（空格或逗号分隔特征），文本或嵌套列表"},
        "k": {"type": "integer", "description": "聚类数 K，默认 2"},
    },
    "required": ["data"],
}, image=True, category="机器学习")
def ml_kmeans(data, k=2):
    """KMeans 聚类：把样本自动分成 K 簇，画散点+簇中心，输出各簇样本数。"""
    from sklearn.cluster import KMeans
    if isinstance(data, str):
        rows = [[float(x) for x in ln.replace(",", " ").split()]
                for ln in data.splitlines() if ln.strip()]
    else:
        rows = [[float(x) for x in r] for r in data]
    mat = np.array(rows)
    if mat.shape[0] < 6:
        raise ValueError("样本太少（至少 6 行）。")
    kk = max(2, int(_num(k, 2)))
    kk = min(kk, mat.shape[0])
    km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(mat)
    labels = km.labels_
    fig = plt.figure(figsize=(6.8, 5.2))
    if mat.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2], c=labels, cmap="viridis", s=36)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   km.cluster_centers_[:, 2], c="red", marker="x", s=110, label="簇中心")
        ax.legend(fontsize=9)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(mat[:, 0], mat[:, 1], c=labels, cmap="viridis", s=46)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c="red", marker="x", s=140, label="簇中心")
        ax.legend(fontsize=9)
    ax.set_title(f"KMeans 聚类（K={kk}）")
    ax.grid(True, alpha=0.4)
    counts = [int((labels == i).sum()) for i in range(kk)]
    return {"text": f"K={kk}，样本 {mat.shape[0]} 个，特征 {mat.shape[1]} 个。\n"
                    f"簇内平方和 inertia={km.inertia_:.4g}\n各簇样本数：{counts}",
            "image": _fig_to_pil(fig)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "训练数据：每行一个样本，最后一列是类别标签（如 '1.0 2.0 A'），文本或嵌套列表"},
        "max_depth": {"type": "integer", "description": "树最大深度，留空不限"},
        "test_ratio": {"type": "number", "description": "测试集比例，默认 0.2"},
    },
    "required": ["data"],
}, image=True, category="机器学习")
def ml_tree(data, max_depth=None, test_ratio=0.2):
    """决策树分类：训练分类器，画树结构+混淆矩阵，输出准确率与特征重要性。"""
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    feats, labels = [], []
    rows = data.splitlines() if isinstance(data, str) else data
    for line in rows:
        if isinstance(line, str):
            toks = line.replace(",", " ").split()
            if len(toks) < 2:
                continue
            feats.append([float(x) for x in toks[:-1]])
            labels.append(toks[-1])
        else:
            if len(line) < 2:
                continue
            feats.append([float(x) for x in line[:-1]])
            labels.append(str(line[-1]))
    if len(feats) < 6:
        raise ValueError("样本太少（至少 6 行）。")
    X = np.array(feats); y = np.array(labels)
    if len(set(y)) < 2:
        raise ValueError("标签列至少需要 2 个类别。")
    depth = None if max_depth in (None, "") else int(_num(max_depth, 0)) or None
    ratio = min(0.8, max(0.1, _num(test_ratio, 0.2)))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ratio, random_state=0)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_tree(clf, filled=True, rounded=True, fontsize=8,
              feature_names=[f"特征{i + 1}" for i in range(X.shape[1])], ax=axs[0])
    cm = confusion_matrix(yte, clf.predict(Xte))
    im = axs[1].imshow(cm, cmap="Blues")
    axs[1].set_title("混淆矩阵")
    axs[1].set_xlabel("预测类别"); axs[1].set_ylabel("真实类别")
    fig.colorbar(im, ax=axs[1])
    fig.tight_layout()
    return {"text": f"测试集准确率 acc={acc:.1%}（{Xte.shape[0]} 个样本）\n"
                    f"特征重要性：{[f'{v:.3g}' for v in clf.feature_importances_]}",
            "image": _fig_to_pil(fig)}


# ============================================================
# 数值计算工具
# ============================================================

@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "函数表达式（用 x 作自变量），如 x**3-2*x-5"},
        "xmin": {"type": "number", "description": "求根搜索下界，默认 -10"},
        "xmax": {"type": "number", "description": "求根搜索上界，默认 10"},
    },
    "required": ["expr"],
}, category="数值计算")
def numeric_solve(expr, xmin=-10.0, xmax=10.0):
    """数值求根：用二分法/牛顿法求 f(x)=0 的实根。适合多项式等难解析求解的方程。"""
    from scipy import optimize
    f = lambda x: float(_sym(expr).subs("x", float(x)).evalf())
    lo, hi = min(_num(xmin, -10), _num(xmax, 10)), max(_num(xmin, -10), _num(xmax, 10))
    if f(lo) * f(hi) > 0:
        # 端点同号，改用牛顿法从区间中点起迭代
        try:
            root = optimize.newton(f, (lo + hi) / 2, maxiter=100)
            return {"text": f"数值根 x ≈ {root:.10g}\n（f(x) = {float(_sym(expr).subs('x', float(root)).evalf()):.3g}）"}
        except Exception as e:
            raise ValueError(f"未找到根：{e}")
    root = optimize.brentq(f, lo, hi)
    return {"text": f"数值根 x ≈ {root:.10g}\n（f(x) = {float(_sym(expr).subs('x', float(root)).evalf()):.3g}）"}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（列 x,y）或 {x:[...],y:[...]} 字典"},
        "func": {"type": "string", "description": "拟合函数模板，含参数 a,b,...（用 x 作自变量），如 a*exp(b*x)"},
        "x_col": {"type": "string", "description": "自变量列名，默认 x"},
        "y_col": {"type": "string", "description": "因变量列名，默认 y"},
    },
    "required": ["data", "func"],
}, category="数值计算")
def curve_fit(data, func, x_col="x", y_col="y"):
    """非线性曲线拟合：用最小二乘法把 data 拟合到用户给的函数模板，返回参数与 R²。"""
    from scipy import optimize
    df = _to_df(data)
    x = pd.to_numeric(df[x_col], errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").dropna().to_numpy(float)
    # 提取模板里的参数名（a,b,c,...），x 为自变量
    import re as _re
    params = [s for s in _re.findall(r"\b([a-z])\b", func) if s != "x"]
    params = list(dict.fromkeys(params))  # 去重保序
    if not params:
        params = ["a", "b"]
    syms = symbols(",".join(params))
    p0 = [1.0] * len(params)
    try:
        popt, _ = optimize.curve_fit(lambda xv, *p: _vec_fit(func, params, p, xv), x, y, p0=p0)
    except Exception as e:
        raise ValueError(f"拟合失败，请检查函数模板：{e}")
    ypred = _vec_fit(func, params, popt, x)
    ss_res = float(((y - ypred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    pairs = "，".join(f"{p} ≈ {v:.6g}" for p, v in zip(params, popt))
    return {"text": f"拟合参数：{pairs}\nR² = {r2:.4f}\n拟合式：{func.replace(chr(42)*2, '^')}"}


def _vec_fit(func, params, p, xv):
    """把拟合函数模板转成可调用函数（参数值化）。"""
    subs = dict(zip(params, p))
    return np.array([float(_sym(func).subs({**subs, "x": float(v)}).evalf()) for v in np.atleast_1d(xv)])


# ============================================================
# 复变函数工具
# ============================================================

def _to_complex_arg(z):
    """把字符串或数值转成复数。"""
    if isinstance(z, (int, float)):
        return complex(z)
    return complex(str(z).replace("i", "j").replace("I", "j"))


@_reg
@_tool({
    "properties": {
        "z": {"description": "复数（如 '2+3i' 或数值）"},
        "op": {"type": "string", "description": "运算：norm(模), arg(幅角), conj(共轭), real(实部), imag(虚部)"},
    },
    "required": ["z"],
}, category="复变函数")
def complex_eval(z, op="norm"):
    """复数运算：求模/幅角/共轭/实部/虚部。"""
    c = _to_complex_arg(z)
    op = (op or "norm").lower().strip()
    if op in ("norm", "mod", "模", "绝对值", "abs"):
        return {"text": f"|{z}| = {abs(c):.10g}"}
    if op in ("arg", "argument", "幅角", "角度"):
        import cmath
        ang = cmath.phase(c)
        import math
        return {"text": f"arg({z}) = {ang:.10g} rad = {math.degrees(ang):.10g}°"}
    if op in ("conj", "conjugate", "共轭"):
        return {"text": f"conj({z}) = {c.conjugate()}"}
    if op in ("real", "实部", "re"):
        return {"text": f"Re({z}) = {c.real:.10g}"}
    if op in ("imag", "虚部", "im"):
        return {"text": f"Im({z}) = {c.imag:.10g}"}
    return {"text": f"({z}) = {c}（实部 {c.real}，虚部 {c.imag}，模 {abs(c)}）"}


@_reg
@_tool({
    "properties": {
        "expr": {"type": "string", "description": "复变函数表达式（用 z 作变量），如 sin(z)/z"},
        "z": {"description": "求值点复数，如 '1+i'"},
    },
    "required": ["expr", "z"],
}, category="复变函数")
def complex_f_value(expr, z):
    """求复变函数 f(z) 在给定复点的值。"""
    import sympy as _sp
    zs = _sp.symbols("z")
    try:
        e = _sp.sympify(str(expr).replace("^", "**"), {"I": _sp.I, "i": _sp.I, "pi": _sp.pi, "e": _sp.E})
        cz = _sp.sympify(str(z).replace("i", "I").replace("j", "I"))
        val = _sp.simplify(e.subs(zs, cz))
        return {"text": f"f({z}) = {val}\n（latex: ${_sp.latex(val)}$）"}
    except Exception as ex:
        raise ValueError(f"复变函数求值失败：{ex}")


# ============================================================
# Word 报告导出
# ============================================================

@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行为列名）或 {列名:[值]} 字典"},
        "data_name": {"type": "string", "description": "数据文件/名称，显示在报告标题里"},
        "out_path": {"type": "string", "description": "导出文件路径（.docx）。留空则保存到用户文档目录"},
    },
    "required": ["data"],
}, category="统计分析")
def export_report(data, data_name="数据", out_path=""):
    """生成一份专业 Word 统计报告（含变量信息、描述统计），保存到 .docx 并返回路径。"""
    from stats_report import build_word_report
    from stats_utils import descriptive_stats_table
    import tempfile
    df = _to_df(data)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    desc_table = descriptive_stats_table(df, numeric_cols) if numeric_cols else None
    doc_bytes = build_word_report(df, str(data_name), desc_table=desc_table, alpha=0.05)
    if out_path:
        path = str(out_path)
        if not path.lower().endswith(".docx"):
            path += ".docx"
    else:
        out_dir = os.path.join(os.environ.get("USERPROFILE") or os.path.expanduser("~"), "Documents")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"MatLite_报告_{pd.Timestamp.now():%Y%m%d_%H%M%S}.docx")
    with open(path, "wb") as f:
        f.write(doc_bytes)
    return {"text": f"✅ 报告已生成：{path}\n（含变量信息与描述统计，共 {df.shape[0]} 行 × {df.shape[1]} 列）"}


# ============================================================
# 拍照识题 / OCR（复用 ocr_page 的视觉模型调用）
# ============================================================

def _ai_cfg():
    """读 %APPDATA%\\MatLite\\config.json 里的 AI 连接配置。"""
    from modules import app_settings as _AS
    cfg = _AS.load()
    return {
        "base_url": cfg.get("base_url", "http://localhost:11434"),
        "model": cfg.get("model", "qwen3-vl:8b"),
        "api_key": cfg.get("api_key", ""),
    }


@_reg
@_tool({
    "properties": {
        "image_path": {"type": "string", "description": "图片文件路径（本地）"},
        "image_b64": {"type": "string", "description": "图片 base64 数据（PNG/JPEG）"},
        "prompt": {"type": "string", "description": "识别提示词，默认：转录题目并转成 Python 表达式"},
    },
    "required": [],
}, category="拍照识题")
def ocr_recognize(image_path="", image_b64="", prompt=""):
    """用本地视觉大模型（Ollama）识别图片内容。路径或 base64 二选一。"""
    import base64 as _b64
    from PIL import Image as _Img
    if not image_path and not image_b64:
        raise ValueError("请提供 image_path（图片路径）或 image_b64（图片 base64）之一。")
    if image_b64:
        raw = _b64.b64decode(image_b64)
    else:
        if not os.path.exists(image_path):
            raise ValueError(f"图片文件不存在：{image_path}")
        raw = open(image_path, "rb").read()
    im = _Img.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
    cfg = _ai_cfg()
    url = cfg["base_url"].rstrip("/") + "/api/generate"
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = "Bearer " + cfg["api_key"]
    payload = {
        "model": cfg["model"],
        "prompt": prompt or ("请转录图片中的数学题目文字，并给出关键表达式与解题思路。" ),
        "images": [b64],
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300, proxies=_NO_PROXY)
    r.raise_for_status()
    text = r.json().get("response") or ""
    return {"text": text}


# ============================================================
# v1.6.0 P1 数据科学：分布库 + 机器学习
# ============================================================


@_reg
@_tool({
    "properties": {
        "dist_name": {"description": "分布名：正态分布/t 分布/卡方 χ² 分布/F 分布/Beta 分布/Gamma 分布/韦布尔分布/二项分布/泊松分布/指数分布/均匀分布/几何分布/超几何分布"},
        "params": {"type": "array", "items": {"type": "number"}, "description": "参数列表，按分布顺序（正态 [μ,σ]；t [df]；F [df1,df2]；Beta [α,β]；Gamma [k,θ]；超几何 [M,n,N]）"},
        "x1": {"type": "number", "description": "下界（可空）"},
        "x2": {"type": "number", "description": "上界（可空）"},
    },
    "required": ["dist_name", "params"],
}, category="概率分布")
def distribution_prob(dist_name, params, x1=None, x2=None):
    """常用分布概率计算：给定分布与参数，求均值/标准差/95% 分位数及指定区间的概率。"""
    from modules.prob_page import make_dist
    vals = [float(v) for v in (params or [])]
    d = make_dist(dist_name, vals)
    lines = [f"分布：{dist_name}",
             f"均值={d.mean():.6g}，标准差={d.std():.6g}",
             f"95% 分位数={d.ppf(0.95):.6g}"]
    if x2 is not None and x1 is None:
        lines.append(f"P(X ≤ {x2:g}) = {d.cdf(x2):.6g}")
    elif x1 is not None and x2 is not None:
        lines.append(f"P({x1:g} ≤ X ≤ {x2:g}) = {d.cdf(x2) - d.cdf(x1):.6g}")
    elif x1 is not None:
        lines.append(f"P(X ≥ {x1:g}) = {1 - d.cdf(x1):.6g}")
    return {"text": "\n".join(lines)}


@_reg
@_tool({
    "properties": {
        "data": {"description": "一维数据：数值列表或逗号/空格分隔字符串"},
        "which": {"description": "拟合分布：正态/指数/对数正态/t/Gamma/Beta/Weibull"},
    },
    "required": ["data"],
}, category="概率分布")
def distribution_fit(data, which="正态"):
    """分布拟合：给一批数据，估计最常见分布的参数，并用 K-S 检验判断拟合优劣。"""
    import numpy as np
    from scipy import stats as st
    vals = np.array(_nums(data), dtype=float)
    if len(vals) < 5:
        raise ValueError("至少需要 5 个数据点。")
    w = str(which or "")
    if "t" in w or "学生" in w:
        p = st.t.fit(vals); d = st.t(*p); label = f"t(df={p[0]:.4g})"
    elif "gamma" in w.lower() or "伽马" in w or "Γ" in w:
        p = st.gamma.fit(vals); d = st.gamma(*p); label = f"Gamma(a={p[0]:.4g}, scale={p[-1]:.4g})"
    elif "beta" in w.lower() or "贝塔" in w or "Β" in w:
        p = st.beta.fit(vals); d = st.beta(*p); label = f"Beta(a={p[0]:.4g}, b={p[1]:.4g})"
    elif "weib" in w.lower() or "韦布" in w:
        p = st.weibull_min.fit(vals); d = st.weibull_min(*p); label = f"Weibull(c={p[0]:.4g})"
    elif "指数" in w:
        p = st.expon.fit(vals); d = st.expon(*p); label = f"指数(scale={p[-1]:.4g})"
    elif "对数正态" in w:
        p = st.lognorm.fit(vals); d = st.lognorm(*p); label = f"对数正态(σ={p[0]:.4g})"
    else:
        p = st.norm.fit(vals); d = st.norm(*p); label = f"正态(μ={p[0]:.4g}, σ={p[-1]:.4g})"
    ks, ks_p = st.kstest(vals, d.cdf)
    return {"text": f"拟合分布：{label}\nK-S 检验：统计量={ks:.4f}，p={ks_p:.4f}\n" +
            ("数据近似服从该分布（p>0.05）。" if ks_p > 0.05 else "数据可能不服从该分布（p≤0.05）。")}


@_reg
@_tool({
    "properties": {
        "data": {"description": "数据：CSV 文本（首行列名）或 {列名:[值]} 字典或嵌套列表"},
        "label_col": {"description": "标签（类别）列名，默认最后一列"},
        "model": {"description": "模型：随机森林(默认)/决策树/SVM/kNN"},
        "test_ratio": {"type": "number", "description": "测试集比例，默认 0.2"},
        "cv": {"type": "integer", "description": "交叉验证折数，默认 5，0=关闭"},
    },
    "required": ["data"],
}, category="机器学习")
def ml_classify(data, label_col=None, model="随机森林", test_ratio=0.2, cv=5):
    """机器学习分类：随机森林/决策树/SVM/kNN，输出测试集准确率、交叉验证结果与混淆矩阵/ROC AUC。"""
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    df = _to_df(data)
    if label_col is None:
        col = df.columns[-1]
    else:
        col = _pick_col(label_col) or label_col
        if col not in df.columns:
            raise ValueError(f"找不到标签列：{label_col}")
    X = df.drop(columns=[col]).select_dtypes(include="number").values.astype(float)
    y = df[col].astype(str).values
    if X.shape[0] < 8:
        raise ValueError("样本太少（至少 8 行）。")
    if len(set(y)) < 2:
        raise ValueError("标签至少需要 2 个类别。")
    m = str(model or "随机森林")
    if "随机森林" in m or "random" in m.lower():
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(n_estimators=200, random_state=0); nm = "随机森林"
    elif "决策树" in m or "decision" in m.lower():
        from sklearn.tree import DecisionTreeClassifier
        mdl = DecisionTreeClassifier(random_state=0); nm = "决策树"
    elif "svm" in m.lower() or "支持向量" in m:
        from sklearn.svm import SVC
        mdl = SVC(kernel="rbf", probability=True, random_state=0); nm = "SVM"
    elif "knn" in m.lower() or "近邻" in m:
        from sklearn.neighbors import KNeighborsClassifier
        mdl = KNeighborsClassifier(n_neighbors=max(1, min(5, int(X.shape[0])))); nm = "kNN"
    else:
        raise ValueError("不支持的模型，请选 随机森林/决策树/SVM/kNN。")
    ratio = float(test_ratio or 0.2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ratio, random_state=0)
    mdl.fit(Xtr, ytr)
    acc = mdl.score(Xte, yte)
    lines = [f"模型：{nm}", f"测试集准确率 acc={acc:.1%}（{Xte.shape[0]} 个样本）"]
    cvf = int(cv or 0)
    if cvf and X.shape[0] > max(cvf, 5):
        try:
            s = cross_val_score(mdl, X, y, cv=cvf)
            lines.append(f"交叉验证（{cvf} 折）：acc={s.mean():.1%} ± {s.std():.1%}")
        except Exception as e:
            lines.append(f"交叉验证失败：{e}")
    pred = mdl.predict(Xte)
    classes = sorted(set(y))
    if len(classes) == 2:
        pos = classes[1]
        proba = mdl.predict_proba(Xte)
        fpr, tpr, _ = roc_curve((yte == pos), proba[:, list(classes).index(pos)])
        lines.append(f"ROC AUC（类「{pos}」）={auc(fpr, tpr):.3g}（越接近 1 越好）")
    cm = confusion_matrix(yte, pred, labels=classes)
    lines.append("混淆矩阵（行=真实，列=预测）：")
    for row in cm:
        lines.append("  " + "  ".join(str(int(x)) for x in row))
    return {"text": "\n".join(lines)}