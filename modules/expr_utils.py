# -*- coding: utf-8 -*-
"""表达式工具：统一各页面重复的 SAFE dict、evalf、nums 等辅助函数。

SAFE 是所有页面通用的数学函数字典（供 eval 使用，禁用 __builtins__），
避免了 4 个页面各自维护一份且可能存在分歧的问题。
"""

import numpy as np

# ---- 统一数学函数字典（各页面共用） ----
SAFE = {
    # 三角函数
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    # 双曲函数
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    # 基础运算
    "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
    # 对数
    "log": np.log, "ln": np.log, "lg": np.log10, "log2": np.log2, "log10": np.log10,
    # 常数
    "pi": np.pi, "e": np.e, "inf": np.inf,
    # 取整
    "ceil": np.ceil, "floor": np.floor,
}


def evalf(expr, var, val):
    """安全求值：用 SAFE 字典限制 eval 的命名空间，支持 ╳→** 转换。"""
    e = expr.strip().replace("^", "**")
    return eval(e, {"__builtins__": {}}, {**SAFE, var: val})


def evalf_subs(expr, subs):
    """按字典代入多个变量后安全求值（配合 slide/f 滑杆参数等使用）。"""
    e = expr.strip().replace("^", "**")
    return eval(e, {"__builtins__": {}}, {**SAFE, **subs})


def eval2_subs(e, X, Y, subs=None):
    """二维向量化求值（可用于曲面/等高线图），可额外代入其它参数（如滑杆 a）。

    SAFE 字典里的函数都是 numpy ufunc（会广播），直接传入整个数组，
    一次 numpy 调用即可完成逐点计算，避免 25 万次 Python eval 循环。
    """
    ee = e.replace("^", "**")
    with np.errstate(all="ignore"):
        out = eval(ee, {"__builtins__": {}}, {**SAFE, "x": X, "y": Y, **(subs or {})})
    return np.broadcast_to(np.asarray(out, dtype=float), X.shape).copy()


def eval2(e, X, Y):
    """二维向量化求值（无额外参数时调用 eval2_subs）。"""
    return eval2_subs(e, X, Y, None)


def nums(text):
    """解析以空格、逗号等分隔的数字字符串，返回 float 列表。"""
    s = text.strip().replace(",", " ").replace("，", " ").replace("；", " ").replace(";", " ")
    return [float(x) for x in s.split()]