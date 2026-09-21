# -*- coding: utf-8 -*-
"""剪贴板工具：把 matplotlib 图复制成位图到系统剪贴板（Windows，纯 ctypes，无额外依赖）。"""

import ctypes
import io
from ctypes import wintypes

from PIL import Image

CF_DIB = 8
GMEM_MOVEABLE = 0x0002


def copy_figure_to_clipboard(fig):
    """把 matplotlib 图渲染后复制到系统剪贴板。成功返回 True。"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        bmp = io.BytesIO()
        img.save(bmp, format="BMP")
        dib = bmp.getvalue()[14:]   # 去掉 14 字节位图文件头，只留 DIB
        _set_dib(dib)
        return True
    except Exception:
        return False


def copy_text_to_clipboard(text, root=None):
    """把文本复制到剪贴板。"""
    import tkinter
    try:
        if root is None:
            root = tkinter._default_root
        root.clipboard_clear()
        root.clipboard_append(text)
        return True
    except Exception:
        return False


def _set_dib(dib_bytes):
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
    kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
    kernel32.GlobalFree.argtypes = (ctypes.c_void_p,)
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = (wintypes.UINT, ctypes.c_void_p)

    h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib_bytes))
    if not h:
        return
    p = kernel32.GlobalLock(h)
    if p:
        ctypes.memmove(p, dib_bytes, len(dib_bytes))
        kernel32.GlobalUnlock(h)
    if user32.OpenClipboard(0):
        try:
            user32.EmptyClipboard()
            if not user32.SetClipboardData(CF_DIB, h):
                kernel32.GlobalFree(h)
        except Exception:
            kernel32.GlobalFree(h)
        finally:
            user32.CloseClipboard()
    else:
        kernel32.GlobalFree(h)