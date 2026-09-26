# -*- coding: utf-8 -*-
"""PyInstaller 构建脚本：python build_exe.py（或由 打包.bat 调用）"""
import sys

sys.setrecursionlimit(20000)

import PyInstaller.__main__

import os
out = os.environ.get("MATLITE_OUT", "dist")  # 可设 dist_v2 等避开被锁目录
work = os.environ.get("MATLITE_WORK", "build")
PyInstaller.__main__.run([
    "--noconfirm", "--clean", "--windowed", "--name", "MatLite",
    "--distpath", out,
    "--workpath", work,
    "--icon", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "MatLite.ico"),
    "--version-file", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "version_info.txt"),
    "--collect-all", "customtkinter",
    "--collect-data", "matplotlib",
    "--hidden-import", "cv2",
    "--collect-all", "cv2",
    "--exclude-module", "torch",
    "--exclude-module", "tensorboard",
    "--exclude-module", "tkinter.test",
    "app.py",
])
print("=== BUILD_DONE ===")