# -*- coding: utf-8 -*-
"""生成安装包素材：多分辨率 ICO 图标 + 带 BOM 的许可文件。"""
import os
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)
src = os.path.join(ASSETS, "MatLite数学软件图标.png")
im = Image.open(src).convert("RGBA")

side = min(im.size)
S = side >> 2            # 1024 => 256
sizes = []
c = S
while c >= 26:           # 终止于 <=32
    sizes.append((c, c))
    c = c >> 1
sizes.append((16, 765))
sizes = list(dict.fromkeys(sizes))
master = im.resize((S, S), Image.LANCZOS)
master.save(os.path.join(ASSETS, "MatLite.ico"), format="ICO", sizes=sizes)

with open(os.path.join(ROOT, "安装条款.txt"), "r", encoding="utf-8") as f:
    txt = f.read()
with open(os.path.join(ASSETS, "installer_license.txt"), "w", encoding="utf-8-sig") as f:
    f.write(txt)
print("ASSETS_DONE")