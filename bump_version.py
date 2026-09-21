# -*- coding: utf-8 -*-
"""版本号统一升级脚本：同步更新所有版本号来源。

一键把 v1.2.3 升级到新版本，自动同步：
  * modules/app_settings.py  的 APP_VERSION
  * installer.iss            的 MyAppVersion / VersionInfoVersion / VersionInfoProductVersion
  * assets/version_info.txt  的 filevers / prodvers / FileVersion / ProductVersion
  * .version                 临时版本文件

用法：
  python bump_version.py              # 补丁号 +1（默认）
  python bump_version.py minor        # 次版本 +1：X.(Y+1).0
  python bump_version.py major        # 主版本 +1：(X+1).0.0
  python bump_version.py 1.3.0        # 直接设为指定版本
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

ISS = os.path.join(ROOT, "installer.iss")
APP_SETTINGS = os.path.join(ROOT, "modules", "app_settings.py")
VERSION_INFO = os.path.join(ROOT, "assets", "version_info.txt")
VERSION_FILE = os.path.join(ROOT, ".version")


def read_iss():
    with open(ISS, "r", encoding="utf-8-sig") as f:
        return f.read()


def _parse_parts(s):
    try:
        parts = [int(p) for p in s.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return parts
    except ValueError:
        sys.exit(f"无法解析版本号: {s}")


def bump():
    text = read_iss()
    m = re.search(r'(?m)^\s*#define\s+MyAppVersion\s+"([^"]+)"\s*$', text)
    if not m:
        sys.exit("未在 installer.iss 中找到 MyAppVersion 定义")
    old = m.group(1)
    parts = _parse_parts(old)

    arg = (sys.argv[1] if len(sys.argv) > 1 else "patch").lower()
    if arg == "major":
        parts = [parts[0] + 1, 0, 0]
    elif arg == "minor":
        parts = [parts[0], parts[1] + 1, 0]
    elif re.fullmatch(r"\d+(\.\d+){1,2}", arg):
        parts = _parse_parts(arg)
    else:  # default patch
        parts[2] = parts[2] + 1

    new = ".".join(str(p) for p in parts)
    new_tuple = "({}, {}, 0, 0)".format(*parts)  # version_info.txt 用 4 段

    # 1) installer.iss —— 三处版本定义
    text = text.replace(m.group(0), m.group(0).replace(old, new))
    text = re.sub(r'(?m)^\s*VersionInfoVersion=\d+\.\d+\.\d+\s*$',
                  f"VersionInfoVersion={new}", text)
    text = re.sub(r'(?m)^\s*VersionInfoProductVersion=\d+\.\d+\.\d+\s*$',
                  f"VersionInfoProductVersion={new}", text)
    with open(ISS, "w", encoding="utf-8-sig") as f:
        f.write(text)

    # 2) modules/app_settings.py —— APP_VERSION = "x.y.z"
    if os.path.exists(APP_SETTINGS):
        s = open(APP_SETTINGS, encoding="utf-8").read()
        s = re.sub(r'(?m)^APP_VERSION\s*=\s*"[^"]*"',
                   f'APP_VERSION = "{new}"', s, count=1)
        with open(APP_SETTINGS, "w", encoding="utf-8") as f:
            f.write(s)

    # 3) assets/version_info.txt —— filevers/prodvers + FileVersion/ProductVersion
    if os.path.exists(VERSION_INFO):
        v = open(VERSION_INFO, encoding="utf-8-sig").read()
        v = re.sub(r'filevers=\([^)]*\)', f'filevers={new_tuple}', v, count=1)
        v = re.sub(r'prodvers=\([^)]*\)', f'prodvers={new_tuple}', v, count=1)
        v = re.sub(r"(StringStruct\(u'FileVersion', u'[^']*'\))",
                   f"StringStruct(u'FileVersion', u'{new}')", v, count=1)
        v = re.sub(r"(StringStruct\(u'ProductVersion', u'[^']*'\))",
                   f"StringStruct(u'ProductVersion', u'{new}')", v, count=1)
        with open(VERSION_INFO, "w", encoding="utf-8") as f:
            f.write(v)

    # 4) .version 临时文件
    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        f.write(new)

    print(f"OLD_VERSION={old}")
    print(f"NEW_VERSION={new}")
    print(f"RELEASE_DIR=releases\\v{new}")


if __name__ == "__main__":
    bump()