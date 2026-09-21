# -*- coding: utf-8 -*-
"""公告与更新检查：后台线程拉取公告 JSON + 最新版本号（单向、只拉取、不上传任何数据）。

数据源：免费静态托管（默认 GitHub raw 文件，可在设置里改地址）。
任何网络失败都静默忽略，绝不影响本地功能。

配置读写统一通过 app_settings 模块，避免多套读写逻辑冲突。
"""

import json
import os
import webbrowser
import threading

import requests
import customtkinter as ctk

from modules import app_settings as _S

DEFAULT_ANNOUNCE_URL = "https://raw.githubusercontent.com/zoilzo/MatLite/main/announce.json"
DEFAULT_RELEASE_REPO = "zoilzo/MatLite"
_NO_PROXY = {"http": None, "https": None}


def _ver_key(tag):
    """把 'v1.2.3' 或 '1.2.3' 转成可比较的整数元组。"""
    parts = str(tag).lstrip("vV").split(".")
    out = []
    for p in parts:
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        try:
            out.append(int(num))
        except ValueError:
            break
    return tuple(out)


def check_updates(current_version):
    """后台线程调用。返回 dict 或 None（失败静默返回 None）。
    结果里可能含 'announce' 与/或 'version'。"""
    cfg = _S.load()
    if not cfg.get("announce_on", True):
        return None
    res = {}

    # ---- 公告 ----
    url = (cfg.get("announce_url") or DEFAULT_ANNOUNCE_URL).strip()
    if url:
        try:
            r = requests.get(url, timeout=3, proxies=_NO_PROXY,
                             headers={"User-Agent": "MatLite/" + str(current_version)})
            r.raise_for_status()
            d = r.json()
            aid = str(d.get("id", "")).strip()
            if aid and aid != str(cfg.get("last_seen_id", "")).strip():
                res["announce"] = {
                    "id": aid,
                    "title": str(d.get("title", "公告")),
                    "body": str(d.get("body", "")),
                    "url": str(d.get("url", "")),
                }
        except Exception:
            pass

    # ---- 版本升级提醒 ----
    repo = (cfg.get("release_repo") or DEFAULT_RELEASE_REPO).strip()
    if repo:
        try:
            r = requests.get(f"https://api.github.com/repos/{repo}/releases/latest",
                             timeout=3, proxies=_NO_PROXY,
                             headers={"User-Agent": "MatLite/" + str(current_version),
                                      "Accept": "application/vnd.github+json"})
            r.raise_for_status()
            d = r.json()
            tag = str(d.get("tag_name", "")).strip()
            if tag and _ver_key(tag) and _ver_key(tag) > _ver_key(current_version):
                res["version"] = {
                    "tag": tag.lstrip("vV"),
                    "name": str(d.get("name", "")),
                    "url": str(d.get("html_url", "")),
                }
        except Exception:
            pass

    return res or None


def show_updates(app, data):
    """在主线程展示公告/升级提醒对话框。"""
    ann = data.get("announce")
    ver = data.get("version")
    if not ann and not ver:
        return

    dlg = ctk.CTkToplevel(app)
    dlg.title("通知")
    dlg.geometry("500x360")
    dlg.transient(app)
    dlg.attributes("-topmost", True)
    dlg.grid_columnconfigure(0, weight=1)

    title = ""
    body = ""
    link = ""
    if ver:
        title = f"发现新版本 v{ver['tag']}"
        body = (f"当前版本 v{ver['tag']} 已发布。\n\n{ver['name'] or '建议更新'}\n\n"
                f"请前往下载页安装最新版。")
        link = ver.get("url", "")
        if ann:
            body += f"\n\n—— 最新公告 ——\n{ann['title']}\n\n{ann['body']}"
    else:
        title = ann.get("title") or "公告"
        body = ann.get("body") or ""
        link = ann.get("url", "")

    ctk.CTkLabel(dlg, text="📢 " + title, font=ctk.CTkFont(size=16, weight="bold")).grid(
        row=0, column=0, padx=20, pady=(18, 6), sticky="w")
    ctk.CTkLabel(dlg, text=body, font=ctk.CTkFont(size=13), justify="left",
                 anchor="nw", wraplength=440).grid(row=1, column=0, padx=20, pady=(0, 10))

    btns = ctk.CTkFrame(dlg, fg_color="transparent")
    btns.grid(row=2, column=0, padx=20, pady=(6, 12), sticky="ew")
    btns.grid_columnconfigure((0, 1), weight=1)
    if link:
        ctk.CTkButton(btns, text="前往查看", height=36,
                      command=lambda: webbrowser.open(link)).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
    ctk.CTkButton(btns, text="知道了", height=36, fg_color="gray40",
                  command=dlg.destroy).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    # 记住了公告 id，下次不再弹出
    if ann and ann.get("id"):
        _S.set_val("last_seen_id", ann["id"])

    dlg.after(120, dlg.focus_set)
    dlg.grab_set()