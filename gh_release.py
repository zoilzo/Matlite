# -*- coding: utf-8 -*-
"""GitHub 发布脚本（Git Data API 上传源码 + Releases API 上传资产）。

token 从环境变量 MATLITE_GITHUB_TOKEN 读取；否则回退到 仓库/.github_token 或桌面 token.txt。
token 只在进程内存中用于请求头，绝不写入仓库/历史/日志。

用法：
  # 全量发布（上传源码到 main + 建 Release + 上传资产 + 公告）
  MATLITE_GITHUB_TOKEN=$(cat C:/Users/19627/Desktop/token.txt) python gh_release.py --version 1.8.0

  # 复用已存在的 Release：只补传资产 + 公告（源码/Release 已创建时用）
  MATLITE_GITHUB_TOKEN=$(cat C:/Users/19627/Desktop/token.txt) python gh_release.py --version 1.8.0 --recover

  # 只补发旧版本 Release（新建 Release + 传资产，不重传源码/公告）
  MATLITE_GITHUB_TOKEN=$(cat C:/Users/19627/Desktop/token.txt) python gh_release.py --version 1.6.0 --release-only
"""
import os
import sys
import base64
import json
import subprocess
import argparse

import requests

# 控制台可能是 GBK，强制 stdout/stderr 用 utf-8，避免 emoji/特殊字符抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "zoilzo/MatLite"
API = "https://api.github.com"
COMMITTER = {"name": "zoilzo", "email": "17588853057@163.com"}


def load_token():
    t = (os.environ.get("MATLITE_GITHUB_TOKEN") or "").strip()
    if t:
        return t
    for cand in (os.path.join(SCRIPT_DIR, ".github_token"),
                 r"C:\Users\19627\Desktop\token.txt"):
        if os.path.exists(cand):
            return open(cand, encoding="utf-8").read().strip()
    sys.exit("未找到 GitHub token。请设环境变量 MATLITE_GITHUB_TOKEN，或放置 仓库/.github_token / 桌面/token.txt。")


TOKEN = load_token()


def gh(method, url, **kw):
    kw.setdefault("headers", {})["Authorization"] = f"token {TOKEN}"
    kw.setdefault("headers", {}).setdefault("Accept", "application/vnd.github+json")
    kw.setdefault("timeout", 90)
    r = requests.request(method, url, **kw)
    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"API {method} {url} -> {r.status_code}: {body}")
    if r.status_code in (204, 202) or not r.content:
        return {}
    return r.json()


def upload_asset(rel_id, name, path, ctype):
    print(f"  -> 上传 {name} ...")
    if not os.path.exists(path):
        print(f"     [跳过] 不存在: {path}")
        return
    with open(path, "rb") as fh:
        up = requests.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rel_id}/assets?name={name}",
            headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json",
                     "Content-Type": ctype},
            data=fh, timeout=600)
    if up.status_code >= 400:
        print(f"     [失败] {name} -> {up.status_code}: {up.text[:200]}")
        return
    print(f"     [OK] {name} -> {up.json().get('browser_download_url')}")


def publish_source_to_main(version):
    """把当前工作树被 git 跟踪的源码上传到 main（只读磁盘，不 stage 未跟踪文件）。"""
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    files = [f for f in files if f and os.path.isfile(f)]
    print(f"源码上传：{len(files)} 个跟踪文件")
    if not files:
        print("  无跟踪文件，跳过源码上传")
        try:
            return gh("GET", f"{API}/repos/{REPO}/git/refs/heads/main")["object"]["sha"]
        except Exception:
            return "main"

    blobs = []
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            data = fh.read()
        blob = gh("POST", f"{API}/repos/{REPO}/git/blobs",
                  json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
        blobs.append({"path": f.replace("\\", "/"), "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if (i + 1) % 20 == 0:
            print(f"  已上传 {i+1}/{len(files)} blobs")

    ref = None
    try:
        ref = gh("GET", f"{API}/repos/{REPO}/git/refs/heads/main")["object"]["sha"]
    except Exception:
        pass

    tree = gh("POST", f"{API}/repos/{REPO}/git/trees", json={"tree": blobs})
    print("  tree 已创建:", tree["sha"])
    commit_data = {"message": f"MatLite v{version}: deep compute + reproducible workflow",
                    "tree": tree["sha"]}
    if ref:
        commit_data["parents"] = [ref]
    commit = gh("POST", f"{API}/repos/{REPO}/git/commits", json=commit_data)
    print("  commit 已创建:", commit["sha"])

    if ref:
        gh("PATCH", f"{API}/repos/{REPO}/git/refs/heads/main", json={"sha": commit["sha"], "force": False})
    else:
        gh("POST", f"{API}/repos/{REPO}/git/refs", json={"ref": "refs/heads/main", "sha": commit["sha"]})
    print("  main 已更新 ->", commit["sha"][:10])
    return commit["sha"]


def find_release(tag):
    """按 tag 查找已存在的 Release，返回 dict 或 None。"""
    page = 1
    while True:
        revs = gh("GET", f"{API}/repos/{REPO}/releases", params={"per_page": 100, "page": page})
        for r in revs:
            if r.get("tag_name") == tag:
                return r
        if len(revs) < 100:
            break
        page += 1
    return None


def create_release(version, target_sha):
    tag = "v" + version
    rel_dir = os.path.join(SCRIPT_DIR, "releases", tag)
    notes = os.path.join(rel_dir, "RELEASE_NOTES.md")
    body = open(notes, encoding="utf-8").read() if os.path.exists(notes) else f"MatLite v{version} 发布。"
    release = gh("POST", f"{API}/repos/{REPO}/releases", json={
        "tag_name": tag, "target_commitish": target_sha, "name": f"MatLite v{version}",
        "body": body, "draft": False, "prerelease": False})
    print(f"Release 已创建: {release['html_url']}")
    return release


def upload_all_assets(rel_id, version):
    """同步本版本资产到 Release。GitHub 不保留非 ASCII 文件名，故用 ASCII 输出名。
    先删除已存在的同名非目标资产（清掉被 GitHub 改名的残留），再补齐缺失。"""
    rel_dir = os.path.join(SCRIPT_DIR, "releases", "v" + version)
    want = [
        ("MatLiteSetup.exe", os.path.join(rel_dir, "MatLiteSetup.exe"), "application/octet-stream"),
        ("MatLite-portable.zip", os.path.join(rel_dir, "MatLite-便携版.zip"), "application/zip"),
        ("README.txt", os.path.join(rel_dir, "使用说明.txt"), "text/plain"),
        ("LICENSE.txt", os.path.join(rel_dir, "安装条款.txt"), "text/plain"),
    ]
    want_names = {w[0] for w in want}
    # 清理残留资产（其名不在目标集合中的全部删除）
    try:
        existing = gh("GET", f"{API}/repos/{REPO}/releases/{rel_id}/assets")
    except Exception:
        existing = []
    for a in existing:
        if a.get("name") not in want_names:
            print(f"  [清理] 删除残留资产 {a['name']!r} ...")
            try:
                gh("DELETE", f"{API}/repos/{REPO}/releases/assets/{a['id']}")
            except Exception as e:
                print(f"    [失败] {e}")
    # 补齐缺失资产
    for name, path, ctype in want:
        if name in want_names and any(a.get("name") == name for a in existing):
            continue
        upload_asset(rel_id, name, path, ctype)


def update_main_file(relpath, message):
    """把本地 relpath 文件经 Contents API 写回 main 分支（幂等：有 sha 则 PUT 更新，无则新建）。"""
    src = os.path.join(SCRIPT_DIR, relpath)
    if not os.path.exists(src):
        print(f"  跳过 {relpath}：文件不存在")
        return
    data = open(src, "rb").read()
    sha = None
    try:
        sha = gh("GET", f"{API}/repos/{REPO}/contents/{relpath}", params={"ref": "main"})["sha"]
    except Exception:
        pass
    try:
        gh("PUT", f"{API}/repos/{REPO}/contents/{relpath}", json={
            "message": message, "content": base64.b64encode(data).decode(),
            "branch": "main", "sha": sha, "committer": COMMITTER, "author": COMMITTER})
        print(f"  [OK] {relpath} 已更新")
    except Exception as e:
        print(f"  [失败] {relpath}:", e)


def upload_announce(version):
    print("上传 announce.json ...")
    update_main_file("announce.json", f"公告: MatLite v{version} 发布")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="版本号，如 1.8.0")
    ap.add_argument("--release-only", action="store_true", help="只补发旧版 Release（不重传源码/公告）")
    ap.add_argument("--recover", action="store_true", help="复用已存在 Release，只补传资产+公告（不重传源码）")
    args = ap.parse_args()

    if not args.version:
        sys.path.insert(0, SCRIPT_DIR)
        from modules import app_settings as _S
        version = _S.APP_VERSION
    else:
        version = args.version.lstrip("v")

    print(f"=== GitHub 发布 v{version} ===")

    if args.recover:
        tag = "v" + version
        rel = find_release(tag)
        if not rel:
            print(f"  [失败] 未找到 Release {tag}，请先运行全量发布或 --release-only。")
            return
        rel_id = rel["id"]
        print(f"  复用已有 Release #{rel_id}: {rel['html_url']}")
        upload_all_assets(rel_id, version)
        upload_announce(version)
        print("\n=== 恢复完成 ===")
        return

    if args.release_only:
        rel = create_release(version, "main")
        upload_all_assets(rel["id"], version)
        print("\n=== Release 补发完成 ===")
        return

    # 全量发布：源码 -> main -> Release -> 资产 -> 公告（幂等：先删同 tag 旧 Release）
    tag = "v" + version
    old = find_release(tag)
    if old:
        print("  删除已存在的同 tag Release ...")
        try:
            gh("DELETE", f"{API}/repos/{REPO}/releases/{old['id']}")
        except Exception as e:
            print(f"    [失败] 删除 Release: {e}")
        try:
            gh("DELETE", f"{API}/repos/{REPO}/git/refs/tags/{tag}")
        except Exception:
            pass
    target = publish_source_to_main(version)
    rel = create_release(version, target)
    upload_all_assets(rel["id"], version)
    # 同步 main 上的站点页与公告（让页面链接指向正确的 ASCII 资产名）
    for rp, msg in ((os.path.join(SCRIPT_DIR, "index.html"), "官网页面同步 v" + version),
                    (os.path.join(SCRIPT_DIR, "RELEASE_NOTES.md"), "发版说明 v" + version),
                    (os.path.join(SCRIPT_DIR, "announce.json"), "公告 v" + version + " 发布")):
        if os.path.exists(rp):
            update_main_file(os.path.relpath(rp, SCRIPT_DIR), msg)
    print("\n=== 发布完成 ===")


if __name__ == "__main__":
    main()
