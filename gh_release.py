# -*- coding: utf-8 -*-
"""GitHub 发布脚本（Git Data API 上传源码 + Releases API 上传资产）。

重大改动（v1.8 发布用）：
  * token 从环境变量 MATLITE_GITHUB_TOKEN 读取；否则回退到 仓库/.github_token 或
    桌面  token.txt。token 只在进程内存中用于请求头，绝不写入仓库/历史/日志。
  * 支持 --version x.y.z 指定版本号（默认取当前 app_settings.APP_VERSION）。
  * 支持 --release-only：只创建 Release 并上传资产，不重传源码到 main。
  * 资产补齐 MatLiteSetup.exe / 便携zip / 使用说明.txt / 安装条款.txt。

用法：
  # 首次/主版本发布（上传源码 + 建 Release + 上传资产 + 公告）
  MATLITE_GITHUB_TOKEN=$(cat C:/Users/19627/Desktop/token.txt) python gh_release.py --version 1.8.0

  # 只补发 Release（不进源码，只带各版本安装包）
  MATLITE_GITHUB_TOKEN=$(cat C:/Users/19627/Desktop/token.txt) python gh_release.py --version 1.6.0 --release-only
"""
import os
import sys
import base64
import json
import subprocess
import argparse

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "zoilzo/MatLite"
API = "https://api.github.com"
COMMITTER = {"name": "zoilzo", "email": "17588853057@163.com"}


def load_token():
    """读取 GitHub token：环境变量优先，其次仓库 .github_token，最后 桌面 token.txt。"""
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
    kw.setdefault("timeout", 60)
    r = requests.request(method, url, **kw)
    r.raise_for_status()
    return r.json()


def upload_asset(rel_id, name, path, ctype):
    print(f"  \u2b06  上传 {name} ...")
    with open(path, "rb") as fh:
        up = requests.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rel_id}/assets?name={name}",
            headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json",
                     "Content-Type": ctype},
            data=fh, timeout=600)
    up.raise_for_status()
    print(f"    \u2713 {name} -> {up.json().get('browser_download_url')}")


def publish_source_to_main(version):
    """把当前工作树里被 git 跟踪的源码上传到 main（只读磁盘，不 stage 未跟踪文件）。"""
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    files = [f for f in files if f and os.path.isfile(f)]
    print(f"源码上传：{len(files)} 个跟踪文件")
    if not files:
        print("  无跟踪文件，跳过源码上传")
        ref = None
        try:
            ref = gh("GET", f"{API}/repos/{REPO}/git/refs/heads/main")["object"]["sha"]
        except requests.HTTPError:
            pass
        return ref

    blobs = []
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            data = fh.read()
        blob = gh("POST", f"{API}/repos/{REPO}/git/blobs",
                  json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
        blobs.append({"path": f.replace("\\", "/"), "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if (i + 1) % 20 == 0:
            print(f"  \u5df2\u4e0a\u4f20 {i+1}/{len(files)} blobs")

    ref = None
    try:
        ref = gh("GET", f"{API}/repos/{REPO}/git/refs/heads/main")["object"]["sha"]
    except requests.HTTPError:
        pass

    tree = gh("POST", f"{API}/repos/{REPO}/git/trees", json={"tree": blobs})
    print("  tree \u5df2\u521b\u5efa:", tree["sha"])
    commit_data = {"message": f"MatLite v{version}: \u6df1\u5ea6\u8ba1\u7b97 + \u53ef\u590d\u73b0\u5de5\u4f5c\u6d41\n\n- \u77e9\u9635\u5206\u89e3 LU/QR/SVD\uff1a\u4e00\u952e\u5206\u89e3\u5e76\u9a8c\u8bc1\u91cd\u5efa\n- \u5e38\u5fae\u5206\u65b9\u7a0b\u6570\u503c\u89e3\uff1a\u65b9\u5411\u573a + \u89e3\u66f2\u7ebf (RK45/\u6b27\u62c9)\n- \u65b9\u5411\u573a\uff1ady/dx=f(x,y) \u659c\u7387\u573a\u53ef\u89c6\u5316\n- \u4efb\u610f\u5947\u70b9\u7559\u6570\uff1a\u81ea\u52a8\u5217\u51fa\u5168\u90e8\u5947\u70b9\u5e76\u9010\u4e00\u6c42\u7559\u6570\n- \u53ef\u590d\u73b0\u5de5\u4f5c\u6d41\uff1a\u6bcf\u9898\u53ef\u5bfc\u51fa\u53ef\u8fd0\u884c .py + .md \u914d\u65b9",
                    "tree": tree["sha"]}
    if ref:
        commit_data["parents"] = [ref]
    commit = gh("POST", f"{API}/repos/{REPO}/git/commits", json=commit_data)
    print("  commit \u5df2\u521b\u5efa:", commit["sha"])

    if ref:
        gh("PATCH", f"{API}/repos/{REPO}/git/refs/heads/main", json={"sha": commit["sha"], "force": False})
    else:
        gh("POST", f"{API}/repos/{REPO}/git/refs", json={"ref": "refs/heads/main", "sha": commit["sha"]})
    print("  main \u5df2\u66f4\u65b0 ->", commit["sha"][:10])
    return commit["sha"]


def create_release(version, target_sha):
    """创建 Release（tag 为 v{version}，指向 target_sha），并上传资产。"""
    tag = "v" + version
    rel_dir = os.path.join(SCRIPT_DIR, "releases", tag)
    notes = os.path.join(rel_dir, "RELEASE_NOTES.md")
    if os.path.exists(notes):
        body = open(notes, encoding="utf-8").read()
    else:
        body = f"MatLite v{version} 发布。"
    release = gh("POST", f"{API}/repos/{REPO}/releases", json={
        "tag_name": tag, "target_commitish": target_sha, "name": f"MatLite v{version}",
        "body": body, "draft": False, "prerelease": False})
    rel_id = release["id"]
    print(f"Release \u5df2\u521b\u5efa: {release['html_url']}")
    return rel_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="版本号，如 1.8.0")
    ap.add_argument("--release-only", action="store_true", help="只建 Release+资产，不重传源码到 main")
    args = ap.parse_args()
    if not args.version:
        sys.path.insert(0, SCRIPT_DIR)
        from modules import app_settings as _S
        version = _S.APP_VERSION
    else:
        version = args.version.lstrip("v")
    rel_dir = os.path.join(SCRIPT_DIR, "releases", "v" + version)
    print(f"=== GitHub \u53d1\u5e03 v{version} ===")

    # 建 Release 指向的目标 commit
    target = "main"
    if not args.release_only:
        target = publish_source_to_main(version)

    rel_id = create_release(version, target)

    # 上传资产
    base = os.path.join(rel_dir, "MatLiteSetup.exe")
    zipf = os.path.join(rel_dir, "MatLite-\u4fbf\u643a\u7248.zip")
    usage = os.path.join(rel_dir, "\u4f7f\u7528\u8bf4\u660e.txt")
    lic = os.path.join(rel_dir, "\u5b89\u88c5\u6761\u6b3e.txt")
    if os.path.exists(base):
        upload_asset(rel_id, "MatLiteSetup.exe", base, "application/octet-stream")
    else:
        print(f"  \u26a0 \u672a\u627e\u5230 {base}")
    if os.path.exists(zipf):
        upload_asset(rel_id, "MatLite-\u4fbf\u643a\u7248.zip", zipf, "application/zip")
    else:
        print(f"  \u26a0 \u672a\u627e\u5230 {zipf}")
    if os.path.exists(usage):
        upload_asset(rel_id, "\u4f7f\u7528\u8bf4\u660e.txt", usage, "text/plain")
    if os.path.exists(lic):
        upload_asset(rel_id, "\u5b89\u88c5\u6761\u6b3e.txt", lic, "text/plain")

    # 仅主版本发布时更新仓库根目录公告（保留最新版）
    if not args.release_only:
        announce_src = os.path.join(SCRIPT_DIR, "announce.json")
        if os.path.exists(announce_src):
            print("\u4e0a\u4f20 announce.json ...")
            data = open(announce_src, "rb").read()
            sha = None
            try:
                sha = gh("GET", f"{API}/repos/{REPO}/contents/announce.json", params={"ref": "main"})["sha"]
            except requests.HTTPError:
                pass
            gh("PUT", f"{API}/repos/{REPO}/contents/announce.json", json={
                "message": f"\u516c\u544a: MatLite v{version} \u53d1\u5e03", "content": base64.b64encode(data).decode(),
                "branch": "main", "sha": sha, "committer": COMMITTER, "author": COMMITTER})
            print("  \u2713 announce.json \u5df2\u66f4\u65b0")

    print("\n=== GitHub \u53d1\u5e03\u5b8c\u6210\uff01===")


if __name__ == "__main__":
    main()
