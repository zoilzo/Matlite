# -*- coding: utf-8 -*-
"""机器学习入门：KMeans 聚类 / 决策树分类。

数据格式：每行一个样本，特征用空格或逗号分隔；决策树时最后一列是类别标签。
机器学习引擎用 scikit-learn，界面全中文，面向不懂代码的学生。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OPS = ["KMeans 聚类", "决策树分类", "随机森林分类", "支持向量机 SVM", "kNN 最近邻", "PCA 降维"]

SAMPLE = """1.0 2.0
1.5 1.8
5.0 8.0
8.0 8.0
1.0 0.6
9.0 11.0
8.0 2.0
10.0 2.0
9.0 3.0"""


def parse_matrix(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()])
    return rows


class MlPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数 =================
        left = ctk.CTkScrollableFrame(self, width=390, corner_radius=12, label_text="机器学习")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="数据（每行一个样本；决策树时最后一列是类别）", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        self.data = ctk.CTkTextbox(left, height=180, font=ctk.CTkFont(family="Consolas", size=12))
        self.data.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.data.insert("1.0", SAMPLE)

        ctk.CTkLabel(left, text="操作", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 2))
        self.op = ctk.CTkOptionMenu(left, values=OPS)
        self.op.grid(row=3, column=0, sticky="ew", padx=12)
        self.op.set(OPS[0])

        ctk.CTkLabel(left, text="聚类数 K（KMeans 用）", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="w", padx=14, pady=(8, 2))
        self.k = ctk.CTkEntry(left, height=32)
        self.k.grid(row=5, column=0, sticky="ew", padx=12)
        self.k.insert(0, "2")

        ctk.CTkLabel(left, text="决策树最大深度（决策树用，可空=不限）", font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, sticky="w", padx=14, pady=(8, 2))
        self.depth = ctk.CTkEntry(left, height=32)
        self.depth.grid(row=7, column=0, sticky="ew", padx=12)
        self.depth.insert(0, "")

        ctk.CTkLabel(left, text="测试集比例（分类用，默认 0.2）", font=ctk.CTkFont(size=12)).grid(
            row=8, column=0, sticky="w", padx=14, pady=(8, 2))
        self.test_ratio = ctk.CTkEntry(left, height=32)
        self.test_ratio.grid(row=9, column=0, sticky="ew", padx=12)
        self.test_ratio.insert(0, "0.2")

        ctk.CTkLabel(left, text="交叉验证折数 CV（0=关闭，分类用）", font=ctk.CTkFont(size=12)).grid(
            row=10, column=0, sticky="w", padx=14, pady=(8, 2))
        self.cv_folds = ctk.CTkEntry(left, height=32)
        self.cv_folds.grid(row=11, column=0, sticky="ew", padx=12)
        self.cv_folds.insert(0, "5")

        ctk.CTkButton(left, text="⚡ 训 练", height=42, command=self._run).grid(
            row=12, column=0, sticky="ew", padx=12, pady=(10, 2))

        tip = ("说明：\n"
               "· KMeans：按距离把样本自动分成 K 簇，画散点+簇中心；\n"
               "· 决策树/随机森林/SVM/kNN：用最后一列当标签训练分类器，\n"
               "   画混淆矩阵，二分类画 ROC，并自动交叉验证；\n"
               "· PCA：降维到前 2 个主成分画散点，看数据结构；\n"
               "· 数据太少或类别不平衡时结果仅供参考。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=360).grid(
            row=13, column=0, sticky="w", padx=14, pady=(4, 12))

        # ================= 右：结果 + 画布 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(right, text="训练结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13), height=300)
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        box = ctk.CTkFrame(right, fg_color="transparent")
        box.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 4))
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)
        self.figure = plt.Figure(figsize=(7, 58), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        toolbar = NavigationToolbar2Tk(self.canvas, box)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.ax.set_title("机器学习图形")
        self.canvas.draw()

    def _msg(self, s):
        self.out.configure(state="normal")
        self.out.insert("end", s + "\n")
        self.out.see("end")
        self.out.configure(state="disabled")

    def _clear_out(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")

    def _run(self):
        self._clear_out()
        op = self.op.get()
        try:
            if op == OPS[0]:
                mat = np.array(parse_matrix(self.data.get("1.0", "end")))
                if mat.shape[0] < 5:
                    raise ValueError("样本太少（至少 6 行）。")
                self._kmeans(mat)
            elif op == OPS[1]:
                self._classify(self.data.get("1.0", "end"), "决策树分类")
            elif op == OPS[2]:
                self._classify(self.data.get("1.0", "end"), "随机森林分类")
            elif op == OPS[3]:
                self._classify(self.data.get("1.0", "end"), "支持向量机 SVM")
            elif op == OPS[4]:
                self._classify(self.data.get("1.0", "end"), "kNN 最近邻")
            elif op == OPS[5]:
                self._pca(self.data.get("1.0", "end"))
        except Exception as e:
            self._msg(f"出错：{e}")
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("机器学习", op, self.out.get("1.0", "end")[:200])

    def _kmeans(self, mat):
        from sklearn.cluster import KMeans
        k = max(2, int(float(self.k.get() or 2)))
        k = min(k, mat.shape[0])
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(mat)
        labels = km.labels_
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        if mat.shape[1] >= 3:
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2], c=labels, cmap="viridis", s=40)
            self.ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                            km.cluster_centers_[:, 2], c="red", marker="x", s=120, label="簇中心")
        else:
            self.ax.scatter(mat[:, 0], mat[:, 1], c=labels, cmap="viridis", s=60)
            self.ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                            c="red", marker="x", s=150, label="簇中心")
            self.ax.legend(fontsize=9)
        self.ax.set_title(f"KMeans 聚类（K={k}）")
        self.ax.set_xlabel("特征 1")
        self.ax.set_ylabel("特征 2")
        self.ax.grid(True)
        self._msg(f"K={k}，样本 {mat.shape[0]} 个，特征 {mat.shape[1]} 个。")
        self._msg(f"簇内平方和 inertia={km.inertia_:.4g}（越小越紧凑）。")
        counts = [int((labels == i).sum()) for i in range(k)]
        self._msg(f"各簇样本数：{counts}")

    def _classify(self, text, name):
        """通用分类器：决策树/随机森林/SVM/kNN。训练+测试评估 + 交叉验证 + 混淆矩阵 + ROC(二分类)。"""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        feats, labels = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            toks = line.replace(",", " ").split()
            if len(toks) < 2:
                continue
            feats.append([float(x) for x in toks[:-1]])
            labels.append(toks[-1])
        if len(feats) < 8:
            raise ValueError("样本太少（至少 8 行）。")
        X = np.array(feats)
        y = np.array(labels)
        if len(set(y)) < 2:
            raise ValueError("标签列至少需要 2 个类别。")
        depth = None
        if self.depth.get().strip():
            depth = int(float(self.depth.get()))
        if name == "决策树分类":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=depth, random_state=0)
        elif name == "随机森林分类":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=0)
        elif name == "支持向量机 SVM":
            from sklearn.svm import SVC
            model = SVC(kernel="rbf", probability=True, random_state=0)
        elif name == "kNN 最近邻":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=max(1, int(float(self.k.get() or 5))))
        else:
            raise ValueError("未知分类模型。")
        ratio = float(self.test_ratio.get() or 0.2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ratio, random_state=0)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = model.score(Xte, yte)
        self._msg(f"{name}：测试集准确率 acc={acc:.1%}（{Xte.shape[0]} 个样本）")
        cvf = int(float(self.cv_folds.get() or 0))
        if cvf and X.shape[0] > max(cvf, 6):
            try:
                scores = cross_val_score(model, X, y, cv=cvf, scoring="accuracy")
                self._msg(f"交叉验证（{cvf} 折）：acc={scores.mean():.1%} ± {scores.std():.1%}")
            except Exception as e:
                self._msg(f"交叉验证失败：{e}")
        if hasattr(model, "feature_importances_"):
            self._msg(f"特征重要性：{[f'{v:.3g}' for v in model.feature_importances_]}")
        self.figure.clear()
        classes = sorted(set(y))
        ncls = len(classes)
        if ncls == 2:
            ax1 = self.figure.add_subplot(1, 2, 1)
            self._confusion(ax1, yte, pred, classes)
            ax2 = self.figure.add_subplot(1, 2, 2)
            pos = classes[1]
            proba = model.predict_proba(Xte)
            fpr, tpr, _ = roc_curve((yte == pos), proba[:, list(classes).index(pos)])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, lw=2, color="crimson", label=f"AUC={roc_auc:.3g}")
            ax2.plot([0, 1], [0, 1], ls="--", color="gray")
            ax2.set_title("ROC 曲线")
            ax2.set_xlabel("假正率")
            ax2.set_ylabel("真正率")
            ax2.legend(fontsize=9)
            ax2.grid(True)
            self.figure.subplots_adjust(wspace=0.4)
            self._msg(f"二分类（关键类「{pos}」）ROC AUC={roc_auc:.3g}（越接近 1 越好）")
        else:
            self.ax = self.figure.add_subplot(111)
            self._confusion(self.ax, yte, pred, classes)

    def _confusion(self, ax, yte, pred, classes):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yte, pred, labels=classes)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("混淆矩阵")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        self.figure.colorbar(im, ax=ax)
        cmax = cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="k" if cm[i, j] < (cmax / 2 if cmax > 0 else 1) else "w")

    def _pca(self, text):
        """PCA 降维：投影到前 2 个主成分画散点，看数据的紧致结构。"""
        from sklearn.decomposition import PCA
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            toks = line.replace(",", " ").split()
            if len(toks) < 2:
                continue
            rows.append([float(x) for x in toks])
        X = np.array(rows)
        if X.shape[0] < 5:
            raise ValueError("样本太少（至少 5 行）。")
        n_comp = max(1, min(2, X.shape[0] - 1, X.shape[1]))
        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        yy = Z[:, 1] if Z.shape[1] > 1 else np.zeros_like(Z[:, 0])
        self.ax.scatter(Z[:, 0], yy, s=50, c="steelblue")
        for i in range(min(X.shape[0], 25)):
            self.ax.annotate(str(i + 1), (Z[i, 0], yy[i]), fontsize=7)
        self.ax.set_title(f"PCA 降维（前 {n_comp} 个主成分）")
        self.ax.set_xlabel("主成分 1")
        self.ax.set_ylabel("主成分 2" if Z.shape[1] > 1 else "主成分 1")
        self.ax.grid(True)
        self._msg(f"主成分 1 解释方差 {evr[0]:.1%}，主成分 2 解释 {evr[1] if len(evr) > 1 else 0:.1%}，合计 {evr.sum():.1%}")