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
from modules import ui_kit as ui
from modules.i18n import tr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OPS = ["KMeans 聚类", "决策树分类", "随机森林分类", "支持向量机 SVM", "kNN 最近邻", "PCA 降维",
        "特征重要性（可解释性）", "模型对比", "交叉验证对比"]

COMPARE_MODELS = ["决策树分类", "随机森林分类", "支持向量机 SVM", "kNN 最近邻", "逻辑回归", "GBDT 提升树"]

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


class MlPage(ui.BasePage):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数 =================
        left = self.left
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text=tr("数据（每行一个样本；决策树时最后一列是类别）"), font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        self.data = ctk.CTkTextbox(left, height=180, font=ctk.CTkFont(family="Consolas", size=12))
        self.data.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.data.insert("1.0", SAMPLE)

        ctk.CTkLabel(left, text=tr("操作"), font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 2))
        self.op = ctk.CTkOptionMenu(left, values=OPS)
        self.op.grid(row=3, column=0, sticky="ew", padx=12)
        self.op.set(OPS[0])

        ctk.CTkLabel(left, text=tr("聚类数 K（KMeans 用）"), font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="w", padx=14, pady=(8, 2))
        self.k = ctk.CTkEntry(left, height=32)
        self.k.grid(row=5, column=0, sticky="ew", padx=12)
        self.k.insert(0, "2")

        ctk.CTkLabel(left, text=tr("决策树最大深度（决策树用，可空=不限）"), font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, sticky="w", padx=14, pady=(8, 2))
        self.depth = ctk.CTkEntry(left, height=32)
        self.depth.grid(row=7, column=0, sticky="ew", padx=12)
        self.depth.insert(0, "")

        ctk.CTkLabel(left, text=tr("测试集比例（分类用，默认 0.2）"), font=ctk.CTkFont(size=12)).grid(
            row=8, column=0, sticky="w", padx=14, pady=(8, 2))
        self.test_ratio = ctk.CTkEntry(left, height=32)
        self.test_ratio.grid(row=9, column=0, sticky="ew", padx=12)
        self.test_ratio.insert(0, "0.2")

        ctk.CTkLabel(left, text=tr("交叉验证折数 CV（0=关闭，分类用）"), font=ctk.CTkFont(size=12)).grid(
            row=10, column=0, sticky="w", padx=14, pady=(8, 2))
        self.cv_folds = ctk.CTkEntry(left, height=32)
        self.cv_folds.grid(row=11, column=0, sticky="ew", padx=12)
        self.cv_folds.insert(0, "5")

        ctk.CTkButton(left, text=tr("⚡ 训 练"), height=42, command=self._run).grid(
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

        # 可解释模型（特征重要性/模型对比/交叉验证用）
        ctk.CTkLabel(left, text=tr("可解释模型（特征重要性/对比/交叉验证用）"), font=ctk.CTkFont(size=12)).grid(
            row=14, column=0, sticky="w", padx=14, pady=(4, 1))
        self.model_dd = ctk.CTkOptionMenu(left, values=COMPARE_MODELS)
        self.model_dd.grid(row=15, column=0, sticky="ew", padx=14)
        self.model_dd.set("随机森林分类")

        # ================= 右：结果 + 画布 =================
        right = self.right
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(right, text=tr("训练结果"), font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13), height=300)
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))
        box, self.figure, self.canvas, _tb = self.show_plot()
        self.ax = self.figure.gca()
        self.ax.set_title("机器学习图形")

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
            elif op == OPS[6]:
                self._feature_importance(self.data.get("1.0", "end"))
            elif op == OPS[7]:
                self._model_compare(self.data.get("1.0", "end"))
            elif op == OPS[8]:
                self._cv_compare(self.data.get("1.0", "end"))
        except Exception as e:
            self._msg(f"出错：{e}")
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("机器学习", op, self.out.get("1.0", "end")[:200])

    def _kmeans(self, mat):
        from sklearn.cluster import KMeans
        k = max(2, int(float(self.k.get() or 2)))
        k = min(k, mat.shape[0])
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(mat)
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
            model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0, n_jobs=-1)
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
                scores = cross_val_score(model, X, y, cv=cvf, scoring="accuracy", n_jobs=-1)
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
    # ================= v1.9.0：可解释机器学习 =================
    def _parse_xy(self, text):
        """解析特征(除最后一列)与标签(最后一列)。"""
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
        if len(feats) < 1450:
            raise ValueError("样本太少（至少 6 行）。")
        X = np.array(feats)
        y = np.array(labels)
        if len(set(y)) < 2:
            raise ValueError("标签列至少需要 2 个类别。")
        return X, y

    def _depth_opt(self):
        try:
            return int(float(self.depth.get())) if self.depth.get().strip() else None
        except Exception:
            return None

    def _k_opt(self):
        try:
            return float(self.k.get()) if self.k.get().strip() else 5
        except Exception:
            return 511

    def _make_model(self, name, depth=None, k=None):
        """按名称构造可训练的分类器，返回 (model, 短名)。"""
        if name == "决策树分类":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(max_depth=depth, random_state=0), "决策树"
        if name == "随机森林分类":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0, n_jobs=-1), "随机森林"
        if name == "支持向量机 SVM":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", probability=True, random_state=0), "SVM"
        if name == "kNN 最近邻":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(n_neighbors=max(1, int(k or 5))), "kNN"
        if name == "逻辑回归":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=0), "逻辑回归"
        if name == "GBDT 提升树":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=0), "GBDT"
        raise ValueError("未知模型：" + name)

    def _feature_importance(self, text):
        from sklearn.inspection import permutation_importance
        X, y = self._parse_xy(text)
        name = self.model_dd.get()
        model, short = self._make_model(name, self._depth_opt(), self._k_opt())
        model.fit(X, y)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            std = None
            method = "树模型自带重要性（不纯度增益）"
        else:
            perm = permutation_importance(model, X, y, n_repeats=15, random_state=0,
                                          scoring="accuracy", n_jobs=-1)
            importances = perm.importances_mean
            std = perm.importances_std
            method = "排列重要性（Permutation，打乱特征看精度下降）"
        order = np.argsort(-importances)
        self._msg(f"模型：{short}；特征数：{X.shape[1]}；方法：{method}")
        for i in order:
            if std is not None:
                self._msg(f"  特征 {i + 1}：{importances[i]:.4f} ± {std[i]:.4f}")
            else:
                self._msg(f"  特征 {i + 1}：{importances[i]:.4f}")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        names = [f"F{i + 1}" for i in order]
        self.ax.barh(names, importances[order], color="steelblue")
        if std is not None:
            self.ax.errorbar(importances[order], range(len(order)), xerr=std[order],
                             fmt="none", ecolor="k", capsize=2)
        self.ax.set_xlabel("特征重要性")
        self.ax.set_title(f"{short} 特征重要性（可解释性）")
        self.ax.grid(True, axis="x", alpha=0.4)
        self.ax.invert_yaxis()

    def _model_compare(self, text):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        X, y = self._parse_xy(text)
        ratio = float(self.test_ratio.get() or 0.2)
        ratio = max(0.1, min(0.4, ratio))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ratio, random_state=0)
        classes = sorted(set(y))
        rows = []
        for name in COMPARE_MODELS:
            try:
                model, short = self._make_model(name, self._depth_opt(), self._k_opt())
                model.fit(Xtr, ytr)
                acc = float(model.score(Xte, yte))
                auc = float("nan")
                if len(classes) == 2 and hasattr(model, "predict_proba"):
                    pos = classes[1]
                    proba = model.predict_proba(Xte)
                    auc = float(roc_auc_score((yte == pos), proba[:, list(classes).index(pos)]))
                rows.append((short, acc, auc, name))
            except Exception as e:
                rows.append((short, float("nan"), float("nan"), name))
        self._msg("模型对比（同一训练/测试切分）：")
        for short, acc, auc, name in rows:
            acc_t = f"{acc:.1%}" if not np.isnan(acc) else "失败"
            auc_t = f"AUC={auc:.3f}" if not np.isnan(auc) else "AUC=—"
            self._msg(f"  {short}：acc={acc_t}，{auc_t}")
        best = max((r for r in rows if not np.isnan(r[1])), key=lambda r: r[1], default=None)
        if best:
            self._msg(f"\n最佳：{best[0]}（acc={best[1]:.1%}）")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        names = [r[0] for r in rows]
        accs = [r[1] for r in rows]
        self.ax.bar(names, accs, color=["#4C72B0" if not np.isnan(a) else "#cccccc" for a in accs])
        for i, a in enumerate(accs):
            if not np.isnan(a):
                self.ax.text(i, a + 0.005, f"{a:.2f}", ha="center", fontsize=7)
        self.ax.set_ylim(0, 1.05)
        self.ax.set_ylabel("测试集准确率")
        self.ax.set_title("各模型准确率对比")
        self.ax.grid(True, axis="y", alpha=0.3)

    def _cv_compare(self, text):
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        X, y = self._parse_xy(text)
        depth = self._depth_opt()
        fold = int(float(self.cv_folds.get() or 5))
        fold = max(2, min(fold, max(2, X.shape[0] // 2)))
        rows = []
        for name in COMPARE_MODELS:
            try:
                model, short = self._make_model(name, depth, self._k_opt())
                cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=0)
                sc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                rows.append((short, float(sc.mean()), float(sc.std()), sc))
            except Exception:
                pass
        self._msg(f"交叉验证对比（{fold} 折，Stratified）：")
        for short, mean, sd, sc in rows:
            self._msg(f"  {short}：acc={mean:.1%} ± {sd:.1%}")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        names = [r[0] for r in rows]
        data = [r[3] for r in rows]
        bp = self.ax.boxplot(data, labels=names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ecae1")
        self.ax.set_ylabel("交叉验证准确率")
        self.ax.set_title(f"各模型 {fold} 折交叉验证准确率分布")
        self.ax.grid(True, axis="y", alpha=0.3)
