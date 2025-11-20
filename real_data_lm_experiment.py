"""
real_data_lm_experiment.py

在真实数据 (training_data.csv / test_data.csv) 上
使用基于 trigram 语言模型的 ClassConditionalLMClassifier 做新闻主题分类。

要求：和 lm_classifier.py 放在同一目录。
"""

import pandas as pd
from typing import Tuple, List

# 从你之前的文件中复用语言模型分类器
from lm_classifier import ClassConditionalLMClassifier

# 真实数据的类别名字（AG News 风格）
TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# -----------------------------
# 1. 读入并清洗真实数据
# -----------------------------

def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    读取真实数据：
      - training_data.csv：有 header，列名为 'text', 'label'
      - test_data.csv：无 header，需要手动指定列名

    返回：
      X_train, y_train, X_test, y_test
    """
    # 1) 训练数据：已经是干净的 text + label
    train_df = pd.read_csv(train_path)

    # 做一些简单的 sanity check
    if not {"text", "label"} <= set(train_df.columns):
        raise ValueError(
            f"training_data.csv should have columns ['text', 'label'], "
            f"but got {list(train_df.columns)}"
        )

    # 确保类型正确
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)

    # 2) 测试数据：你的文件里第一行被当成列名了，需要重新读
    # 原来的列名是类似一整句新闻标题，所以我们强制 header=None。
    test_df = pd.read_csv(test_path, header=None, names=["text", "label"])

    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    # 3) 去掉可能的空文本行
    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"[INFO] Loaded real data:")
    print(f"  Train size = {len(X_train)}")
    print(f"  Test  size = {len(X_test)}")
    print(f"  Train label distribution:\n{train_df['label'].value_counts()}")
    print(f"  Test  label distribution:\n{test_df['label'].value_counts()}\n")

    return X_train, y_train, X_test, y_test


# -----------------------------
# 2. 在真实数据上训练 + 测试 LM 分类器
# -----------------------------

def run_real_data_experiment():
    # 读数据
    X_train, y_train, X_test, y_test = load_real_data(
        train_path="training_data.csv",
        test_path="test_data.csv",
    )

    # 定义基于 trigram 的 LM 分类器
    clf = ClassConditionalLMClassifier(
        n=3,           # trigram
        unk_threshold=2,  # 稍微提高一点 unk_threshold，减少稀疏
        alpha=0.5,     # 平滑系数，可以调
    )

    print("[INFO] Fitting trigram LM classifier on real training data...")
    clf.fit(X_train, y_train)

    # 计算整体准确率
    acc = clf.score(X_test, y_test)
    print(f"\n[RESULT] LM classifier accuracy on REAL test set: {acc:.4f}\n")

    # 打印若干预测示例，方便 qualitative 分析
    print("Sample predictions on real test data:")
    num_examples = 10
    for i in range(num_examples):
        text = X_test[i]
        true_label = y_test[i]
        pred_label = clf.predict([text])[0]
        print("-" * 80)
        print("TEXT       :", text)
        print("True label :", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred label :", pred_label, "-", TOPIC_NAMES[pred_label])

    # （可选）如果你安装了 scikit-learn，可以输出更详细的分类报告
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        import numpy as np
        y_pred = clf.predict(X_test)
        print("\nClassification report (per-class precision/recall/F1):")
        print(classification_report(y_test, y_pred, target_names=TOPIC_NAMES))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
    except ImportError:
        print("\n[NOTE] scikit-learn is not installed, so no classification_report/confusion_matrix.")
        print("       If you want them, please run: pip install scikit-learn")


# -----------------------------
# 3. 主入口
# -----------------------------

if __name__ == "__main__":
    run_real_data_experiment()
