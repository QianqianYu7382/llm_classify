"""
real_data_embedding_experiment.py

使用预训练的 SentenceTransformer 语言模型 (all-MiniLM-L6-v2)
对真实新闻数据 (training_data.csv / test_data.csv) 做主题分类。

流程：
  1. 读取 train / test 数据
  2. 使用预训练 LM 将每条文本编码为句向量 embedding
  3. 使用 Logistic Regression 在 embedding 上训练分类器
  4. 在 test 集上评估准确率，并打印分类报告

这是一个典型的 modern language-model-based classifier：
  - 语言模型负责生成上下文语义表示
  - 分类器只是一个线性决策层
"""

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 真实数据的类别名称（和之前保持一致）
TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# -----------------------------
# 1. 读取真实数据
# -----------------------------

def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
):
    """
    读取 training_data.csv / test_data.csv

    training_data.csv: 有 header, columns = ['text', 'label']
    test_data.csv    : 没有 header，我们强制设为 ['text', 'label']
    """
    # 训练集
    train_df = pd.read_csv(train_path)
    if not {"text", "label"} <= set(train_df.columns):
        raise ValueError(
            f"training_data.csv should have columns ['text', 'label'], "
            f"but got {list(train_df.columns)}"
        )
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)

    # 测试集（之前看到你那份是无 header 的）
    test_df = pd.read_csv(test_path, header=None, names=["text", "label"])
    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    # 去掉空文本
    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print("[INFO] Loaded real data for embedding-based experiment:")
    print(f"  Train size = {len(X_train)}")
    print(f"  Test  size = {len(X_test)}")
    print(f"  Train label distribution:\n{train_df['label'].value_counts()}")
    print(f"  Test  label distribution:\n{test_df['label'].value_counts()}\n")

    return X_train, y_train, X_test, y_test


# -----------------------------
# 2. 使用 SentenceTransformer 编码文本
# -----------------------------

def encode_texts(
    model: SentenceTransformer,
    texts,
    batch_size: int = 256,
):
    """
    将一组文本编码成 embedding 矩阵 (numpy array)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings


# -----------------------------
# 3. 在 embedding 上训练 Logistic Regression
# -----------------------------

def run_embedding_experiment():
    # 1) 加载数据
    X_train, y_train, X_test, y_test = load_real_data()

    # 2) 加载预训练句向量模型
    print("[INFO] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3) 编码文本为 embedding
    print("[INFO] Encoding training texts to embeddings...")
    X_train_emb = encode_texts(st_model, X_train)
    print("[INFO] Encoding test texts to embeddings...")
    X_test_emb = encode_texts(st_model, X_test)

    X_train_emb = np.array(X_train_emb)
    X_test_emb = np.array(X_test_emb)
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    print(f"[INFO] Embedding shape: train {X_train_emb.shape}, test {X_test_emb.shape}")

    # 4) 训练 Logistic Regression 分类器
    print("[INFO] Training Logistic Regression classifier on embeddings...")
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="multinomial",
    )
    clf.fit(X_train_emb, y_train_arr)

    # 5) 在 test 集上评估
    y_pred = clf.predict(X_test_emb)
    acc = accuracy_score(y_test_arr, y_pred)
    print(f"\n[RESULT] Embedding-based LM classifier accuracy on REAL test set: {acc:.4f}\n")

    # 6) 分类报告 & 混淆矩阵
    print("Classification report:")
    print(classification_report(y_test_arr, y_pred, target_names=TOPIC_NAMES))

    print("Confusion matrix:")
    print(confusion_matrix(y_test_arr, y_pred))

    # 7) 打印一些示例，方便你写 qualitative 分析
    print("\nSample predictions on real test data (embedding-based):")
    num_examples = min(10, len(X_test))
    for i in range(num_examples):
        text = X_test[i]
        true_label = y_test[i]
        pred_label = y_pred[i]
        print("-" * 80)
        print("TEXT       :", text)
        print("True label :", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred label :", pred_label, "-", TOPIC_NAMES[pred_label])


# -----------------------------
# 主入口
# -----------------------------

if __name__ == "__main__":
    run_embedding_experiment()
