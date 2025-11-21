"""
real_data_embedding_experiment.py (with TXT output)

使用预训练的 SentenceTransformer 语言模型 (all-MiniLM-L6-v2)
对真实新闻数据做分类，并将结果写入 embedding_results.txt
"""

import pandas as pd
import numpy as np
import sys
from io import StringIO

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 真实数据类别名称
TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# -----------------------------
# 1. 加载数据
# -----------------------------
def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path, header=None, names=["text", "label"])

    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]

    return (
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_df["text"].tolist(),
        test_df["label"].tolist(),
    )


# -----------------------------
# 2. 编码文本
# -----------------------------
def encode_texts(model, texts, batch_size=256):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


# -----------------------------
# 3. 主流程：增加 TXT 输出
# -----------------------------
def run_embedding_experiment():

    # 创建一个 buffer 捕获所有 print 输出
    buffer = StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer  # redirect print

    print("[INFO] Loading data...")

    X_train, y_train, X_test, y_test = load_real_data()

    print("[INFO] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[INFO] Encoding training data...")
    X_train_emb = encode_texts(st_model, X_train)

    print("[INFO] Encoding test data...")
    X_test_emb = encode_texts(st_model, X_test)

    print(f"[INFO] Embedding shape: train {X_train_emb.shape}, test {X_test_emb.shape}")

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="multinomial",
    )

    print("[INFO] Training classifier...")
    clf.fit(X_train_emb, y_train)

    print("[INFO] Evaluating on test set...")
    y_pred = clf.predict(X_test_emb)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy on REAL test set = {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=TOPIC_NAMES))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSample predictions (first 10):")
    for i in range(min(10, len(X_test))):
        print("-" * 70)
        print("TEXT       :", X_test[i])
        print("True label :", y_test[i], "-", TOPIC_NAMES[y_test[i]])
        print("Pred label :", y_pred[i], "-", TOPIC_NAMES[y_pred[i]])

    # 恢复 print 输出
    sys.stdout = sys_stdout

    # 写入文件
    output_text = buffer.getvalue()
    output_path = "embedding_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n[INFO] All results written to {output_path}\n")


# -----------------------------
# 主入口
# -----------------------------
if __name__ == "__main__":
    run_embedding_experiment()
