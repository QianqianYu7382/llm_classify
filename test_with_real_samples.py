"""
测试：从真实训练数据中采样，看看模型能否正确分类
这样可以验证模型本身是否有问题
"""

import sys
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from lm_classifier import ClassConditionalLMClassifier
from synthetic_data import TOPIC_NAMES


def test_with_real_samples():
    """从真实数据中采样，测试模型性能"""
    
    buffer = StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    
    print("=" * 80)
    print("TEST: Using Real Training Data Samples")
    print("=" * 80)
    print("This test samples real data from training set to simulate synthetic data.")
    print("If n-gram performs well here, it proves the model itself works correctly.")
    print("=" * 80)
    print()
    
    # 加载数据
    print("[STEP 1] Loading data...")
    train_df = pd.read_csv("training_data.csv")
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    train_df = train_df[train_df["text"].str.strip() != ""]
    
    # 使用少量数据快速测试
    train_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"].tolist()
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print()
    
    # 训练 n-gram 模型
    print("[STEP 2] Training n-gram model...")
    ngram_clf = ClassConditionalLMClassifier(n=3, unk_threshold=1, alpha=0.5)
    ngram_clf.fit(X_train, y_train)
    print("  Model trained")
    print()
    
    # 测试 n-gram 模型在真实测试数据上的表现
    print("[STEP 3] Testing n-gram model on REAL test data...")
    ngram_acc = ngram_clf.score(X_test, y_test)
    print(f"  N-gram accuracy: {ngram_acc:.4f}")
    
    ngram_preds = ngram_clf.predict(X_test)
    print("\n  Classification report:")
    print(classification_report(y_test, ngram_preds, target_names=TOPIC_NAMES))
    print()
    
    # 从每个类别采样一些真实数据作为"合成数据"
    print("[STEP 4] Sampling real data from each class (simulating synthetic data)...")
    n_per_class = 50
    X_synthetic = []
    y_synthetic = []
    
    for label in range(4):
        class_texts = [text for text, lab in zip(X_train, y_train) if lab == label]
        if len(class_texts) >= n_per_class:
            sampled = pd.Series(class_texts).sample(n=n_per_class, random_state=42+label).tolist()
            X_synthetic.extend(sampled)
            y_synthetic.extend([label] * n_per_class)
    
    print(f"  Sampled {len(X_synthetic)} real texts (simulating synthetic data)")
    print()
    
    # 测试 n-gram 模型在这些"合成数据"上的表现
    print("[STEP 5] Testing n-gram model on sampled REAL data...")
    ngram_acc_syn = ngram_clf.score(X_synthetic, y_synthetic, use_uniform_prior=True)
    print(f"  N-gram accuracy (uniform prior): {ngram_acc_syn:.4f}")
    
    ngram_preds_syn = ngram_clf.predict(X_synthetic, use_uniform_prior=True)
    print("\n  Classification report:")
    print(classification_report(y_synthetic, ngram_preds_syn, target_names=TOPIC_NAMES))
    print()
    
    # 训练 embedding 模型
    print("[STEP 6] Training embedding model on sampled REAL data...")
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
        X_synthetic, y_synthetic,
        test_size=0.3,
        random_state=42,
        stratify=y_synthetic
    )
    
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_syn_train_emb = st_model.encode(X_syn_train, show_progress_bar=False)
    X_syn_test_emb = st_model.encode(X_syn_test, show_progress_bar=False)
    
    embedding_clf = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="multinomial")
    embedding_clf.fit(X_syn_train_emb, y_syn_train)
    
    embedding_preds = embedding_clf.predict(X_syn_test_emb)
    embedding_acc = accuracy_score(y_syn_test, embedding_preds)
    
    print(f"  Embedding accuracy: {embedding_acc:.4f}")
    print("\n  Classification report:")
    print(classification_report(y_syn_test, embedding_preds, target_names=TOPIC_NAMES))
    print()
    
    # 结果总结
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"N-gram on REAL test data:              {ngram_acc:.4f}")
    print(f"N-gram on sampled REAL data:           {ngram_acc_syn:.4f}")
    print(f"Embedding on sampled REAL test data:   {embedding_acc:.4f}")
    print()
    
    if ngram_acc_syn >= embedding_acc:
        print("✅ SUCCESS: N-gram performs better on real data samples!")
        print("   This suggests the model itself is fine.")
        print("   The problem is likely with the text generation method.")
    else:
        print("⚠️  Even with real data, n-gram doesn't outperform embedding.")
    print("=" * 80)
    
    # Restore stdout and save results
    sys.stdout = sys_stdout
    output_text = buffer.getvalue()
    
    # Save to file
    output_path = "real_samples_test_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    
    print(f"\n[INFO] Results saved to {output_path}\n")
    print(output_text)


if __name__ == "__main__":
    test_with_real_samples()

