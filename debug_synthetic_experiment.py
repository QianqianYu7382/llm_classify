"""
快速调试版本：在小数据集上验证 n-gram 模型在自己生成的数据上表现更好
"""

import sys
from io import StringIO

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from lm_classifier import ClassConditionalLMClassifier
from synthetic_data import TOPIC_NAMES


def load_real_data(train_path: str = "training_data.csv", max_samples: int = 5000):
    """加载少量真实训练数据用于快速测试"""
    train_df = pd.read_csv(train_path)
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    train_df = train_df[train_df["text"].str.strip() != ""]
    
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    return train_df["text"].tolist(), train_df["label"].tolist()


def debug_experiment():
    """调试实验：找出为什么n-gram在自己生成的数据上表现差"""
    
    print("=" * 80)
    print("DEBUG EXPERIMENT: Small Dataset")
    print("=" * 80)
    print()
    
    # Step 1: 加载少量真实数据
    print("[STEP 1] Loading small real dataset...")
    X_train, y_train = load_real_data(max_samples=5000)
    print(f"  Loaded {len(X_train)} training samples")
    print(f"  Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print()
    
    # Step 2: 训练 n-gram 模型
    print("[STEP 2] Training n-gram model...")
    ngram_clf = ClassConditionalLMClassifier(n=3, unk_threshold=1, alpha=0.5)
    ngram_clf.fit(X_train, y_train)
    print("  Model trained successfully")
    print()
    
    # Step 3: 生成少量合成数据
    print("[STEP 3] Generating synthetic data (50 per class)...")
    n_per_class = 50
    X_synthetic, y_synthetic = ngram_clf.sample_synthetic_data(
        n_per_class=n_per_class,
        max_length=15,  # 较短的长度
        random_state=42,
        show_progress=True
    )
    print(f"  Generated {len(X_synthetic)} synthetic samples")
    print()
    
    # 显示一些生成的样本
    print("Sample synthetic data:")
    for i in range(min(10, len(X_synthetic))):
        print(f"  [{y_synthetic[i]} - {TOPIC_NAMES[y_synthetic[i]]}] {X_synthetic[i]}")
    print()
    
    # Step 4: 调试 - 检查生成的文本质量
    print("[STEP 4] Analyzing synthetic data quality...")
    avg_length = np.mean([len(text.split()) for text in X_synthetic])
    unk_count = sum(1 for text in X_synthetic if "<UNK>" in text)
    print(f"  Average text length: {avg_length:.2f} words")
    print(f"  Texts with <UNK>: {unk_count}/{len(X_synthetic)} ({100*unk_count/len(X_synthetic):.1f}%)")
    print()
    
    # Step 5: 测试 n-gram 模型 - 详细调试
    print("[STEP 5] Testing n-gram model on synthetic data (with debugging)...")
    
    # 过滤无效样本
    valid_indices = [i for i, text in enumerate(X_synthetic) if text and len(text.split()) >= 2]
    X_syn_valid = [X_synthetic[i] for i in valid_indices]
    y_syn_valid = [y_synthetic[i] for i in valid_indices]
    
    print(f"  Valid samples: {len(X_syn_valid)}/{len(X_synthetic)}")
    
    # 详细检查前几个样本的预测
    print("\n  Detailed prediction analysis (first 5 samples):")
    for i in range(min(5, len(X_syn_valid))):
        text = X_syn_valid[i]
        true_label = y_syn_valid[i]
        
        # 计算每个类别的得分（使用均匀先验）
        scores = {}
        uniform_prior = np.log(1.0 / len(ngram_clf.classes_))
        for y in ngram_clf.classes_:
            lm = ngram_clf.class_lms[y]
            log_px_given_y = lm.sentence_log_prob(text)
            scores[y] = log_px_given_y + uniform_prior
        
        pred_label = max(scores, key=scores.get)
        
        print(f"\n    Sample {i+1}:")
        print(f"      Text: {text[:60]}...")
        print(f"      True label: {true_label} ({TOPIC_NAMES[true_label]})")
        print(f"      Predicted: {pred_label} ({TOPIC_NAMES[pred_label]})")
        print(f"      Scores:")
        for y, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            marker = "✓" if y == true_label else " " if y == pred_label else " "
            print(f"        {marker} Class {y} ({TOPIC_NAMES[y]}): {score:.4f}")
    
    # 计算准确率 - 使用均匀先验（因为合成数据是均匀生成的）
    print("\n  Testing with uniform prior (since synthetic data is uniformly generated)...")
    ngram_preds = ngram_clf.predict(X_syn_valid, use_uniform_prior=True)
    ngram_acc = accuracy_score(y_syn_valid, ngram_preds)
    print(f"  N-gram model accuracy (uniform prior): {ngram_acc:.4f}")
    
    # 也测试使用学习到的先验
    ngram_preds_learned = ngram_clf.predict(X_syn_valid, use_uniform_prior=False)
    ngram_acc_learned = accuracy_score(y_syn_valid, ngram_preds_learned)
    print(f"  N-gram model accuracy (learned prior): {ngram_acc_learned:.4f}")
    print()
    
    # Step 6: 训练并测试 embedding 模型
    print("[STEP 6] Training embedding model on synthetic data...")
    
    # 划分训练/测试集
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
        X_syn_valid, y_syn_valid,
        test_size=0.3,
        random_state=42,
        stratify=y_syn_valid
    )
    
    print(f"  Train: {len(X_syn_train)}, Test: {len(X_syn_test)}")
    
    # 训练 embedding 模型
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_syn_train_emb = st_model.encode(X_syn_train, show_progress_bar=False)
    X_syn_test_emb = st_model.encode(X_syn_test, show_progress_bar=False)
    
    embedding_clf = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="multinomial")
    embedding_clf.fit(X_syn_train_emb, y_syn_train)
    
    embedding_preds = embedding_clf.predict(X_syn_test_emb)
    embedding_acc = accuracy_score(y_syn_test, embedding_preds)
    
    print(f"  Embedding model accuracy: {embedding_acc:.4f}")
    print()
    
    # Step 7: 结果总结
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"N-gram model accuracy (uniform prior):          {ngram_acc:.4f}")
    print(f"N-gram model accuracy (learned prior):          {ngram_acc_learned:.4f}")
    print(f"Embedding model accuracy on SYNTHETIC test data: {embedding_acc:.4f}")
    print(f"Difference (uniform prior): {ngram_acc - embedding_acc:.4f}")
    print()
    
    if ngram_acc >= embedding_acc:
        print("✅ SUCCESS: N-gram model performs better on its own synthetic data!")
    else:
        print("⚠️  ISSUE: N-gram model does NOT perform better on its own synthetic data")
        print("   This suggests a problem with sampling or prediction logic")
    print("=" * 80)


if __name__ == "__main__":
    debug_experiment()

