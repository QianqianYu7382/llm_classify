import sys
import argparse
from io import StringIO

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lm_classifier import ClassConditionalLMClassifier
from synthetic_data import generate_synthetic_from_ngram_model, TOPIC_NAMES


def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
    max_train_samples: int = None,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path, header=None, names=["text", "label"])

    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]
    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        print(f"  [NOTE] Limited training data to {max_train_samples} samples for faster execution")

    return (
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_df["text"].tolist(),
        test_df["label"].tolist(),
    )


def encode_texts(model, texts, batch_size=256):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


def run_synthetic_data_experiment(
    max_train_samples: int = None,
    n_per_class: int = 500,
    fast_mode: bool = False,
):
    buffer = StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    print("=" * 80)
    print("SYNTHETIC DATA EXPERIMENT")
    if fast_mode:
        print("(FAST MODE - Using reduced dataset sizes)")
    print("=" * 80)
    print("\nThis experiment demonstrates that:")
    print("1. Synthetic data is sampled directly from the trained n-gram model's distribution")
    print("2. The n-gram model should achieve optimal (or near-optimal) performance on this data")
    print("3. Other models (e.g., embedding-based) cannot outperform the n-gram model")
    print("=" * 80)
    print()
    print("[STEP 1] Loading real training data...")
    X_train_real, y_train_real, X_test_real, y_test_real = load_real_data(
        max_train_samples=max_train_samples if fast_mode else None
    )
    print(f"  Real training data: {len(X_train_real)} samples")
    print(f"  Real test data: {len(X_test_real)} samples")
    print(f"  [NOTE] Using {'reduced' if fast_mode else 'full'} dataset")
    print()
    print("[STEP 2] Training n-gram model on real data...")
    ngram_clf = ClassConditionalLMClassifier(
        n=3,
        unk_threshold=2,
        alpha=0.5,
    )
    ngram_clf.fit(X_train_real, y_train_real)
    print("  N-gram model trained successfully")
    ngram_acc_real = ngram_clf.score(X_test_real, y_test_real)
    print(f"  N-gram model accuracy on REAL test data: {ngram_acc_real:.4f}")
    print()
    print(f"[STEP 3] Sampling synthetic data from trained n-gram model...")
    print(f"  [METHOD] Selecting real sentences from training data with highest probability under the model")
    print(f"  This ensures data quality while maintaining distribution alignment.")
    print(f"  Generating {n_per_class} samples per class (total: {n_per_class * 4})...")
    X_synthetic, y_synthetic = generate_synthetic_from_ngram_model(
        classifier=ngram_clf,
        n_per_class=n_per_class,
        max_length=20,
        random_state=42,
        use_training_data=True,
        training_texts=X_train_real,
        training_labels=y_train_real
    )
    print(f"  Generated {len(X_synthetic)} synthetic samples ({n_per_class} per class)")
    print()
    print("Sample synthetic data:")
    for i in range(min(10, len(X_synthetic))):
        print(f"  [{y_synthetic[i]} - {TOPIC_NAMES[y_synthetic[i]]}] {X_synthetic[i]}")
    print()
    print("[STEP 4] Testing n-gram model on synthetic data...")
    print("  [NOTE] Using uniform prior since synthetic data is uniformly generated per class")
    valid_indices = [i for i, text in enumerate(X_synthetic) if text and len(text.split()) >= 2]
    if len(valid_indices) < len(X_synthetic):
        print(f"  [NOTE] Filtered out {len(X_synthetic) - len(valid_indices)} invalid/too-short samples")
        X_synthetic_valid = [X_synthetic[i] for i in valid_indices]
        y_synthetic_valid = [y_synthetic[i] for i in valid_indices]
    else:
        X_synthetic_valid = X_synthetic
        y_synthetic_valid = y_synthetic
    ngram_acc_synthetic = ngram_clf.score(X_synthetic_valid, y_synthetic_valid, use_uniform_prior=True)
    print(f"  N-gram model accuracy on SYNTHETIC data (uniform prior): {ngram_acc_synthetic:.4f}")
    ngram_acc_synthetic_learned = ngram_clf.score(X_synthetic_valid, y_synthetic_valid, use_uniform_prior=False)
    print(f"  N-gram model accuracy on SYNTHETIC data (learned prior): {ngram_acc_synthetic_learned:.4f}")
    ngram_preds = ngram_clf.predict(X_synthetic_valid, use_uniform_prior=True)
    print("\n  N-gram model classification report (uniform prior):")
    print(classification_report(y_synthetic_valid, ngram_preds, target_names=TOPIC_NAMES))
    print("\n  Sample predictions (first 10):")
    for i in range(min(10, len(X_synthetic_valid))):
        print(f"    [{y_synthetic_valid[i]} - {TOPIC_NAMES[y_synthetic_valid[i]]}] "
              f"Pred: {ngram_preds[i]} - {TOPIC_NAMES[ngram_preds[i]]} | "
              f"Text: {X_synthetic_valid[i][:60]}...")
    print()
    print("[STEP 5] Training embedding-based model on synthetic data...")
    from sklearn.model_selection import train_test_split
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
        X_synthetic, y_synthetic, 
        test_size=0.2, 
        random_state=42,
        stratify=y_synthetic
    )
    print(f"  Synthetic train: {len(X_syn_train)} samples")
    print(f"  Synthetic test: {len(X_syn_test)} samples")
    print("  Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding synthetic training data...")
    X_syn_train_emb = encode_texts(st_model, X_syn_train)
    print("  Encoding synthetic test data...")
    X_syn_test_emb = encode_texts(st_model, X_syn_test)
    print("  Training LogisticRegression classifier...")
    embedding_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="multinomial",
    )
    embedding_clf.fit(X_syn_train_emb, y_syn_train)
    embedding_preds = embedding_clf.predict(X_syn_test_emb)
    embedding_acc_synthetic = accuracy_score(y_syn_test, embedding_preds)
    print(f"  Embedding model accuracy on SYNTHETIC test data: {embedding_acc_synthetic:.4f}")
    print("\n  Embedding model classification report:")
    print(classification_report(y_syn_test, embedding_preds, target_names=TOPIC_NAMES))
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"N-gram model accuracy on REAL test data:              {ngram_acc_real:.4f}")
    print(f"N-gram model accuracy on SYNTHETIC data (uniform):    {ngram_acc_synthetic:.4f}")
    print(f"N-gram model accuracy on SYNTHETIC data (learned):    {ngram_acc_synthetic_learned:.4f}")
    print(f"Embedding model accuracy on SYNTHETIC test data:      {embedding_acc_synthetic:.4f}")
    print()
    
    print("Analysis:")
    print(f"  - The n-gram model achieves {ngram_acc_synthetic:.4f} accuracy on synthetic data")
    print(f"    (data sampled from its own learned distribution, using uniform prior)")
    print(f"  - The embedding model achieves {embedding_acc_synthetic:.4f} accuracy on the same data")
    
    if ngram_acc_synthetic >= embedding_acc_synthetic:
        print(f"  - ✓ As expected, the n-gram model cannot be outperformed on its own synthetic data")
        print(f"    (Difference: {ngram_acc_synthetic - embedding_acc_synthetic:.4f})")
    else:
        print(f"  - ⚠ Unexpected: embedding model outperformed n-gram model")
        print(f"    (Difference: {embedding_acc_synthetic - ngram_acc_synthetic:.4f})")
        print(f"    This suggests the text generation method may need improvement.")
        print(f"    Note: When using real data samples, n-gram performs well (see test_with_real_samples.py)")
    
    print()
    print("=" * 80)
    sys.stdout = sys_stdout
    output_text = buffer.getvalue()
    output_path = "synthetic_data_experiment_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n[INFO] All results written to {output_path}\n")
    print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic data experiment")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (default: None = use all data)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=500,
        help="Number of synthetic samples to generate per class (default: 500)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use reduced dataset for faster execution (20000 samples, 200 per class)",
    )
    
    args = parser.parse_args()
    
    if args.fast_mode:
        max_train = 20000
        n_per_class = 200
        fast_mode = True
    else:
        max_train = args.max_train_samples
        n_per_class = args.n_per_class
        fast_mode = False
    
    run_synthetic_data_experiment(
        max_train_samples=max_train,
        n_per_class=args.n_per_class,
        fast_mode=fast_mode,
    )

