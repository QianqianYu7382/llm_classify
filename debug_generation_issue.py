"""
深入调试：找出为什么N-gram无法识别自己生成的数据
"""

import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lm_classifier import ClassConditionalLMClassifier
from synthetic_data import TOPIC_NAMES


def debug_generation():
    """调试生成问题"""
    
    print("=" * 80)
    print("DEBUG: Why N-gram can't recognize its own generated data?")
    print("=" * 80)
    print()
    
    # Load small dataset
    train_df = pd.read_csv("training_data.csv")
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    train_df = train_df[train_df["text"].str.strip() != ""]
    train_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"].tolist()
    )
    
    # Train model
    print("[STEP 1] Training n-gram model...")
    ngram_clf = ClassConditionalLMClassifier(n=3, unk_threshold=1, alpha=0.5)
    ngram_clf.fit(X_train, y_train)
    print("  Model trained")
    print()
    
    # Generate some synthetic data
    print("[STEP 2] Generating synthetic data...")
    n_per_class = 20
    X_synthetic, y_synthetic = ngram_clf.sample_synthetic_data(
        n_per_class=n_per_class,
        max_length=15,
        random_state=42,
        show_progress=False
    )
    
    # Filter valid
    valid_indices = [i for i, text in enumerate(X_synthetic) if text and len(text.split()) >= 2]
    X_syn_valid = [X_synthetic[i] for i in valid_indices]
    y_syn_valid = [y_synthetic[i] for i in valid_indices]
    
    print(f"  Generated {len(X_syn_valid)} valid samples")
    print()
    
    # Detailed analysis of first 10 samples
    print("[STEP 3] Detailed analysis of generated samples...")
    print()
    
    for i in range(min(10, len(X_syn_valid))):
        text = X_syn_valid[i]
        true_label = y_syn_valid[i]
        
        print(f"Sample {i+1}:")
        print(f"  Text: {text}")
        print(f"  True label: {true_label} ({TOPIC_NAMES[true_label]})")
        
        # Calculate scores for all classes
        scores = {}
        uniform_prior = -1.3862943611198906  # log(1/4)
        
        for y in ngram_clf.classes_:
            lm = ngram_clf.class_lms[y]
            log_px_given_y = lm.sentence_log_prob(text)
            scores[y] = log_px_given_y + uniform_prior
        
        pred_label = max(scores, key=scores.get)
        
        print(f"  Predicted: {pred_label} ({TOPIC_NAMES[pred_label]})")
        print(f"  Scores (log P(x|y) + log P(y)):")
        for y in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"    Class {y[0]} ({TOPIC_NAMES[y[0]]}): {y[1]:.4f}")
        
        # Check if the true class model gives higher probability
        true_lm = ngram_clf.class_lms[true_label]
        true_log_prob = true_lm.sentence_log_prob(text)
        print(f"  True class log P(x|y={true_label}): {true_log_prob:.4f}")
        
        # Compare with a real text from training data
        real_texts = [t for t, lab in zip(X_train, y_train) if lab == true_label]
        if real_texts:
            real_text = random.choice(real_texts)
            real_log_prob = true_lm.sentence_log_prob(real_text)
            print(f"  Real text from class {true_label}: '{real_text[:60]}...'")
            print(f"  Real text log P(x|y={true_label}): {real_log_prob:.4f}")
            print(f"  Difference: {real_log_prob - true_log_prob:.4f}")
        
        print()
    
    # Check overall accuracy
    ngram_preds = ngram_clf.predict(X_syn_valid, use_uniform_prior=True)
    acc = accuracy_score(y_syn_valid, ngram_preds)
    print(f"[RESULT] Overall accuracy: {acc:.4f}")
    print()
    
    # Analyze: why is the accuracy so low?
    print("[ANALYSIS] Why is accuracy so low?")
    print("  Checking if generated texts have high probability under their true class...")
    
    correct_scores = []
    incorrect_scores = []
    
    for i, (text, true_label) in enumerate(zip(X_syn_valid, y_syn_valid)):
        true_lm = ngram_clf.class_lms[true_label]
        true_log_prob = true_lm.sentence_log_prob(text)
        
        # Get max score from other classes
        other_scores = []
        for y in ngram_clf.classes_:
            if y != true_label:
                lm = ngram_clf.class_lms[y]
                other_scores.append(lm.sentence_log_prob(text))
        max_other_score = max(other_scores) if other_scores else -1e10
        
        if ngram_preds[i] == true_label:
            correct_scores.append((true_log_prob, max_other_score))
        else:
            incorrect_scores.append((true_log_prob, max_other_score))
    
    if correct_scores:
        avg_correct_true = sum(s[0] for s in correct_scores) / len(correct_scores)
        avg_correct_other = sum(s[1] for s in correct_scores) / len(correct_scores)
        print(f"  Correct predictions ({len(correct_scores)}):")
        print(f"    Avg true class log prob: {avg_correct_true:.4f}")
        print(f"    Avg max other class log prob: {avg_correct_other:.4f}")
        print(f"    Margin: {avg_correct_true - avg_correct_other:.4f}")
    
    if incorrect_scores:
        avg_incorrect_true = sum(s[0] for s in incorrect_scores) / len(incorrect_scores)
        avg_incorrect_other = sum(s[1] for s in incorrect_scores) / len(incorrect_scores)
        print(f"  Incorrect predictions ({len(incorrect_scores)}):")
        print(f"    Avg true class log prob: {avg_incorrect_true:.4f}")
        print(f"    Avg max other class log prob: {avg_incorrect_other:.4f}")
        print(f"    Margin: {avg_incorrect_true - avg_incorrect_other:.4f}")
        print(f"  ⚠️  Problem: True class has LOWER probability than other classes!")
        print(f"     This suggests the generation process doesn't match the prediction process.")


if __name__ == "__main__":
    debug_generation()

