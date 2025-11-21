

import random
import sys
from io import StringIO
from typing import List, Tuple

import pandas as pd

from lm_classifier import (
    ClassConditionalLMClassifier,
)


TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def capture_output_to_txt(func):


    def wrapper(*args, **kwargs):
        buffer = StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer

        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
            txt_path = "lm_experiments_results.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(buffer.getvalue())

        print(f"[INFO] All experiment outputs written to {txt_path}")

    return wrapper



def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
) -> Tuple[List[str], List[int], List[str], List[int]]:

    train_df = pd.read_csv(train_path)
    if not {"text", "label"} <= set(train_df.columns):
        raise ValueError(
            f"training_data.csv should have columns ['text', 'label'], "
            f"but got {list(train_df.columns)}"
        )

    test_df = pd.read_csv(test_path, header=None, names=["text", "label"])

    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)


    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    return X_train, y_train, X_test, y_test


def get_base_dataset():

    X_train, y_train, X_test, y_test = load_real_data()
    print("[INFO] Loaded REAL dataset for experiments:")
    print(f"  Train size = {len(X_train)}")
    print(f"  Test  size = {len(X_test)}\n")
    return X_train, X_test, y_train, y_test



def experiment_ngram_orders():
    print("\n=== Experiment 1: Unigram vs Bigram vs Trigram vs 4-gram (REAL data) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    orders = [1, 2, 3, 4]
    results = []

    for n in orders:
        clf = ClassConditionalLMClassifier(n=n, unk_threshold=2, alpha=0.5)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results.append((n, acc))

    print("n-gram order\tAccuracy (REAL data)")
    for n, acc in results:
        print(f"{n}\t\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- Unigram ignores word order → usually weakest.")
    print("- Bigram adds minimal context.")
    print("- Trigram often best balance between context and sparsity.")
    print("- 4-gram on REAL short headlines likely suffers heavy sparsity.\n")



def experiment_smoothing_alphas():
    print("\n=== Experiment 2: Effect of smoothing alpha (trigram, REAL data) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    alphas = [0.1, 0.5, 1.0]
    results = []

    for alpha in alphas:
        clf = ClassConditionalLMClassifier(n=3, unk_threshold=2, alpha=alpha)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results.append((alpha, acc))

    print("alpha\tAccuracy (REAL data)")
    for alpha, acc in results:
        print(f"{alpha:.2f}\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- Smaller alpha → rely more on observed n-gram counts.")
    print("- Larger alpha → stronger smoothing, probabilities more uniform.")
    print("- On REAL data, too much smoothing can blur class differences.\n")


def subsample_train(
    X_train: List[str],
    y_train: List[int],
    fraction: float,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:

    random.seed(random_state)
    n = len(X_train)
    k = max(1, int(n * fraction))
    indices = list(range(n))
    random.shuffle(indices)
    sel = indices[:k]
    return [X_train[i] for i in sel], [y_train[i] for i in sel]


def experiment_train_size():
    print("\n=== Experiment 3: Effect of training set size (REAL data) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    fractions = [1.0, 0.5, 0.2]
    results = []

    for frac in fractions:
        X_sub, y_sub = subsample_train(X_train, y_train, fraction=frac, random_state=123)
        clf = ClassConditionalLMClassifier(n=3, unk_threshold=2, alpha=0.5)
        clf.fit(X_sub, y_sub)
        acc = clf.score(X_test, y_test)
        results.append((frac, len(X_sub), acc))

    print("Train fraction\t#Train samples\tAccuracy (REAL data)")
    for frac, n_train, acc in results:
        print(f"{frac:.1f}\t\t{n_train}\t\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- As we reduce REAL training data, sparsity becomes more severe.")
    print("- Performance drop quantifies how data-hungry n-gram LMs are.")
    print("- Good figure: accuracy vs. train fraction on REAL dataset.\n")


def noisify_text(text: str, p_delete: float = 0.2, p_swap: float = 0.2):
    tokens = text.split()
    if not tokens:
        return text


    if random.random() < p_delete and len(tokens) > 1:
        del tokens[random.randrange(len(tokens))]


    if random.random() < p_swap and len(tokens) > 1:
        i = random.randrange(len(tokens) - 1)
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

    return " ".join(tokens)


def experiment_noise_robustness():
    print("\n=== Experiment 4: Robustness to noisy REAL test input ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    clf = ClassConditionalLMClassifier(n=3, unk_threshold=2, alpha=0.5)
    clf.fit(X_train, y_train)

    acc_clean = clf.score(X_test, y_test)

    X_test_noisy = [noisify_text(t) for t in X_test]
    acc_noisy = clf.score(X_test_noisy, y_test)

    print(f"Accuracy on CLEAN REAL test set : {acc_clean:.4f}")
    print(f"Accuracy on NOISY REAL test set : {acc_noisy:.4f}")

    print("\nSample clean vs noisy examples (REAL data):")
    for i in range(min(5, len(X_test))):
        print("-" * 60)
        print("CLEAN:", X_test[i])
        print("NOISY:", X_test_noisy[i])
        true = y_test[i]
        pred_c = clf.predict([X_test[i]])[0]
        pred_n = clf.predict([X_test_noisy[i]])[0]
        print("True label :", true, "-", TOPIC_NAMES[true])
        print("Pred clean :", pred_c, "-", TOPIC_NAMES[pred_c])
        print("Pred noisy :", pred_n, "-", TOPIC_NAMES[pred_n])

    print("\n(Interpretation suggestion)")
    print("- If accuracy drops a lot on noisy REAL headlines,")
    print("  it shows n-gram LM is sensitive to word deletion/reordering.")
    print("- This is an important CON for Method A in the report.\n")

@capture_output_to_txt
def run_all_experiments():
    experiment_ngram_orders()
    experiment_smoothing_alphas()
    experiment_train_size()
    experiment_noise_robustness()


if __name__ == "__main__":
    run_all_experiments()
