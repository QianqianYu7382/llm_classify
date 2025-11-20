"""
lm_experiments.py

在 synthetic_data + lm_classifier 的基础上，跑 4 组实验：
1) 不同 n-gram 阶数 (1, 2, 3, 4) 的性能对比
2) 不同平滑系数 alpha 对性能的影响
3) 不同训练集大小 (100%, 50%, 20%) 对性能的影响
4) 在干净 vs 加噪声的测试集上的鲁棒性对比
"""

import random
from typing import List, Tuple

from synthetic_data import generate_synthetic_news, TOPIC_NAMES
from lm_classifier import (
    ClassConditionalLMClassifier,
    train_test_split_simple,
)


# ------------------------------------------------
# 辅助：生成统一的 train/test 划分，供所有实验复用
# ------------------------------------------------

def get_base_dataset(
    n_per_class: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
):
    texts, labels = generate_synthetic_news(
        n_per_class=n_per_class,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split_simple(
        texts, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ------------------------------------------------
# 实验 1：不同 n-gram 阶数
# ------------------------------------------------

def experiment_ngram_orders():
    print("\n=== Experiment 1: Unigram vs Bigram vs Trigram vs 4-gram ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    orders = [1, 2, 3, 4]
    results = []

    for n in orders:
        clf = ClassConditionalLMClassifier(
            n=n,
            unk_threshold=1,
            alpha=0.5,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results.append((n, acc))

    print("n-gram order\tAccuracy")
    for n, acc in results:
        print(f"{n}\t\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- Unigram ignores word order, so it may underperform.")
    print("- Bigram captures short dependencies.")
    print("- Trigram often balances context vs sparsity best.")
    print("- 4-gram may suffer from data sparsity on short titles.\n")


# ------------------------------------------------
# 实验 2：不同 alpha（平滑系数）
# ------------------------------------------------

def experiment_smoothing_alphas():
    print("\n=== Experiment 2: Effect of smoothing alpha (trigram) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    alphas = [0.1, 0.5, 1.0]
    results = []

    for alpha in alphas:
        clf = ClassConditionalLMClassifier(
            n=3,              # 固定 trigram
            unk_threshold=1,
            alpha=alpha,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results.append((alpha, acc))

    print("alpha\tAccuracy")
    for alpha, acc in results:
        print(f"{alpha:.2f}\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- Smaller alpha relies more on observed counts (less smoothing).")
    print("- Larger alpha pushes probabilities toward a more uniform distribution.")
    print("- Too much smoothing can hurt discriminative power.\n")


# ------------------------------------------------
# 实验 3：不同训练集大小 (100%, 50%, 20%)
# ------------------------------------------------

def subsample_train(
    X_train: List[str],
    y_train: List[int],
    fraction: float,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    从训练集中按比例采样一部分样本，用于模拟数据变少的情况。
    """
    random.seed(random_state)
    n = len(X_train)
    k = max(1, int(n * fraction))
    indices = list(range(n))
    random.shuffle(indices)
    sel = indices[:k]
    X_sub = [X_train[i] for i in sel]
    y_sub = [y_train[i] for i in sel]
    return X_sub, y_sub


def experiment_train_size():
    print("\n=== Experiment 3: Effect of training set size (trigram) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    fractions = [1.0, 0.5, 0.2]
    results = []

    for frac in fractions:
        X_sub, y_sub = subsample_train(X_train, y_train, fraction=frac, random_state=123)
        clf = ClassConditionalLMClassifier(
            n=3,
            unk_threshold=1,
            alpha=0.5,
        )
        clf.fit(X_sub, y_sub)
        acc = clf.score(X_test, y_test)
        results.append((frac, len(X_sub), acc))

    print("Train fraction\t#Train samples\tAccuracy")
    for frac, n_train, acc in results:
        print(f"{frac:.1f}\t\t{n_train}\t\t{acc:.4f}")

    print("\n(Interpretation suggestion)")
    print("- As we reduce training data, n-gram probabilities become sparser.")
    print("- This degradation shows that classical n-gram LMs are data-hungry.")
    print("- You can plot fraction vs accuracy in the report for a nice figure.\n")


# ------------------------------------------------
# 实验 4：加噪声测试鲁棒性
# ------------------------------------------------

def noisify_text(
    text: str,
    p_delete: float = 0.2,
    p_swap: float = 0.2,
    random_state: int = None,
) -> str:
    """
    对句子做简单扰动：
        - 以概率 p_delete 删除一个词
        - 以概率 p_swap 随机交换一对相邻词

    这里只做很简单的 noise，主要是说明模型对词序/缺词的敏感性。
    """
    if random_state is not None:
        random.seed(random_state)

    tokens = text.split()
    if not tokens:
        return text

    # 随机删除一个词
    if len(tokens) > 1 and random.random() < p_delete:
        idx_del = random.randrange(len(tokens))
        del tokens[idx_del]

    # 随机交换相邻词
    if len(tokens) > 1 and random.random() < p_swap:
        i = random.randrange(len(tokens) - 1)
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

    return " ".join(tokens)


def experiment_noise_robustness():
    print("\n=== Experiment 4: Robustness to noisy test input (trigram) ===")
    X_train, X_test, y_train, y_test = get_base_dataset()

    # 先在干净数据上训练 trigram LM
    clf = ClassConditionalLMClassifier(
        n=3,
        unk_threshold=1,
        alpha=0.5,
    )
    clf.fit(X_train, y_train)

    # 评估在 clean test 上的性能
    acc_clean = clf.score(X_test, y_test)

    # 构造 noisy test
    X_test_noisy = [
        noisify_text(t, p_delete=0.2, p_swap=0.2) for t in X_test
    ]
    acc_noisy = clf.score(X_test_noisy, y_test)

    print(f"Accuracy on clean test set : {acc_clean:.4f}")
    print(f"Accuracy on noisy test set : {acc_noisy:.4f}")

    # 打印几个示例，方便写 qualitative 分析
    print("\nSample clean vs noisy examples:")
    for i in range(5):
        print("-" * 60)
        print("CLEAN:", X_test[i])
        print("NOISY:", X_test_noisy[i])
        true_label = y_test[i]
        pred_clean = clf.predict([X_test[i]])[0]
        pred_noisy = clf.predict([X_test_noisy[i]])[0]
        print("True label :", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred clean :", pred_clean, "-", TOPIC_NAMES[pred_clean])
        print("Pred noisy :", pred_noisy, "-", TOPIC_NAMES[pred_noisy])

    print("\n(Interpretation suggestion)")
    print("- If accuracy drops notably on the noisy test set,")
    print("  this shows that the n-gram LM is sensitive to word deletion and reordering.")
    print("- This is an important CON (limitation) you can highlight in the report.\n")


# ------------------------------------------------
# 主入口：依次跑 4 个实验
# ------------------------------------------------

if __name__ == "__main__":
    # 你可以自己决定要不要全部跑，如果只想跑某几个就注释掉。
    experiment_ngram_orders()
    experiment_smoothing_alphas()
    experiment_train_size()
    experiment_noise_robustness()
