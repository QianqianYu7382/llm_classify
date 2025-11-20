"""
lm_classifier.py

基于类别条件 n-gram 语言模型的新闻主题分类器：
    - 使用 synthetic_data.py 生成合成数据
    - 对每个类别训练一个 trigram 语言模型 P(x|y)
    - 使用 Bayes 规则：y_hat = argmax_y [ log P(x|y) + log P(y) ]
"""

import re
import math
import random
from collections import Counter
from typing import List, Tuple, Dict

from synthetic_data import generate_synthetic_news, TOPIC_NAMES


# -------------------------------
# 1. 简单分词函数
# -------------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    非常简单的分词函数：
    - 全部转为小写
    - 只保留字母和数字
    - 返回 token 列表
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


# -------------------------------
# 2. n-gram 语言模型（默认 trigram）
# -------------------------------

class NgramLanguageModel:
    """
    简单的 n-gram 语言模型（默认 trigram），使用加性平滑 (add-alpha)
    用来估计每个类别下的 P(x | y)。
    """

    def __init__(self, n: int = 3, unk_threshold: int = 1, alpha: float = 1.0):
        """
        :param n: n-gram 阶数，例如 3 = trigram
        :param unk_threshold: 词频 <= 这个值的词会被映射为 <UNK>
        :param alpha: 平滑系数（加性平滑）
        """
        assert n >= 1
        self.n = n
        self.unk_threshold = unk_threshold
        self.alpha = alpha

        # 词表 & 计数字典
        self.vocab = set()
        self.word_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        # 特殊 token
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self._fitted = False

    def _build_vocab(self, texts: List[str]) -> None:
        """
        根据训练文本构建词表，统计词频。
        """
        freq = Counter()
        for text in texts:
            tokens = simple_tokenize(text)
            freq.update(tokens)

        self.word_counts = freq
        # 只保留频率 > unk_threshold 的词
        self.vocab = {w for w, c in freq.items() if c > self.unk_threshold}
        # 加入特殊 token
        self.vocab.add(self.unk_token)
        self.vocab.add(self.bos_token)
        self.vocab.add(self.eos_token)

    def _map_token(self, token: str) -> str:
        """
        将低频词或未登录词映射为 <UNK>
        """
        token = token.lower()
        if token in self.vocab:
            return token
        return self.unk_token

    def fit(self, texts: List[str]) -> None:
        """
        在给定文本列表上训练 n-gram 语言模型：
        - 统计 n-gram 计数和 (n-1)-gram context 计数
        """
        self._build_vocab(texts)

        n = self.n
        for text in texts:
            tokens = [self._map_token(t) for t in simple_tokenize(text)]
            # 开头加 (n-1) 个 BOS，结尾加 EOS
            seq = [self.bos_token] * (n - 1) + tokens + [self.eos_token]

            for i in range(n - 1, len(seq)):
                ngram = tuple(seq[i - n + 1: i + 1])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self._fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def conditional_prob(self, ngram: Tuple[str, ...]) -> float:
        """
        计算 P(w_t | context) 带加性平滑：
            (count(ngram) + alpha) / (count(context) + alpha * |V|)
        """
        assert self._fitted, "Language model not fitted."

        context = ngram[:-1]
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        V = self.vocab_size
        alpha = self.alpha

        num = count_ngram + alpha
        den = count_context + alpha * V
        if den == 0:
            # 如果 context 从未出现过，退化为均匀分布
            return 1.0 / V
        return num / den

    def sentence_log_prob(self, text: str) -> float:
        """
        计算给定句子在该 LM 下的 log P(text)
        = sum_t log P(w_t | context)
        """
        assert self._fitted, "Language model not fitted."

        n = self.n
        tokens = [self._map_token(t) for t in simple_tokenize(text)]
        seq = [self.bos_token] * (n - 1) + tokens + [self.eos_token]

        log_p = 0.0
        for i in range(n - 1, len(seq)):
            ngram = tuple(seq[i - n + 1: i + 1])
            p = self.conditional_prob(ngram)
            log_p += math.log(p + 1e-12)  # 避免 log(0)

        return log_p


# -------------------------------
# 3. 类别条件 LM 分类器
# -------------------------------

class ClassConditionalLMClassifier:
    """
    基于类别条件 n-gram 语言模型的分类器：
        - 每个类别 y 都有一个 NgramLanguageModel 来估计 P(x|y)
        - 同时估计类先验 P(y)
        - 预测时使用：argmax_y [ log P(x|y) + log P(y) ]
    """

    def __init__(self, n: int = 3, unk_threshold: int = 1, alpha: float = 1.0):
        self.n = n
        self.unk_threshold = unk_threshold
        self.alpha = alpha

        self.class_lms: Dict[int, NgramLanguageModel] = {}
        self.class_priors: Dict[int, float] = {}
        self.classes_: List[int] = []
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """
        在标注数据上训练分类器：
            - 估计每个类别的先验 P(y)
            - 为每个类别训练一个 NgramLanguageModel (P(x|y))
        """
        labels = list(labels)
        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels

        # 估计 P(y)
        total = len(labels)
        priors = {}
        for y in unique_labels:
            count_y = sum(1 for lab in labels if lab == y)
            priors[y] = count_y / total
        self.class_priors = priors

        # 为每个类别训练一个 LM
        for y in unique_labels:
            class_texts = [t for t, lab in zip(texts, labels) if lab == y]
            lm = NgramLanguageModel(
                n=self.n,
                unk_threshold=self.unk_threshold,
                alpha=self.alpha,
            )
            lm.fit(class_texts)
            self.class_lms[y] = lm

        self._fitted = True

    def _predict_one(self, text: str) -> int:
        """
        对单个样本做预测：
            y_hat = argmax_y [ log P(x|y) + log P(y) ]
        """
        assert self._fitted, "Classifier not fitted."

        best_y = None
        best_score = -1e18  # 近似负无穷
        for y in self.classes_:
            lm = self.class_lms[y]
            log_px_given_y = lm.sentence_log_prob(text)
            log_py = math.log(self.class_priors[y] + 1e-12)
            score = log_px_given_y + log_py
            if score > best_score:
                best_score = score
                best_y = y
        return best_y

    def predict(self, texts: List[str]) -> List[int]:
        """
        对多个文本批量预测标签
        """
        return [self._predict_one(t) for t in texts]

    def score(self, texts: List[str], labels: List[int]) -> float:
        """
        简单计算分类准确率
        """
        preds = self.predict(texts)
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        return correct / len(labels) if labels else 0.0


# -------------------------------
# 4. 一个简单的 train/test 划分工具
# -------------------------------

def train_test_split_simple(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    一个不用 sklearn 的简单 train/test 划分函数：
        - 打乱索引
        - 按比例切分
    """
    random.seed(random_state)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    split = int(len(indices) * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


# -------------------------------
# 5. 主程序：在合成数据上训练 & 评估 LM 分类器
# -------------------------------

if __name__ == "__main__":
    # Step 3a: 使用合成数据做实验
    texts, labels = generate_synthetic_news(
        n_per_class=500,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split_simple(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    clf = ClassConditionalLMClassifier(
        n=3,           # trigram
        unk_threshold=1,
        alpha=0.5,     # 平滑系数，可以自己调
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"LM classifier accuracy on synthetic data: {acc:.4f}")

    # 打印一些预测示例
    print("\nSample predictions:")
    for i in range(5):
        text = X_test[i]
        true_label = y_test[i]
        pred_label = clf.predict([text])[0]
        print("-" * 60)
        print("TEXT      :", text)
        print("True label:", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred label:", pred_label, "-", TOPIC_NAMES[pred_label])
