

import re
import math
import random
from collections import Counter
from typing import List, Tuple, Dict

from synthetic_data import generate_synthetic_news, TOPIC_NAMES




def simple_tokenize(text: str) -> List[str]:

    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens



class NgramLanguageModel:


    def __init__(self, n: int = 3, unk_threshold: int = 1, alpha: float = 1.0):

        assert n >= 1
        self.n = n
        self.unk_threshold = unk_threshold
        self.alpha = alpha


        self.vocab = set()
        self.word_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()


        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self._fitted = False

    def _build_vocab(self, texts: List[str]) -> None:

        freq = Counter()
        for text in texts:
            tokens = simple_tokenize(text)
            freq.update(tokens)

        self.word_counts = freq

        self.vocab = {w for w, c in freq.items() if c > self.unk_threshold}

        self.vocab.add(self.unk_token)
        self.vocab.add(self.bos_token)
        self.vocab.add(self.eos_token)

    def _map_token(self, token: str) -> str:

        token = token.lower()
        if token in self.vocab:
            return token
        return self.unk_token

    def fit(self, texts: List[str]) -> None:

        self._build_vocab(texts)

        n = self.n
        for text in texts:
            tokens = [self._map_token(t) for t in simple_tokenize(text)]
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

        assert self._fitted, "Language model not fitted."

        context = ngram[:-1]
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        V = self.vocab_size
        alpha = self.alpha

        num = count_ngram + alpha
        den = count_context + alpha * V
        if den == 0:

            return 1.0 / V
        return num / den

    def sentence_log_prob(self, text: str) -> float:

        assert self._fitted, "Language model not fitted."

        n = self.n
        tokens = [self._map_token(t) for t in simple_tokenize(text)]
        seq = [self.bos_token] * (n - 1) + tokens + [self.eos_token]

        log_p = 0.0
        for i in range(n - 1, len(seq)):
            ngram = tuple(seq[i - n + 1: i + 1])
            p = self.conditional_prob(ngram)
            log_p += math.log(p + 1e-12)  

        return log_p



class ClassConditionalLMClassifier:


    def __init__(self, n: int = 3, unk_threshold: int = 1, alpha: float = 1.0):
        self.n = n
        self.unk_threshold = unk_threshold
        self.alpha = alpha

        self.class_lms: Dict[int, NgramLanguageModel] = {}
        self.class_priors: Dict[int, float] = {}
        self.classes_: List[int] = []
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> None:

        labels = list(labels)
        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels


        total = len(labels)
        priors = {}
        for y in unique_labels:
            count_y = sum(1 for lab in labels if lab == y)
            priors[y] = count_y / total
        self.class_priors = priors


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

        assert self._fitted, "Classifier not fitted."

        best_y = None
        best_score = -1e18
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

        return [self._predict_one(t) for t in texts]

    def score(self, texts: List[str], labels: List[int]) -> float:

        preds = self.predict(texts)
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        return correct / len(labels) if labels else 0.0



def train_test_split_simple(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
):

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



if __name__ == "__main__":

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
        n=3,           
        unk_threshold=1,
        alpha=0.5,     
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"LM classifier accuracy on synthetic data: {acc:.4f}")


    print("\nSample predictions:")
    for i in range(5):
        text = X_test[i]
        true_label = y_test[i]
        pred_label = clf.predict([text])[0]
        print("-" * 60)
        print("TEXT      :", text)
        print("True label:", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred label:", pred_label, "-", TOPIC_NAMES[pred_label])