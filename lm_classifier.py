

import re
import math
import random
import numpy as np
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

    def _get_next_word_distribution(self, context: Tuple[str, ...]) -> List[Tuple[str, float]]:
        assert self._fitted, "Language model not fitted."
        probs = []
        for word in self.vocab:
            if word == self.bos_token:
                continue
            ngram = context + (word,)
            prob = self.conditional_prob(ngram)
            probs.append((word, prob))
        if not probs:
            return [(self.eos_token, 1.0)]
        total = sum(p for _, p in probs)
        if total > 0:
            probs = [(w, p / total) for w, p in probs]
        else:
            probs = [(w, 1.0 / len(probs)) for w, _ in probs]
        return probs

    def sample(self, max_length: int = 30, random_state: int = None) -> str:
        assert self._fitted, "Language model not fitted."
        if random_state is not None:
            random.seed(random_state)
        n = self.n
        tokens = []
        context = tuple([self.bos_token] * (n - 1))
        for step in range(max_length):
            dist = self._get_next_word_distribution(context)
            if not dist:
                break
            words, probs = zip(*dist)
            next_word = random.choices(words, weights=probs, k=1)[0]
            if next_word == self.eos_token:
                break
            tokens.append(next_word)
            context = context[1:] + (next_word,)
        if tokens:
            filtered_tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token, self.unk_token]]
            if filtered_tokens:
                return " ".join(filtered_tokens)
        if self.vocab:
            fallback_words = [w for w in self.vocab if w not in [self.bos_token, self.eos_token, self.unk_token]]
            if fallback_words:
                return random.choice(fallback_words)
        return "<UNK>"



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

    def _predict_one(self, text: str, use_uniform_prior: bool = False) -> int:
        assert self._fitted, "Classifier not fitted."
        best_y = None
        best_score = -1e18
        for y in self.classes_:
            lm = self.class_lms[y]
            log_px_given_y = lm.sentence_log_prob(text)
            if use_uniform_prior:
                log_py = math.log(1.0 / len(self.classes_))
            else:
                log_py = math.log(self.class_priors[y] + 1e-12)
            score = log_px_given_y + log_py
            if score > best_score:
                best_score = score
                best_y = y
        return best_y

    def predict(self, texts: List[str], use_uniform_prior: bool = False) -> List[int]:
        return [self._predict_one(t, use_uniform_prior=use_uniform_prior) for t in texts]

    def score(self, texts: List[str], labels: List[int], use_uniform_prior: bool = False) -> float:
        preds = self.predict(texts, use_uniform_prior=use_uniform_prior)
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        return correct / len(labels) if labels else 0.0

    def sample_synthetic_data(self, n_per_class: int = 500, max_length: int = 50, random_state: int = 42, show_progress: bool = True, use_training_data: bool = False, training_texts: List[str] = None, training_labels: List[int] = None) -> Tuple[List[str], List[int]]:
        assert self._fitted, "Classifier not fitted."
        if use_training_data and training_texts is not None and training_labels is not None:
            return self._sample_from_training_data(n_per_class, training_texts, training_labels, random_state, show_progress)
        else:
            return self._generate_new_sentences(n_per_class, max_length, random_state, show_progress)
    
    def _sample_from_training_data(self, n_per_class: int, training_texts: List[str], training_labels: List[int], random_state: int, show_progress: bool) -> Tuple[List[str], List[int]]:
        import random as rnd
        rnd.seed(random_state)
        texts = []
        labels = []
        for class_idx, y in enumerate(self.classes_):
            if show_progress:
                print(f"    Sampling class {y} ({class_idx + 1}/{len(self.classes_)})...", end="", flush=True)
            class_texts = [text for text, label in zip(training_texts, training_labels) if label == y]
            if len(class_texts) < n_per_class:
                selected_texts = class_texts
            else:
                lm = self.class_lms[y]
                text_scores = []
                for text in class_texts:
                    log_prob = lm.sentence_log_prob(text)
                    text_scores.append((text, log_prob))
                text_scores.sort(key=lambda x: x[1], reverse=True)
                selected_texts = [text for text, _ in text_scores[:n_per_class]]
            texts.extend(selected_texts)
            labels.extend([y] * len(selected_texts))
            if show_progress:
                print(f" {len(selected_texts)} samples ✓")
        return texts, labels
    
    def _generate_new_sentences(self, n_per_class: int, max_length: int, random_state: int, show_progress: bool) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        random.seed(random_state)
        seed_offset = 0
        total_samples = len(self.classes_) * n_per_class
        samples_generated = 0
        for class_idx, y in enumerate(self.classes_):
            if show_progress:
                print(f"    Sampling class {y} ({class_idx + 1}/{len(self.classes_)})...", end="", flush=True)
            lm = self.class_lms[y]
            for i in range(n_per_class):
                text = lm.sample(max_length=max_length, random_state=random_state + seed_offset)
                texts.append(text)
                labels.append(y)
                seed_offset += 1
                samples_generated += 1
                if show_progress and (samples_generated % 50 == 0 or samples_generated == total_samples):
                    print(f" {samples_generated}/{total_samples}", end="", flush=True)
            if show_progress:
                print(" ✓")
        return texts, labels



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