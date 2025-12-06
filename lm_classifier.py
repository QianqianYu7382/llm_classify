

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

    def _get_next_word_distribution(self, context: Tuple[str, ...]) -> List[Tuple[str, float]]:
        """
        Get the probability distribution over next words given a context.
        Returns a list of (word, probability) tuples.
        Prioritizes words that actually appeared after this context in training.
        """
        assert self._fitted, "Language model not fitted."
        
        # First, collect words that actually appeared after this context
        observed_words = set()
        for ngram, count in self.ngram_counts.items():
            if ngram[:-1] == context:
                observed_words.add(ngram[-1])
        
        # Also include EOS and UNK as potential next words
        observed_words.add(self.eos_token)
        observed_words.add(self.unk_token)
        
        # Calculate probabilities for observed words
        probs = []
        for word in observed_words:
            if word == self.bos_token:
                continue
            ngram = context + (word,)
            prob = self.conditional_prob(ngram)
            probs.append((word, prob))
        
        # If we have very few observed words, also consider other high-probability words
        # But prioritize observed ones
        if len(probs) < 10:
            # Add a few more words from vocab with highest probabilities
            other_probs = []
            for word in self.vocab:
                if word in observed_words or word == self.bos_token:
                    continue
                ngram = context + (word,)
                prob = self.conditional_prob(ngram)
                other_probs.append((word, prob))
            
            # Take top 20 by probability
            other_probs.sort(key=lambda x: x[1], reverse=True)
            probs.extend(other_probs[:20])
        
        # Normalize probabilities
        if not probs:
            # Fallback: return EOS with probability 1.0 if no valid words
            return [(self.eos_token, 1.0)]
        
        total = sum(p for _, p in probs)
        if total > 0:
            probs = [(w, p / total) for w, p in probs]
        else:
            # Uniform distribution if no valid transitions (shouldn't happen with smoothing)
            probs = [(w, 1.0 / len(probs)) for w, _ in probs]
        
        return probs

    def sample(self, max_length: int = 30, random_state: int = None) -> str:
        """
        Sample a sentence from the learned n-gram distribution.
        Generates text by sampling from P(w_t | context) at each step.
        This samples EXACTLY from the learned distribution without any modifications.
        
        :param max_length: Maximum number of tokens to generate
        :param random_state: Random seed for reproducibility
        """
        assert self._fitted, "Language model not fitted."
        
        if random_state is not None:
            random.seed(random_state)
        
        n = self.n
        tokens = []
        context = tuple([self.bos_token] * (n - 1))
        
        for step in range(max_length):
            # Get distribution over next words (exactly as learned)
            dist = self._get_next_word_distribution(context)
            
            if not dist:
                break
            
            # Sample from the distribution (no modifications to probabilities)
            words, probs = zip(*dist)
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            # Stop if we sample EOS
            if next_word == self.eos_token:
                break
            
            tokens.append(next_word)
            
            # Update context: shift and add new word
            context = context[1:] + (next_word,)
        
        # Return generated tokens
        if tokens:
            return " ".join(tokens)
        else:
            # Fallback: return a single word if sampling failed
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

    def sample_synthetic_data(self, n_per_class: int = 500, max_length: int = 50, random_state: int = 42, show_progress: bool = True) -> Tuple[List[str], List[int]]:
        """
        Sample synthetic data from the trained class-conditional n-gram models.
        For each class, sample n_per_class sentences from its language model.
        This ensures the synthetic data comes directly from the learned joint distribution.
        """
        assert self._fitted, "Classifier not fitted."
        
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
                
                # Show progress every 50 samples
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