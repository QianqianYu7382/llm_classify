
import pandas as pd
import numpy as np
import sys
from io import StringIO

from lm_classifier import ClassConditionalLMClassifier
from sklearn.metrics import classification_report, confusion_matrix

TOPIC_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_real_data(
    train_path: str = "training_data.csv",
    test_path: str = "test_data.csv",
):
    train_df = pd.read_csv(train_path)
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

    print(f"[INFO] Loaded real data:")
    print(f"  Train size = {len(X_train)}")
    print(f"  Test  size = {len(X_test)}")
    print(f"  Train label distribution:\n{train_df['label'].value_counts()}")
    print(f"  Test  label distribution:\n{test_df['label'].value_counts()}\n")

    return X_train, y_train, X_test, y_test



def run_real_data_experiment():

    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer


    print("[INFO] Loading data...")
    X_train, y_train, X_test, y_test = load_real_data()

    clf = ClassConditionalLMClassifier(
        n=3,          
        unk_threshold=2,
        alpha=0.5,
    )

    print("[INFO] Fitting trigram LM classifier...")
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"\n[RESULT] LM classifier accuracy on REAL test set: {acc:.4f}\n")

    print("Sample predictions on real test data:")
    for i in range(min(10, len(X_test))):
        text = X_test[i]
        true_label = y_test[i]
        pred_label = clf.predict([text])[0]
        print("-" * 80)
        print("TEXT       :", text)
        print("True label :", true_label, "-", TOPIC_NAMES[true_label])
        print("Pred label :", pred_label, "-", TOPIC_NAMES[pred_label])


    y_pred = clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=TOPIC_NAMES))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


    sys.stdout = original_stdout


    output_path = "real_lm_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    print(f"[INFO] All results written to {output_path}")
if __name__ == "__main__":
    run_real_data_experiment()
