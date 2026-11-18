"""Help-seeking classifier: what is the learner asking the AI to do?

Labels (grounded in help-seeking research in ITS, e.g. Aleven et al. 2003/2016):
  answer      - executive help-seeking: "give me the code / the answer"
  explain     - instrumental help-seeking: "why / how does X work"
  verify      - "is my reasoning / code right?" (monitoring)
  plan        - learner proposes an approach, asks about next step
  offtask     - unrelated

Two implementations, compared against hand labels in the paper:
  TfidfClassifier  - CPU, deterministic, trained on the seed set (data/seed_labels.csv)
  LLMClassifier    - zero-shot with a rubric prompt (prompts/classify.txt)
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

LABELS = ["answer", "explain", "verify", "plan", "offtask"]
ROOT = Path(__file__).resolve().parent.parent


def load_seed(path=ROOT / "data" / "seed_labels.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


class TfidfClassifier:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union

        self.pipe = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
        )

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict(self, text: str) -> str:
        return self.pipe.predict([text])[0]

    def predict_many(self, texts):
        return list(self.pipe.predict(texts))


_LLM_LABEL = re.compile(r"\b(" + "|".join(LABELS) + r")\b", re.I)


class LLMClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = (ROOT / "prompts" / "classify.txt").read_text()

    def predict(self, text: str) -> str:
        out = self.llm.complete([{"role": "system", "content": self.prompt},
                                 {"role": "user", "content": text}], max_tokens=5)
        m = _LLM_LABEL.search(out or "")
        return m.group(1).lower() if m else "explain"


_default: TfidfClassifier | None = None


def default_classifier() -> TfidfClassifier:
    """Load the saved model from models/ if present, else train from the seed set."""
    global _default
    if _default is None:
        path = ROOT / "models" / "help_seeking_tfidf_lr.joblib"
        clf = TfidfClassifier()
        if path.exists():
            import joblib
            clf.pipe = joblib.load(path)
        else:
            clf.fit(*load_seed())
        _default = clf
    return _default
