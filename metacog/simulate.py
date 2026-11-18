"""Simulated participants, driven through the real HTTP API (FastAPI TestClient).

PURPOSE: end-to-end validation of the study pipeline (randomisation, classifier,
policy, code grading, logging, analysis) before running real people.
The behavioural model below is an ASSUMPTION, not a finding; never report simulated
effects as results.

Learner model: skill theta ~ N(0,1), reliance rho ~ Beta(2,2).
  - each chat turn is answer-seeking w.p. rho (else explain/verify/plan)
  - a plan/monitor nudge makes the learner attempt themselves w.p. 0.6
  - own attempts accumulate 'effort'; post-test P(correct) = sigmoid(theta + b*effort)
  - confidence = sigmoid(theta + overconf*rho + noise); monitoring nudges reduce overconf
"""
from __future__ import annotations

import math
import os
import random

MESSAGES = {
    "answer": ["just give me the code", "write it for me", "can you solve this", "show me the full solution"],
    "explain": ["why does the loop work", "how does range work here", "can you explain dictionaries"],
    "verify": ["is my code correct", "can you check my solution", "will this work for an empty list"],
    "plan": ["i think i should loop and keep a running total, is that ok", "should i use a while loop here"],
}
SOLUTIONS = {
    "t1": "def sum_squares(xs):\n    return sum(x*x for x in xs)",
    "t2": "def second_last(xs):\n    return xs[-2] if len(xs) >= 2 else None",
    "t3": "def char_counts(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
    "t4": "def halvings(n):\n    c = 0\n    while n > 1:\n        n //= 2\n        c += 1\n    return c",
    "t5": "def evens(a, b):\n    return [x for x in range(a, b) if x % 2 == 0]",
    "t6": "def classify(n):\n    return 'neg' if n < 0 else ('zero' if n == 0 else 'pos')",
}
BUGGY = "def f(*a):\n    return None"

sig = lambda z: 1 / (1 + math.exp(-z))


def run(n: int = 40, seed: int = 0, db_path: str | None = None):
    if db_path:
        os.environ["METACOG_DB"] = db_path
    from fastapi.testclient import TestClient

    from . import app as appmod
    if db_path:
        from pathlib import Path
        appmod.DB = Path(db_path)
        if appmod.DB.exists():
            appmod.DB.unlink()
    client = TestClient(appmod.app)
    rng = random.Random(seed)
    from .problems import POST_TEST, PRE_TEST

    for _ in range(n):
        theta, rho = rng.gauss(0, 1), rng.betavariate(2, 2)
        s = client.post("/start", json={"consent": True}).json()
        pid = s["pid"]
        for it in PRE_TEST:
            ok = rng.random() < sig(theta - 0.3)
            client.post("/answer", json={"pid": pid, "item_id": it[0], "response": it[3] if ok else "?"})
        effort, monitor_nudges = 0.0, 0
        for t in s["tasks"]:
            attempted = False
            for _turn in range(rng.randint(2, 5)):
                kind = "answer" if rng.random() < rho * (0.5 if attempted else 1) else rng.choice(["explain", "verify", "plan"])
                r = client.post("/chat", json={"pid": pid, "task_id": t["id"], "message": rng.choice(MESSAGES[kind])}).json()
                if r["action"] in ("nudge_plan", "nudge_monitor", "argue_back", "teach_back") and rng.random() < 0.6:
                    attempted, effort = True, effort + 1.0
                    monitor_nudges += r["action"] == "nudge_monitor"
                elif kind in ("plan", "verify"):
                    attempted, effort = True, effort + 0.5
            good = rng.random() < sig(theta + 1.5)
            client.post("/submit", json={"pid": pid, "task_id": t["id"], "code": SOLUTIONS[t["id"]] if good else BUGGY})
        overconf = 1.2 * rho - 0.15 * monitor_nudges
        for it in POST_TEST:
            ok = rng.random() < sig(theta - 0.3 + 0.12 * effort)
            conf = int(100 * sig(theta + overconf + rng.gauss(0, 0.5)))
            client.post("/answer", json={"pid": pid, "item_id": it[0], "response": it[3] if ok else "?", "confidence": conf})
        client.post("/survey", json={"pid": pid, "tlx": rng.randint(2, 6), "perceived_learning": rng.randint(2, 7),
                                     "autonomy": rng.randint(2, 7), "open": "simulated"})
    return appmod.DB


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--db", default="data/sim_study.db")
    a = ap.parse_args()
    print("wrote", run(a.n, db_path=a.db))
