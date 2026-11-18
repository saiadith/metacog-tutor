"""Problem bank: intro Python. Pre/post tests are isomorphic (same concept, different
surface) and auto-graded by output prediction; learning-phase tasks are open coding
tasks graded by hidden unit tests."""
from __future__ import annotations

# (id, concept, prompt, answer) -- output-prediction items, no AI allowed
PRE_TEST = [
    ("pre1", "range", "What does `print(list(range(2, 8, 2)))` print?", "[2, 4, 6]"),
    ("pre2", "indexing", "xs = [3, 1, 4, 1, 5]. What is `xs[-2]`?", "1"),
    ("pre3", "accumulator", "t = 0\nfor x in [1, 2, 3]:\n    t += x * x\nprint(t)", "14"),
    ("pre4", "while", "n = 10\nc = 0\nwhile n > 1:\n    n //= 2\n    c += 1\nprint(c)", "3"),
    ("pre5", "dict", "d = {}\nfor ch in 'banana':\n    d[ch] = d.get(ch, 0) + 1\nprint(d['a'])", "3"),
    ("pre6", "return", "def f(x):\n    if x > 2:\n        return 'big'\n    return 'small'\nprint(f(2))", "small"),
]
POST_TEST = [
    ("post1", "range", "What does `print(list(range(1, 10, 3)))` print?", "[1, 4, 7]"),
    ("post2", "indexing", "ys = [9, 8, 7, 6]. What is `ys[-3]`?", "8"),
    ("post3", "accumulator", "t = 1\nfor x in [1, 2, 3]:\n    t *= x + 1\nprint(t)", "24"),
    ("post4", "while", "n = 27\nc = 0\nwhile n > 1:\n    n //= 3\n    c += 1\nprint(c)", "3"),
    ("post5", "dict", "d = {}\nfor w in 'a b a c a'.split():\n    d[w] = d.get(w, 0) + 1\nprint(d['a'])", "3"),
    ("post6", "return", "def g(x):\n    if x % 2 == 0:\n        return 'even'\n    return 'odd'\nprint(g(7))", "odd"),
]
LEARNING_TASKS = [
    ("t1", "accumulator", "Write `sum_squares(xs)` returning the sum of squares of a list.",
     "sum_squares", [([[1, 2, 3]], 14), ([[]], 0), ([[-2]], 4)]),
    ("t2", "indexing", "Write `second_last(xs)` returning the second-to-last element (None if len<2).",
     "second_last", [([[1, 2, 3]], 2), ([[5]], None)]),
    ("t3", "dict", "Write `char_counts(s)` returning a dict of character counts.",
     "char_counts", [(["aab"], {"a": 2, "b": 1}), ([""], {})]),
    ("t4", "while", "Write `halvings(n)` returning how many times n can be integer-halved before reaching 1.",
     "halvings", [([8], 3), ([1], 0), ([10], 3)]),
    ("t5", "range", "Write `evens(a, b)` returning a list of even numbers in [a, b).",
     "evens", [([1, 7], [2, 4, 6]), ([0, 1], [0])]),
    ("t6", "return", "Write `classify(n)` returning 'neg', 'zero' or 'pos'.",
     "classify", [([-3], "neg"), ([0], "zero"), ([5], "pos")]),
]


def normalize(s: str) -> str:
    return " ".join(str(s).strip().strip("'\"").split()).lower()


def grade_item(item, response: str) -> bool:
    return normalize(response) == normalize(item[3])


def run_tests(task, code: str, timeout_s: float = 2.0) -> tuple[int, int]:
    """Run the learner's code against hidden tests in a subprocess (isolation + timeout)."""
    import json
    import subprocess
    import sys

    _, _, _, fn, tests = task
    harness = code + "\nimport json,sys\nres=[]\n" + "".join(
        f"\ntry:\n    res.append({fn}(*{args!r}) == {exp!r})\nexcept Exception:\n    res.append(False)"
        for args, exp in tests) + "\nprint(json.dumps(res))"
    try:
        out = subprocess.run([sys.executable, "-I", "-c", harness], capture_output=True, text=True,
                             timeout=timeout_s).stdout.strip().splitlines()
        res = json.loads(out[-1]) if out else []
    except Exception:
        res = []
    return sum(bool(r) for r in res), len(tests)
