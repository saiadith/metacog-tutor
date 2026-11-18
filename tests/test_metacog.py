from metacog.classifier import TfidfClassifier, default_classifier, load_seed
from metacog.policy import LearnerState, PolicyConfig, decide, on_submit
from metacog.problems import LEARNING_TASKS, run_tests

# Natural, hand-written messages NOT in the seed set (held-out sanity check)
HELD_OUT = [
    ("can u just write the function", "answer"), ("whats the code for this", "answer"),
    ("why does my loop skip the first element", "explain"), ("what does get() do on a dict", "explain"),
    ("is this correct? def f(x): return x*x", "verify"), ("does my solution handle empty input", "verify"),
    ("my plan is to loop over the chars and count them", "plan"),
    ("what's your favourite movie", "offtask"),
]


def test_classifier_heldout_accuracy():
    clf = default_classifier()
    acc = sum(clf.predict(t) == y for t, y in HELD_OUT) / len(HELD_OUT)
    assert acc >= 0.6, acc


def test_classifier_cv_accuracy():
    from sklearn.model_selection import cross_val_score
    X, y = load_seed()
    assert cross_val_score(TfidfClassifier().pipe, X, y, cv=4).mean() > 0.6


def test_control_never_nudges():
    s = LearnerState()
    assert all(decide(s, "answer", condition="control").action == "answer" for _ in range(5))


def test_offloading_triggers_plan_nudge_then_cooldown():
    s, cfg = LearnerState(), PolicyConfig()
    assert decide(s, "answer", cfg).action == "answer"
    d = decide(s, "answer", cfg)
    assert d.action == "nudge_plan" and "without an own attempt" in d.reason
    assert decide(s, "answer", cfg).action == "answer"  # cooldown


def test_nudge_cap_per_problem():
    s, cfg = LearnerState(), PolicyConfig(nudge_cooldown_turns=0, max_nudges_per_problem=2)
    actions = [decide(s, "answer", cfg).action for _ in range(10)]
    assert sum(a != "answer" for a in actions) == 2


def test_fast_copy_submit_flagged():
    s = LearnerState(ai_gave_solution=True, t_ai_solution=100.0)
    assert on_submit(s, 105.0).action == "nudge_eval"
    assert on_submit(LearnerState(), 105.0) is None


def test_code_runner_and_sandbox_timeout():
    t1 = LEARNING_TASKS[0]
    assert run_tests(t1, "def sum_squares(xs):\n    return sum(x*x for x in xs)") == (3, 3)
    assert run_tests(t1, "def sum_squares(xs):\n    return 0") == (1, 3)
    assert run_tests(t1, "while True: pass") == (0, 3)  # killed by timeout


def test_end_to_end_simulation(tmp_path):
    import sqlite3
    from metacog.simulate import run
    db = run(n=6, seed=1, db_path=str(tmp_path / "s.db"))
    con = sqlite3.connect(db)
    conds = dict(con.execute("SELECT condition, COUNT(*) FROM participants GROUP BY condition").fetchall())
    assert conds == {"control": 3, "metacog": 3}  # balanced randomisation
    kinds = {k for (k,) in con.execute("SELECT DISTINCT kind FROM events")}
    assert {"start", "answer", "chat", "submit", "survey"} <= kinds
    import json
    rows = con.execute("SELECT p.condition, e.kind, e.payload FROM events e JOIN participants p USING(pid)").fetchall()
    def is_nudge(kind, pl):
        pl = json.loads(pl)
        return (kind == "chat" and pl["action"] != "answer") or (kind == "submit" and pl.get("nudge"))
    assert not any(is_nudge(k, pl) for c, k, pl in rows if c == "control")
    assert any(is_nudge(k, pl) for c, k, pl in rows if c == "metacog")
