"""Study web app (FastAPI + SQLite). Run: uvicorn metacog.app:app --reload

Flow: consent -> pre-test (no AI) -> 6 learning tasks with AI assistant -> post-test
(no AI, confidence 0-100 per item) -> survey. Condition (control | metacog) is assigned
by balanced randomisation. Every event is logged.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .classifier import default_classifier
from .llm import LLM, system_prompt
from .policy import NUDGE_TEXT, LearnerState, decide, on_submit
from .problems import LEARNING_TASKS, POST_TEST, PRE_TEST, grade_item, run_tests

DB = Path(__file__).resolve().parent.parent / "data" / "study.db"
app = FastAPI(title="metacog-study")
llm = LLM()
STATES: dict[str, LearnerState] = {}


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS participants(pid TEXT PRIMARY KEY, condition TEXT, t REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS events(pid TEXT, t REAL, phase TEXT, kind TEXT, payload TEXT)""")
    return con


def log(pid, phase, kind, **payload):
    with db() as con:
        con.execute("INSERT INTO events VALUES (?,?,?,?,?)", (pid, time.time(), phase, kind, json.dumps(payload)))


def assign_condition() -> str:
    with db() as con:
        counts = dict(con.execute("SELECT condition, COUNT(*) FROM participants GROUP BY condition").fetchall())
    c, m = counts.get("control", 0), counts.get("metacog", 0)
    return random.choice(["control", "metacog"]) if c == m else ("control" if c < m else "metacog")


class Start(BaseModel):
    consent: bool


class Answer(BaseModel):
    pid: str
    item_id: str
    response: str
    confidence: int | None = None


class Chat(BaseModel):
    pid: str
    task_id: str
    message: str


class Submit(BaseModel):
    pid: str
    task_id: str
    code: str


@app.post("/start")
def start(s: Start):
    if not s.consent:
        raise HTTPException(400, "consent required")
    pid = uuid.uuid4().hex[:10]
    cond = assign_condition()
    with db() as con:
        con.execute("INSERT INTO participants VALUES (?,?,?)", (pid, cond, time.time()))
    STATES[pid] = LearnerState()
    log(pid, "consent", "start", condition=cond)
    strip = lambda items: [{"id": i[0], "prompt": i[2]} for i in items]
    return {"pid": pid, "pre_test": strip(PRE_TEST), "tasks": [{"id": t[0], "prompt": t[2]} for t in LEARNING_TASKS],
            "post_test": strip(POST_TEST)}


@app.post("/answer")
def answer(a: Answer):
    item = {i[0]: i for i in PRE_TEST + POST_TEST}.get(a.item_id)
    if not item:
        raise HTTPException(404)
    correct = grade_item(item, a.response)
    log(a.pid, "pre" if a.item_id.startswith("pre") else "post", "answer", item=a.item_id,
        concept=item[1], correct=correct, confidence=a.confidence)
    return {"recorded": True}  # no correctness feedback during tests


def condition_of(pid):
    with db() as con:
        row = con.execute("SELECT condition FROM participants WHERE pid=?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown participant")
    return row[0]


@app.post("/chat")
def chat(c: Chat):
    cond = condition_of(c.pid)
    st = STATES.setdefault(c.pid, LearnerState())
    label = default_classifier().predict(c.message)
    d = decide(st, label, condition=cond)
    task = next(t for t in LEARNING_TASKS if t[0] == c.task_id)
    ctx = f"Current task: {task[2]}"
    if d.action in NUDGE_TEXT:
        reply = NUDGE_TEXT[d.action]
    else:
        mode = {"argue_back": "argue_back", "teach_back": "teach_back"}.get(d.action, "assistant")
        reply = llm.complete([{"role": "system", "content": system_prompt(mode) + "\n" + ctx},
                              {"role": "user", "content": c.message}])
        if d.action == "answer" and "def " in reply:
            st.ai_gave_solution, st.t_ai_solution = True, time.time()
    log(c.pid, "learn", "chat", task=c.task_id, message=c.message, label=label,
        action=d.action, reason=d.reason, reply=reply)
    return {"reply": reply, "action": d.action, "why": d.reason if cond == "metacog" else None}


@app.post("/submit")
def submit(s: Submit):
    st = STATES.setdefault(s.pid, LearnerState())
    task = next(t for t in LEARNING_TASKS if t[0] == s.task_id)
    passed, total = run_tests(task, s.code)
    d = on_submit(st, time.time()) if condition_of(s.pid) == "metacog" else None
    log(s.pid, "learn", "submit", task=s.task_id, passed=passed, total=total,
        nudge=d.action if d else None)
    if passed == total:
        st.new_problem()
    return {"passed": passed, "total": total, "nudge": NUDGE_TEXT.get(d.action) if d else None}


@app.post("/survey")
def survey(payload: dict):
    log(payload.get("pid"), "survey", "survey", **{k: v for k, v in payload.items() if k != "pid"})
    return {"thanks": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).resolve().parent.parent / "app" / "index.html").read_text()
