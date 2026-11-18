"""Offloading detector + metacognitive nudge policy.

Grounded in self-regulated learning (Zimmerman 2002; Winne & Hadwin 1998):
  forethought/planning -> performance/monitoring -> self-reflection/evaluation.
The policy decides, per learner message, one of:
  answer        - normal assistant reply
  nudge_plan    - ask the learner for their approach before helping
  nudge_monitor - ask for confidence / to locate the uncertain step
  nudge_eval    - after an AI solution: ask the learner to explain it back
  argue_back    - present a plausible but flawed solution to critique
  teach_back    - learner explains to a confused "teachable agent"
Every decision is logged with its reason (transparency for the study and for learners).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyConfig:
    offload_streak: int = 2          # consecutive answer-seeking turns w/o own attempt
    fast_submit_s: float = 20.0      # submitting soon after an AI solution = likely copy
    nudge_cooldown_turns: int = 2    # never nudge on consecutive turns
    max_nudges_per_problem: int = 3  # hard cap: avoid becoming a gatekeeper


@dataclass
class LearnerState:
    answer_streak: int = 0
    turns: int = 0
    last_nudge_turn: int = -99
    nudges_this_problem: int = 0
    attempted: bool = False          # has the learner written/submitted own code?
    ai_gave_solution: bool = False
    t_ai_solution: float | None = None
    history: list = field(default_factory=list)

    def new_problem(self):
        self.answer_streak = 0
        self.nudges_this_problem = 0
        self.attempted = False
        self.ai_gave_solution = False
        self.t_ai_solution = None


@dataclass
class Decision:
    action: str
    reason: str


def offloading_signals(s: LearnerState, cfg: PolicyConfig, label: str) -> list[str]:
    sig = []
    if label == "answer" and s.answer_streak >= cfg.offload_streak and not s.attempted:
        sig.append(f"{s.answer_streak} answer-seeking turns without an own attempt")
    if label == "answer" and s.ai_gave_solution:
        sig.append("asking for another solution after already receiving one")
    return sig


def decide(s: LearnerState, label: str, cfg: PolicyConfig = PolicyConfig(), condition: str = "metacog") -> Decision:
    s.turns += 1
    s.history.append(label)
    if label == "answer":
        s.answer_streak += 1
    else:
        s.answer_streak = 0

    if condition == "control":
        return Decision("answer", "control condition")

    can_nudge = (s.turns - s.last_nudge_turn > cfg.nudge_cooldown_turns
                 and s.nudges_this_problem < cfg.max_nudges_per_problem)
    sig = offloading_signals(s, cfg, label)

    def nudge(action, reason):
        s.last_nudge_turn = s.turns
        s.nudges_this_problem += 1
        return Decision(action, reason)

    if can_nudge and sig:
        if not s.attempted:
            return nudge("nudge_plan", "; ".join(sig))
        if s.ai_gave_solution:
            # learner has some code + already saw a solution: make them evaluate one
            return nudge("argue_back", "; ".join(sig))
        return nudge("nudge_monitor", "; ".join(sig))
    if label == "verify" and can_nudge and s.nudges_this_problem == 0:
        return nudge("nudge_monitor", "verification request: ask learner to locate their own uncertainty first")
    if label == "explain" and s.ai_gave_solution and can_nudge:
        return nudge("teach_back", "learner seeking explanation after solution: consolidate via teaching")
    return Decision("answer", "no offloading signal" if not sig else "nudge on cooldown")


def on_submit(s: LearnerState, t_now: float, cfg: PolicyConfig = PolicyConfig()) -> Decision | None:
    """Called when the learner submits a solution. Flags likely copy-without-understanding."""
    s.attempted = True
    if s.ai_gave_solution and s.t_ai_solution is not None and t_now - s.t_ai_solution < cfg.fast_submit_s:
        return Decision("nudge_eval", f"submitted {t_now - s.t_ai_solution:.0f}s after AI solution")
    return None


NUDGE_TEXT = {
    "nudge_plan": "Before I help — what would your first step be? Even a rough idea is fine.",
    "nudge_monitor": "Which line or step are you least sure about, and how confident are you (0–100)?",
    "nudge_eval": "Nice. Can you explain in one sentence *why* this works, without looking at my answer?",
}
