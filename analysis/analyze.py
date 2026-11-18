# %% [markdown]
# Study analysis: learning gain, calibration, offloading, nudge behaviour.
# Usage: python analysis/analyze.py data/study.db   (or data/sim_study.db)

# %%
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower

db = sys.argv[1] if len(sys.argv) > 1 else "data/study.db"
out = Path("analysis/out"); out.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(db)
P = pd.read_sql("SELECT * FROM participants", con)
E = pd.read_sql("SELECT * FROM events", con)
E = pd.concat([E.drop(columns="payload"), pd.json_normalize(E.payload.map(json.loads))], axis=1)
E = E.drop(columns=[c for c in ["condition"] if c in E.columns])  # use participants.condition

# %% learning outcomes (pre/post, no AI)
ans = E[E.kind == "answer"].copy()
ans["correct"] = ans.correct.astype(float)
scores = ans.pivot_table(index="pid", columns="phase", values="correct", aggfunc="mean").reset_index()
df = P.merge(scores, on="pid").rename(columns={"pre": "pre", "post": "post"})
df["gain"] = df.post - df.pre
df["norm_gain"] = np.where(df.pre < 1, (df.post - df.pre) / (1 - df.pre), 0)
ancova = smf.ols("post ~ C(condition, Treatment('control')) + pre", data=df).fit()
summary = {"n": int(len(df)), "by_condition": df.groupby("condition")[["pre", "post", "gain", "norm_gain"]].mean().round(3).to_dict()}
coef = "C(condition, Treatment('control'))[T.metacog]"
summary["ancova_metacog_effect"] = {"b": round(ancova.params[coef], 3), "p": round(ancova.pvalues[coef], 4),
                                     "ci95": [round(x, 3) for x in ancova.conf_int().loc[coef]]}

# %% calibration on post-test (confidence vs correctness)
post = ans[ans.phase == "post"].dropna(subset=["confidence"]).copy()
post["conf"] = post.confidence / 100
post["correct"] = post.correct.astype(float)
post = post.merge(P[["pid", "condition"]], on="pid")
post["brier"] = (post.conf - post.correct) ** 2
post["overconf"] = post.conf - post.correct
summary["calibration"] = post.groupby("condition")[["brier", "overconf"]].mean().round(3).to_dict()

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for cond, g in post.groupby("condition"):
    bins = pd.cut(g.conf, np.linspace(0, 1, 6), include_lowest=True)
    c = g.groupby(bins, observed=True).agg(conf=("conf", "mean"), acc=("correct", "mean"))
    ax.plot(c.conf, c.acc, "o-", label=cond)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="Confidence", ylabel="Accuracy", title="Post-test calibration"); ax.legend()
fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150)

# %% help-seeking and policy behaviour during learning
chat = E[E.kind == "chat"].merge(P[["pid", "condition"]], on="pid")
chat["turn"] = chat.groupby("pid").cumcount()
chat["answer_seeking"] = chat.label == "answer"
summary["answer_seeking_rate"] = chat.groupby("condition").answer_seeking.mean().round(3).to_dict()
summary["actions_metacog"] = chat[chat.condition == "metacog"].action.value_counts().to_dict()
fig, ax = plt.subplots(figsize=(6, 3.5))
for cond, g in chat.groupby("condition"):
    r = g.groupby(pd.cut(g.turn, 6), observed=True).answer_seeking.mean()
    ax.plot(range(len(r)), r.values, "o-", label=cond)
ax.set(xlabel="Session segment", ylabel="Share answer-seeking", title="Offloading over the session"); ax.legend()
fig.tight_layout(); fig.savefig(out / "offloading.png", dpi=150)

# %% power: what effect can n detect?
pw = TTestIndPower()
summary["power"] = {"n_per_group_for_d0.5_80pct": round(pw.solve_power(0.5, power=0.8, alpha=0.05)),
                    "mde_d_at_current_n": round(pw.solve_power(nobs1=max(2, len(df) // 2), power=0.8, alpha=0.05), 2)}

(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
