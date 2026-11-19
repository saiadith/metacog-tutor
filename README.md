# metacog-tutor

When students learn with an AI assistant, it's easy to just ask for the answer and learn nothing. This project adds a small metacognitive layer on top of an LLM tutor: it notices when someone is offloading their thinking and nudges them to plan, check their work or explain things back, instead of handing over another solution. The real test is what people can do afterwards without the AI.

## How it works

Each student message gets classified as answer-seeking, asking for an explanation, checking their work, planning, or off-task. A simple policy keeps track of the session (have they tried anything themselves, did the AI already give a solution, how many nudges so far) and picks a response:

- a normal answer
- "what would your first step be?" if they keep asking for the answer without trying
- "which part are you least sure about?" when they ask the AI to check their work
- argue-back: the AI shows a slightly broken solution and asks them to find the bug
- teach-back: they explain the idea to a confused AI peer

There's a cooldown and a cap of 3 nudges per problem, so it never blocks help completely. Every decision is logged with the reason.

## What's in here

- a study web app (FastAPI + one HTML page): consent, a short quiz, six Python coding tasks with the assistant, a second quiz with confidence ratings, and a survey
- the classifier (TF-IDF + logistic regression, saved in models/)
- a simulator that runs fake participants through the real app to check the whole pipeline end to end
- analysis: learning gain, calibration, how often people offload over time, and a power analysis
- a study protocol and interview guide in study/
- demo/index.html, a standalone page you can open in a browser to see both assistants side by side

One thing the simulation made clear: with 40 people you can only detect a pretty big effect. For a moderate one you'd want around 64 per group, or a design where everyone tries both versions.

## Running it

```
pip install -r requirements.txt
uvicorn metacog.app:app --reload
python -m metacog.simulate --n 40 --db data/sim_study.db
python analysis/analyze.py data/sim_study.db
pytest -q
```

It uses a mock LLM by default. To use a real one, set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL for any OpenAI-compatible endpoint.

## Limitations

The classifier was trained on a small, template-made set of examples, so it should be retrained on real labelled messages from a pilot. The quizzes only test near transfer.
