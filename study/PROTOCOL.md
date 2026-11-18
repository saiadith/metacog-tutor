# Study protocol

## Phase 0: participatory pilot (3–5 learners, think-aloud)
1. Run the app in the metacog condition. Ask learners to think aloud, and record the screen and audio with consent.
2. After each task, ask: *"When the assistant pushed back, what did you think it wanted? Was it right to?"*
3. Log every design change in `DESIGN_CHANGES.md` (date, observation, change, rationale).
   This co-design trail is a first-class contribution of the project.
4. Hand-label all pilot chat messages with the 5 help-seeking labels. Use two raters on a 30% overlap,
   report Cohen's κ, and retrain the classifier on the real data.

## Phase 1: main study (recommended: within-subject)
- Each learner completes 6 tasks: 3 concepts with the plain assistant, 3 with the metacognitive layer.
  The concept-to-condition assignment and the order are counterbalanced (Latin square).
- Primary outcome: post-test accuracy on concepts learned with metacog vs. control
  (mixed-effects logistic regression with random intercepts for learner and item).
- Secondary: calibration (Brier score, over-confidence), answer-seeking rate, time on task,
  perceived autonomy, and qualitative themes from the open responses.
- Target: about 40 learners gives about 240 concept-level observations per condition.

## Interview guide (10 min, after the session)
1. Tell me about a moment the assistant didn't just give you the answer. What happened next?
2. Did you ever feel blocked or patronised? When?
3. How confident are you that you could solve the last task alone tomorrow? Why?
4. If you could change one rule about when the assistant pushes back, what would it be?

## Ethics and data
- Consent text is in the app. Participants get random IDs, no names are stored, data is deleted on request.
- Chat logs can contain personal text, so review them before sharing any excerpt.
- Studies with human participants usually need ethics review at your institution: check before recruiting.
