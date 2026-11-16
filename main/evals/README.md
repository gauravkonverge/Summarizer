# Latency and model evaluation

Scripts that call live AWS Bedrock (Guardrail + models) using the credentials in `.env`.
Each run makes billable calls. Run them from the `main/` folder:

| Script | What it does | Run |
|---|---|---|
| `bench.py` | Guardrail sequential vs parallel timing, then one full pipeline request | `PYTHONPATH=. .venv/Scripts/python.exe evals/bench.py evals/rac_sample.json` |
| `models_bench.py` | Summary/verifier latency for ~12 Bedrock models on one conversation | `PYTHONPATH=. .venv/Scripts/python.exe evals/models_bench.py evals/rac_sample.json evals/models_results.json` |
| `screen_models.py` | One summary call per model (Haiku 4.5/5.5 variants, gpt-oss, GPT-5.x/6.x) for a quick latency screen | `PYTHONPATH=. .venv/Scripts/python.exe evals/screen_models.py evals/rac_sample.json` |
| `eval_models.py` | 5 candidate models x 6 samples; speed plus quality graded by Claude Opus 4.5 against planted key facts and traps | `PYTHONPATH=. .venv/Scripts/python.exe evals/eval_models.py evals/eval_samples.json evals/eval_results.json` |

Set `CLOUDWATCH_EMF_ENABLED=false` in the environment to keep metric lines out of the output.

Data files:

- `eval_samples.json`: 6 synthetic conversations, each with `key_facts` and `traps`.
- `rac_sample.json`: the 27-message RAC repair conversation.
- `*_results.json`: output from the runs on 2026-11-04. `eval_results.json` (Sonnet 4.5, Haiku 4.5,
  Nova, Llama) was produced before the judge rules and S2/S4/S5 facts were corrected, so it is not
  directly comparable with `eval_results_gpt.json` (Haiku 4.5/5.5 and GPT models).
- Edit `CANDIDATES` in `eval_models.py` to evaluate other models; per-model Converse
  `additionalModelRequestFields` (e.g. Haiku 5.5 effort) go in the third tuple element.

Known limitation: the LLM judge is sometimes too strict (for example, it counts a `[CREDIT_CARD]`
placeholder as a PII leak), so read the per-sample notes in the results before trusting the scores.
