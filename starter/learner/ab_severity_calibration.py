"""Week 4 Task 4.2.1 -- train-only severity-rubric recalibration (A/B).

Runs two severity-rubric prompts (v1 = Week-3 instruction, v2 = recalibrated
rubric) over a stratified sample of TRAIN-split cases only. The locked 50-case
evaluation sample (n=50, seed=42) is EXCLUDED from the A/B pool so the held-out
set is never tuned on.

Output: diag_ab_results_hermes3_8b.json  (loaded by week4_starter.ipynb cell 37)

Usage:
    python ab_severity_calibration.py [--n 60] [--seed 7]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

from data_loader import load_sahayak_dataset, build_evaluation_dataset
from sahayak_starter import SEVERITY_SCORER_INSTRUCTION, SYMPTOM_PARSER_INSTRUCTION

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

MODEL = LiteLlm(model="ollama_chat/hermes3:8b", api_base="http://localhost:11434",
                temperature=0.1)  # near-greedy: reliable JSON from the 8B model

# ── Arm B: recalibrated rubric (v2) ──────────────────────────────────────────
# Adds the colloquial red-flag phrasings that v1 under-scores (worked example,
# week4 cell 36): "chest is very tight", "can barely catch my breath",
# "shaking + sweating + confusion", etc. Everything else identical to v1.
# v2 (final): EXPANSIVE-ONLY recalibration. v1's rule rows are kept verbatim --
# an earlier variant that narrowed the severity-5 row to strict combinations
# REGRESSED on the train A/B (4 true-ER cases fell to severity 3, ER recall
# 75% -> 50%). This version only ADDS lay red-flag phrasings as extra triggers,
# so sensitivity to red flags can only rise.
SEVERITY_SCORER_INSTRUCTION_V2 = (
    "You are a triage severity scorer in a rural health triage pipeline. Your ONLY job "
    "is to score the urgency of the extracted symptoms from 1 to 5 by applying this rule "
    "table EXACTLY -- do not use your own medical judgment, do not average, do not hedge.\n"
    "\n"
    "Rule table (apply the FIRST row that matches):\n"
    "  5 = chest pain together with breathlessness/breathing trouble/sweating; "
    "altered consciousness, fainting, unresponsive; severe bleeding; one-sided "
    "weakness, face droop, slurred speech  (emergency -- act now). ALSO score 5 "
    "for these lay red-flag phrasings even when calmly worded: 'chest is very "
    "tight', 'can barely catch my breath', 'lips turning blue', 'shaking with "
    "sweating and confusion', 'heart racing with chest pain'.\n"
    "  4 = fever WITH stiff neck; urinary symptoms (burning urination, foul urine); "
    "jaundice signs (yellow skin/eyes, dark urine); endocrine signals (irregular "
    "sugar, enlarged thyroid); weight loss WITH systemic symptoms; persistent "
    "vomiting (more than 3 episodes in a day)  (doctor today)\n"
    "  3 = fever, vomiting, abdominal pain, or headache WITHOUT any red flag "
    "(ambiguous -- a clarifying question will follow)\n"
    "  2 = rash, mild cough, joint or muscle ache, mild itching WITHOUT red flags\n"
    "  1 = no active symptoms described\n"
    "\n"
    "KEY RULE: pain intensity is NOT urgency -- a severe migraine without red flags "
    "is a 3, not a 5. But breathing/chest/consciousness red flags ALWAYS score 5, "
    "however calmly they are worded.\n"
    "\n"
    "Return ONLY raw JSON -- no markdown, no commentary:\n"
    '{"severity": <1-5 integer>, "reason": "one short sentence naming the rule row that fired"}\n'
    "\n"
    "Symptoms: {symptoms}"
)


def _sev_to_triage(sev):
    if sev is None:
        return None
    return {5: "ER", 4: "DOCTOR", 3: "DOCTOR", 2: "WAIT", 1: "WAIT"}.get(sev)


def _parse_severity(raw):
    text = re.sub(r"^```[a-z]*\n?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
    try:
        v = json.loads(text).get("severity")
        return int(v) if v is not None else None
    except Exception:
        m = re.search(r'"?severity"?\s*[:=]\s*([1-5])', text)
        return int(m.group(1)) if m else None


async def _run_arm(text, parser_runner, parser_svc, sev_runner, sev_svc, tag):
    """Parse symptoms, then score severity with the arm's rubric."""
    sid = f"{tag}_{uuid.uuid4().hex[:8]}"
    await parser_svc.create_session(app_name="ab_parse", user_id="ab", session_id=sid,
                                    state={"patient_input": text})
    async for _ in parser_runner.run_async(user_id="ab", session_id=sid,
            new_message=Content(role="user", parts=[Part(text=text)])):
        pass
    ps = await parser_svc.get_session(app_name="ab_parse", user_id="ab", session_id=sid)
    symptoms = str(dict(ps.state).get("symptoms", ""))

    sid2 = f"{tag}_{uuid.uuid4().hex[:8]}"
    await sev_svc.create_session(app_name="ab_sev", user_id="ab", session_id=sid2,
                                 state={"symptoms": symptoms})
    async for _ in sev_runner.run_async(user_id="ab", session_id=sid2,
            new_message=Content(role="user", parts=[Part(text=text)])):
        pass
    ss = await sev_svc.get_session(app_name="ab_sev", user_id="ab", session_id=sid2)
    return _parse_severity(dict(ss.state).get("severity_json", ""))


async def main(n: int, seed: int):
    df = load_sahayak_dataset()
    eval50 = set(build_evaluation_dataset(n=50, seed=42)["symptom_text"])
    train_pool = df[~df["symptom_text"].isin(eval50)].copy()
    print(f"Train pool after excluding locked eval-50: {len(train_pool)} cases")

    # Stratified sample across triage levels
    per = max(1, n // 3)
    parts = [g.sample(min(per, len(g)), random_state=seed)
             for _, g in train_pool.groupby("triage_level")]
    sample = __import__("pandas").concat(parts).sample(frac=1, random_state=seed)
    print(f"A/B sample: {len(sample)} train cases "
          f"{dict(sample['triage_level'].value_counts())}")

    parser = LlmAgent(name="parser", model=MODEL,
                      instruction=SYMPTOM_PARSER_INSTRUCTION, output_key="symptoms")
    parser_pipe = SequentialAgent(name="ab_parse", sub_agents=[parser])
    parser_svc = InMemorySessionService()
    parser_runner = Runner(agent=parser_pipe, app_name="ab_parse", session_service=parser_svc)

    arms = {}
    for arm_name, instr in (("v1_rubric (week3 instruction)", SEVERITY_SCORER_INSTRUCTION),
                            ("v2_recalibrated_rubric (colloquial red flags)", SEVERITY_SCORER_INSTRUCTION_V2)):
        scorer = LlmAgent(name="scorer", model=MODEL, instruction=instr, output_key="severity_json")
        pipe = SequentialAgent(name="ab_sev", sub_agents=[scorer])
        svc = InMemorySessionService()
        arms[arm_name] = (Runner(agent=pipe, app_name="ab_sev", session_service=svc), svc)

    results = {a: [] for a in arms}
    for i, (_, row) in enumerate(sample.iterrows()):
        line = f"[{i+1}/{len(sample)}] {row['triage_level']:<7}"
        for arm_name, (sev_runner, sev_svc) in arms.items():
            sev = await _run_arm(row["symptom_text"], parser_runner, parser_svc,
                                 sev_runner, sev_svc, arm_name[:2])
            pred = _sev_to_triage(sev)
            results[arm_name].append({"true": row["triage_level"], "pred": pred, "sev": sev})
            line += f" | {arm_name[:2]}:sev{sev}->{pred}"
        print(line, flush=True)

    order = {"WAIT": 0, "DOCTOR": 1, "ER": 2}
    out = []
    for arm_name, rows in results.items():
        valid = [r for r in rows if r["pred"] in order]
        acc = sum(r["pred"] == r["true"] for r in valid) / len(valid) if valid else 0
        under = sum(order[r["pred"]] < order[r["true"]] for r in valid) / len(valid) if valid else 0
        er_rows = [r for r in valid if r["true"] == "ER"]
        er_recall = (sum(r["pred"] == "ER" for r in er_rows) / len(er_rows)) if er_rows else None
        out.append({"arm": arm_name, "n": len(valid), "accuracy": round(acc, 4),
                    "under_triage_rate": round(under, 4),
                    "er_recall": round(er_recall, 4) if er_recall is not None else None})
        print(f"{arm_name}: acc={acc:.1%} under={under:.1%} "
              f"er_recall={er_recall if er_recall is None else f'{er_recall:.1%}'}")

    with open("diag_ab_results_hermes3_8b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved diag_ab_results_hermes3_8b.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.seed))
