#!/usr/bin/env python
"""
Benchmark-runner för både Human-Eval (pass@k) och QA/Math-uppgifter.
"""
import argparse, json, requests, random, time, textwrap, re, ast
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

from eval_utils import extract_number, extract_choice, CHOICE2ID

# ────────────────────────── Server-inställningar ──────────────────────────
SERVER  = "http://localhost:8080/v1/completions"
HEADERS = {"Content-Type": "application/json"}

# ---------- prompt-mallar ----------
PROMPT_TEMPLATE_CODE = textwrap.dedent("""\
    ### Instruction:
    You will be given a Python function **header** that ends with the keyword `pass`.
    Replace that `pass` line with working code **and return ONLY the new body** –
    no `def`, no docstring, no tests.

    ### Function header:
    {stub}

    ### Your answer:
    """)

PROMPT_TEMPLATE_QA = textwrap.dedent("""\
    You are a concise problem-solver.
    Answer the following question in **one line** with **only** the final answer.
    If the answer is numeric, output just the number.

    Question:
    {question}

    Answer:
    """)

STOP_TOKENS_CODE = ["###"]      # stop innan modellen påbörjar nästa sektion
STOP_TOKENS_QA   = ["\n\n"]     # sluta vid första blankrad

BODY_STRIP_RE = re.compile(r"```.*?```", re.S)   # ta bort code-fences

# ────────────────────────── Kod-hjälp ──────────────────────────
def merge_stub_and_body(stub: str, body: str) -> str:
    """Ersätt `pass`-raden i stubben med indenterad body-text."""
    if not body.strip():
        body = "    pass"
    header_lines, merged, replaced = stub.splitlines(), [], False
    for line in header_lines:
        if not replaced and line.strip() == "pass":
            indent = re.match(r"\s*", line).group(0)
            merged.append(textwrap.indent(body.rstrip(), indent))
            replaced = True
        else:
            merged.append(line)
    if not replaced:
        merged.append(body)
    code = "\n".join(merged).rstrip()
    if re.search(r"\bList\[", code) and "from typing import List" not in code:
        code = "from typing import List\n\n" + code
    return code

# ────────────────────────── Modell-anrop ───────────────────────
def generate_code_body(stub: str, *, max_tokens=256, temperature=0.3):
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "prompt": PROMPT_TEMPLATE_CODE.format(stub=stub),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": STOP_TOKENS_CODE,
        "stream": False,
    }
    t0 = time.perf_counter()
    txt = requests.post(SERVER, headers=HEADERS, data=json.dumps(payload)
                        ).json()["choices"][0]["text"]
    txt = BODY_STRIP_RE.sub("", txt).strip()
    return txt, round((time.perf_counter() - t0) * 1000, 1)

def generate_qa_answer(question: str, *, max_tokens=32, temperature=0.0):
    payload = {
        "model": "Qwen2.5-0.5B-Instruct",
        "prompt": PROMPT_TEMPLATE_QA.format(question=question),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": STOP_TOKENS_QA,
        "stream": False,
    }
    t0 = time.perf_counter()
    txt = requests.post(SERVER, headers=HEADERS, data=json.dumps(payload)
                        ).json()["choices"][0]["text"]
    return txt.strip(), round((time.perf_counter() - t0) * 1000, 1)

# ────────────────────────── safe pass@k ────────────────────────
def safe_pass(pred_list, ref, metric, key):
    try:
        return metric._predict_one(pred_list, ref)[key]
    except Exception:
        return 0.0

# ────────────────────────── Huvudprogram ───────────────────────
def main(a):
    ds      = load_dataset(a.path, a.config, split=a.split)
    is_code = a.metric.startswith("pass@")
    ids     = random.sample(range(len(ds)), a.limit) if a.limit else range(len(ds))

    if is_code:
        k_val   = int(a.metric.split("@")[1])
        metric  = load("code_eval", k=[k_val])
        passkey = f"pass@{k_val}"
    else:
        metric  = load(a.metric)

    # few-shot för QA (ej kod)
    prefix = ""
    if a.fewshot and not is_code:
        prefix = "\n\n".join(
            f"Q: {ds[i][a.q]}\nA: "
            f"{CHOICE2ID[ds[i][a.a]] if a.metric=='accuracy' else ds[i][a.a]}"
            for i in ids[:a.fewshot]
        ) + "\n\n"

    rows, successes = [], 0.0
    for idx in tqdm(ids, desc=f"{a.path}-{a.config}"):
        ex = ds[idx]

        if is_code:
            stub, gold = ex[a.q], ex[a.a]
            preds, latencies = [], []
            for _ in range(k_val):
                body, lat = generate_code_body(stub, temperature=0.8)
                preds.append(merge_stub_and_body(stub, body))
                latencies.append(lat)
            successes += safe_pass(preds, gold, metric, passkey)
            pred_out, lat_avg = preds[0], sum(latencies)/k_val

        else:
            gold_fn, pred_fn = (
                (lambda e: CHOICE2ID[e[a.a]], extract_choice) if a.metric=="accuracy"
                else (lambda e: extract_number(e[a.a]), extract_number)
            )
            gold = gold_fn(ex)
            q_text = prefix + ex[a.q]
            pred_txt, lat_avg = generate_qa_answer(q_text)
            pred_out = pred_fn(pred_txt)
            metric.add(prediction=pred_out, reference=gold)

        rows.append({"id": idx, "gold": gold, "pred": pred_out,
                     "latency_ms": round(lat_avg, 2)})

    final = successes/len(ids) if is_code else metric.compute()[a.metric]
    print(f"\n[{a.metric}] {final*100:.2f}%   (N={len(ids)})")

    Path("results").mkdir(exist_ok=True)
    csv = f"{a.path.replace('/','_')}_{a.config}_{a.split}_{a.metric}.csv"
    pd.DataFrame(rows).to_csv(Path("results")/csv, index=False)

# ────────────────────────── CLI wiring ─────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path",   required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split",  default="test")
    p.add_argument("--q",      default="question")
    p.add_argument("--a",      default="answer")
    p.add_argument("--metric", default="pass@5")
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--limit",   type=int, default=None)
    main(p.parse_args())
