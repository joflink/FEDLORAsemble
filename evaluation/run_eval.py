#!/usr/bin/env python
"""benchmark_runner.py
─────────────────────────────────────────────────────────────────────────────
Benchmark‑runner that supports **two** invocation modes:

1. **Router mode** – uses *tgi_router_client.generate* (typically a LoRA or
   mixture‑of‑experts router).
2. **Direct mode** – sends the prompt straight to a single Text‑Generation‑
   Inference (TGI) server so you can benchmark the *base* model without any
   routing‑layer logic.

Both modes return a unified tuple so that the downstream evaluation logic is
completely agnostic of the transport layer.

The script supports:
  • QA / MC / GSM‑style tasks with any evaluate‑metric (accuracy, f1, EM …)
  • HumanEval‑style code‑generation problems with pass@k

Only three helpers (CHOICE2ID, extract_choice, extract_number) are imported
from *eval_utils.py*.

CLI example – LoRA router:
    python benchmark_runner.py \
        --path Taskset --config main --metric accuracy --fewshot 5 \
        --mode router --expert_temp 0.0

CLI example – base model via direct TGI:
    python benchmark_runner.py \
        --path Taskset --config main --metric accuracy --fewshot 5 \
        --mode direct

─────────────────────────────────────────────────────────────────────────────"""
import argparse
import os
import random
import re
import time
from pathlib import Path
from typing import List, Callable, Tuple
import json
import pandas as pd
import requests
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

from eval_utils import CHOICE2ID, extract_choice, extract_number
from tgi_router_client import generate as router_generate

Prompt = str
Generated = str
LatencyMs = float
ExpertId = str
RouterLatency = float
LlmLatency = float


# ───────────────────────── Transport abstraction ──────────────────────────
TGI_SERVER_URL = "http://localhost:8080/v1/completions"      # ← same as router
TGI_MODEL_NAME  = "Qwen2.5-7B-Instruct"
TGI_HEADERS     = {"Content-Type": "application/json"}

def _direct_generate(
    prompt: Prompt,
    *,
    temp: float = 0.0,
    max_new_tokens: int | None = None,
) -> Tuple[Generated, RouterLatency, LlmLatency, LatencyMs, ExpertId]:
    """
    Send prompt straight to TGI’s /v1/completions endpoint,
    parse the same JSON shape as router_generate, and return
    (text, router_lat_ms=0.0, llm_lat_ms, total_lat_ms, "direct").
    """
    # build payload just like in your router client
    payload = {
        "model":        TGI_MODEL_NAME,
        "prompt":       prompt,
        "adapter_id":   None,                    # no adapter in direct mode
        "max_tokens":   max_new_tokens or 256,
        "temperature":  temp,
        "stream":       False,
    }

    t0 = time.perf_counter()
    try:
        # note: router uses data=json.dumps, so we do the same
        rsp = requests.post(TGI_SERVER_URL,
                            headers=TGI_HEADERS,
                            data=json.dumps(payload),
                            timeout=60)
        rsp.raise_for_status()
        llm_latency_ms = (time.perf_counter() - t0) * 1_000

        # parse the identical shape: choices -> [{ "text": ... }]
        resp_json = rsp.json()
        text = resp_json["choices"][0]["text"].strip()

        return text, 0.0, llm_latency_ms, llm_latency_ms, "direct"

    except requests.exceptions.RequestException as e:
        print(f"❌ Direct TGI request failed: {e}\nResponse was:\n{rsp.text}")
        return "", 0.0, 0.0, 0.0, "direct"
    except (KeyError, IndexError, ValueError) as e:
        print(f"❌ Could not parse Direct TGI JSON: {e}\nResponse was:\n{rsp.text}")
        return "", 0.0, 0.0, 0.0, "direct"


def make_generate_fn(mode: str) -> Callable[..., Tuple[Generated, RouterLatency, LlmLatency, LatencyMs, ExpertId]]:
    if mode == "router":
        return lambda *args, **kwargs: router_generate(*args, **kwargs)
    elif mode == "direct":
        return lambda prompt, **kw: _direct_generate(prompt, **kw)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'router' or 'direct'.")


def _safe_choice_label(x):
    x = str(x).strip()
    return CHOICE2ID.get(x, -1)


def _strip_code_fences(txt: str) -> str:
    if "```" not in txt:
        return txt.strip()
    m = re.findall(r"```[\w]*\n(.*?)```", txt, flags=re.S)
    return m[0].strip() if m else txt.strip()


def _keep_first_def(txt: str) -> str:
    i = txt.find("def ")
    if i == -1:
        return txt.strip()
    snippet = txt[i:]
    lines: List[str] = []
    indent0 = None
    for line in snippet.splitlines():
        if line.startswith("def ") and indent0 is not None:
            break
        if indent0 is None and line.startswith("def "):
            indent0 = len(line) - len(line.lstrip())
        lines.append(line)
    return "\n".join(lines).rstrip()


def _pass_at_k_single(pred_list, ref, k_val: int) -> float:
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    metric = load("code_eval")
    scores, _ = metric.compute(predictions=[pred_list], references=[ref], k=[k_val])
    return scores[f"pass@{k_val}"]


def main(a):
    ds = load_dataset(a.path, a.config, split=a.split)
    is_code = a.metric.startswith("pass@")
    k_val = int(a.metric.split("@")[1]) if is_code else None

    if a.metric == "accuracy":
        gold_fn = lambda ex: _safe_choice_label(ex[a.a])
        pred_fn = extract_choice
    elif is_code:
        gold_fn, pred_fn = (lambda ex: ex[a.a], _keep_first_def)
    else:
        gold_fn, pred_fn = (
            lambda ex: extract_number(ex[a.a]),
            extract_number,
        )

    prefix = ""
    if a.fewshot and not is_code:
        ids_fs = random.sample(range(len(ds)), a.fewshot)
        prefix = "\n\n".join(
            f"Q: {ds[i][a.q]}\nA: {gold_fn(ds[i])}" for i in ids_fs
        ) + "\n\n"

    metric = None if is_code else load(a.metric)
    ids = random.sample(range(len(ds)), a.limit) if a.limit else range(len(ds))

    rows = []
    successes = 0.0
    generate_fn = make_generate_fn(a.mode)

    bar_desc = f"{a.path}-{a.config} {a.mode.upper()}"
    for idx in tqdm(ids, desc=bar_desc):
        ex = ds[idx]
        gold = gold_fn(ex)

        if is_code:
            stub = ex[a.q]
            preds, lats, eids = [], [], []
            for _ in range(k_val):
                body, router_lat, llm_lat, lat, eid = generate_fn(stub, temp=0.8)
                body = _strip_code_fences(body)
                preds.append(body)
                lats.append(lat)
                eids.append(eid)
            successes += _pass_at_k_single(preds, gold, k_val)
            pred_out = preds[0]
            lat_avg = sum(lats) / k_val
            eid_str = ",".join(map(str, eids))
        else:
           # ── QA / GSM8K  ─────────────────────────────────────────────
            if a.metric == "accuracy":
                # klassiskt multiple-choice (ARC/MMLU m.m.)
                prompt = (
                    prefix
                    + f"Q: {ex[a.q]}\n"
                      "Respond with **only** the number 0, 1, 2, or 3."
                )
            else:  # exact_match  → t.ex. GSM8K (frisvar, numeriskt facit)
                prompt = (
                    prefix
                    + "You are a careful mathematician.  "
                      "Solve the problem step-by-step **internally** and then "
                      "respond with only the final numeric answer—no words.\n\n"
                    f"Problem: {ex[a.q]}\nAnswer:"
                )
            pred_txt, router_lat, llm_lat, lat_avg, eid_str = generate_fn(prompt, temp=a.expert_temp)
            pred_out = pred_fn(pred_txt)
            metric.add(prediction=pred_out, reference=gold)
        csv_pred = "ABCD"[pred_out] if a.metric == "accuracy" else pred_out
        rows.append({
            "id": idx,
            "gold": "ABCDE"[gold] if a.metric == "accuracy" else gold,
            "pred": csv_pred,
            "latency_ms": round(lat_avg, 1),
            "expert_id": eid_str,
            "router_latency_ms": round(router_lat, 1),
            "llm_latency_ms": round(llm_lat, 1),
        })

    final = (successes / len(ids)) if is_code else metric.compute()[a.metric]
    print(f"\n[{a.mode.upper()} {a.metric}] {final * 100:.2f}%   N={len(ids)}")

    Path("results").mkdir(exist_ok=True)
    csv_name = f"{a.path.replace('/', '_')}_{a.config}_{a.split}_{a.mode.upper()}_{a.metric}.csv"
    pd.DataFrame(rows).to_csv(Path("results") / csv_name, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark LoRA router vs base model via TGI")
    p.add_argument("--path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--q", default="question")
    p.add_argument("--a", default="answer")
    p.add_argument("--metric", default="exact_match")
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mode", choices=["router", "direct"], default="router")
    p.add_argument("--expert_temp", type=float, default=0.0)
    main(p.parse_args())
