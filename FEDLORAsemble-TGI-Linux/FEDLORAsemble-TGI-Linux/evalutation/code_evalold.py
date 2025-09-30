#!/usr/bin/env python
"""
Human-Eval pass@k med MoE-routern (tgi_router_client) **eller**
direkt-anrop till en viss adapter.

* Genererar `n` completion-kroppar / uppgift via router/direct
* Kör OpenAI:s officiella test-sviter
* Beräknar unbiased pass@k
"""

import argparse, json, re, textwrap, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter

import numpy as np
import tqdm
from human_eval.data import read_problems, stream_jsonl, HUMAN_EVAL
from human_eval.execution import check_correctness

# ─── Router-klient ───────────────────────────────────────────
import tgi_router_client   # din egen modul

STOP_SEQ = "###"
STOP_RE  = re.compile(r"\n\s*###")
FENCE_RE = re.compile(r"```.*?```", re.S)

PROMPT = textwrap.dedent("""\
You will be given a Python function header that ends with the keyword `pass`.
Replace that line with working code and **return ONLY the new body** –
no def, no markdown, no comments.

### Function header:
{stub}

### Your answer:
""")

# ---------- unbiased formel (vectoriserad) ----------
def pass_at_k(n_arr, c_arr, k):
    n_arr = np.asarray(n_arr, int)
    c_arr = np.asarray(c_arr, int)
    out   = np.zeros_like(n_arr, dtype=float)
    for i, (n, c) in enumerate(zip(n_arr, c_arr)):
        if n < k:          out[i] = 0.0
        elif n - c < k:    out[i] = 1.0
        else:
            denom = np.arange(n-c+1, n+1)
            out[i] = 1.0 - np.prod(1.0 - k/denom)
    return out

# ---------- generation wrappers ----------
def call_router(stub, temp):
    txt, *_ = tgi_router_client.generate(PROMPT.format(stub=stub), temp=temp)
    return txt

def call_direct(stub, adapter_alias, temp):
    payload = {
        "model": tgi_router_client.TGI_MODEL_NAME,
        "prompt": tgi_router_client.EXPERT_CFG_BY_ALIAS[adapter_alias]["preprompt"] + PROMPT.format(stub=stub),
        "max_tokens": 256,
        "temperature": temp,
        "stop": [STOP_SEQ],
    }
    r = requests.post(tgi_router_client.TGI_SERVER_URL,
                      headers=tgi_router_client.TGI_HEADERS,
                      data=json.dumps(payload))
    return r.json()["choices"][0]["text"]

def clean(raw):
    txt = STOP_RE.split(raw, 1)[0]
    return FENCE_RE.sub("", txt).strip() or "pass"

# ---------- main ----------
def main(a):
    problems = read_problems(HUMAN_EVAL)
    task_ids = list(problems.keys())[: a.limit] if a.limit else problems.keys()

    # choose generator
    if a.direct:
        gen_fn = lambda stub: call_direct(stub, a.direct, a.temp)
        mode   = f"direct-{a.direct}"
    else:
        gen_fn = lambda stub: call_router(stub, a.temp)
        mode   = "router"

    samples = Path("samples_router.jsonl")
    print(f"[{mode}] Generating {a.n} × {len(task_ids)} completions…")
    with samples.open("w") as fout:
        for tid in tqdm.tqdm(task_ids):
            stub = problems[tid]["prompt"]
            for _ in range(a.n):
                raw  = gen_fn(stub)
                fout.write(json.dumps({"task_id": tid,
                                       "completion": clean(raw)})+"\n")

    # run tests
    print("Running test-suites…")
    results = defaultdict(list); comp_id = Counter()
    with ThreadPoolExecutor(max_workers=a.workers) as exe:
        futs=[]
        for s in stream_jsonl(str(samples)):
            tid = s["task_id"]; cid = comp_id[tid]
            futs.append(exe.submit(check_correctness,
                                   problems[tid], s["completion"],
                                   a.timeout, cid))
            comp_id[tid]+=1
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            res = fut.result(); results[res["task_id"]].append(res["passed"])

    total = np.array([len(results[tid]) for tid in task_ids])
    corr  = np.array([sum(results[tid]) for tid in task_ids])
    score = pass_at_k(total, corr, a.k).mean()
    print(f"\n[{mode}] pass@{a.k} = {score:.2%}  (n={a.n}, tasks={len(task_ids)})")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="samples per task")
    p.add_argument("--k", type=int, default=10,  help="pass@k")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--direct", type=str, default=None,
                   help="Skip router; call this adapter alias directly (e.g. 'general').")
    main(p.parse_args())
