#!/usr/bin/env python
"""
MBPP pass@k via TGI-router ELLER direkt-adapter.
"""

import argparse, json, re, textwrap, subprocess, sys, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import requests, tqdm
from datasets import load_dataset

import tgi_router_client  # din router-klient

# ───── TGI-inställningar ────────────────────────────────────────────
SERVER  = tgi_router_client.TGI_SERVER_URL
HEADERS = tgi_router_client.TGI_HEADERS

STOP_SEQ = "###"
STOP_RE  = re.compile(r"\n\s*###")
FENCE_RE = re.compile(r"```.*?```", re.S)

PROMPT = textwrap.dedent("""\
You will be given a Python function header that ends with the keyword `pass`.
Replace that line with working code and **return ONLY the new body** – no def,
no markdown, no comments.

### Function header:
{stub}

### Your answer:
""")

# ───── Ladda MBPP från HF ───────────────────────────────────────────
mbpp = load_dataset("Muennighoff/mbpp", split="test", trust_remote_code=True)
PROBLEMS = {
    i: {"prompt": ex["text"],
        "test_code": f'{ex["test_setup_code"]}\n\n' + "\n".join(ex["test_list"])}
    for i, ex in enumerate(mbpp)
}

# ───── Hjälp-funktioner  ───────────────────────────────────────────
def pass_at_k(n_arr, c_arr, k):
    n_arr = np.asarray(n_arr); c_arr = np.asarray(c_arr)
    out   = np.zeros_like(n_arr, float)
    for i, (n, c) in enumerate(zip(n_arr, c_arr)):
        if n < k:          out[i] = 0.0
        elif n - c < k:    out[i] = 1.0
        else:
            denom = np.arange(n - c + 1, n + 1)
            out[i] = 1.0 - np.prod(1.0 - k / denom)
    return out

def clean(raw: str) -> str:
    return FENCE_RE.sub("", STOP_RE.split(raw, 1)[0]).strip() or "pass"

def merge(stub: str, body: str) -> str:
    out, done = [], False
    for ln in stub.splitlines():
        if not done and ln.strip() == "pass":
            indent = re.match(r"\s*", ln).group(0)
            out.append(textwrap.indent(body.rstrip(), indent)); done = True
        else:
            out.append(ln)
    if not done:
        out.append(body)
    return "\n".join(out)

def run_pytest(code_str: str, test_code: str, timeout: float) -> bool:
    """Kör pytest på temporär fil; True om alla tester passerar."""
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / "solution.py"
        fpath.write_text(code_str + "\n\n" + test_code)
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(fpath)],
                capture_output=True, timeout=timeout
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

def stream_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

# ───── LLM-anrop ───────────────────────────────────────────────────
def call_router(stub, temp):
    txt, *_ = tgi_router_client.generate(PROMPT.format(stub=stub), temp=temp)
    return txt

def call_direct(stub, adapter_alias, temp):
    cfg = tgi_router_client.EXPERT_CFG[adapter_alias]
    payload = {
        "model": tgi_router_client.TGI_MODEL_NAME,
        "prompt": cfg[1] + PROMPT.format(stub=stub),
        "max_tokens": cfg[2],
        "temperature": temp,
        "stop": [STOP_SEQ],
    }
    r = requests.post(SERVER, headers=HEADERS, data=json.dumps(payload))
    return r.json()["choices"][0]["text"]

# ───── main ───────────────────────────────────────────────────────
def main(args):
    task_ids = list(PROBLEMS.keys())[: args.limit] if args.limit else PROBLEMS.keys()

    # välj generator
    if args.direct:
        gen_fn = lambda stub: call_direct(stub, args.direct, args.temp)
        mode   = f"direct:{args.direct}"
    else:
        gen_fn = lambda stub: call_router(stub, args.temp)
        mode   = "router"

    samples = Path("mbpp_samples.jsonl")
    print(f"[{mode}] Generating {args.n} completions × {len(task_ids)} tasks…")
    with samples.open("w") as f_out:
        for tid in tqdm.tqdm(task_ids):
            stub = PROBLEMS[tid]["prompt"]
            for _ in range(args.n):
                body = clean(gen_fn(stub))
                f_out.write(json.dumps({"task_id": tid, "body": body}) + "\n")

    # Kör tester
    print("Running unit tests…")
    passed = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sample in stream_jsonl(samples):
            tid, body = sample["task_id"], sample["body"]
            code      = merge(PROBLEMS[tid]["prompt"], body)
            futs.append(ex.submit(run_pytest, code,
                                  PROBLEMS[tid]["test_code"], args.timeout))
            futs[-1].tid = tid

        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            if fut.result():
                passed[fut.tid] += 1

    totals  = np.array([args.n] * len(task_ids))
    correct = np.array([passed[tid] for tid in task_ids])
    score   = pass_at_k(totals, correct, args.k).mean()
    print(f"\n[{mode}] pass@{args.k} = {score:.2%} "
          f"(n={args.n}, tasks={len(task_ids)})")


# ───── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",       type=int,   default=40, help="samples per task")
    p.add_argument("--k",       type=int,   default=5,  help="pass@k")
    p.add_argument("--limit",   type=int,   default=None)
    p.add_argument("--temp",    type=float, default=0.8)
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--direct",  type=str,   default=None,
                   help="Adapter alias for direct call (skip router)")
    args = p.parse_args()
    main(args)
