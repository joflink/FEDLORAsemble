# test_tgi_router.py – snabb sanity‑suite för alla domäner
# Kör:  python test_tgi_router.py  (kräver att tgi_router_client.py ligger i PYTHONPATH)

from tgi_router_client import generate
import time, csv

TESTS = {
    "reasoning": [
        "A farmer has chickens and cows. There are 30 heads and 100 legs. How many of each animal does he have?",
        "If it takes 3 painters 3 days to paint 3 fences, how many fences can 9 painters paint in 9 days?",
    ],
    "general": [
        "Vad heter huvudstaden i Kroatien?",
        "Explain the difference between HTTP/1.1 and HTTP/2 in two bullet points.",
    ],
    "math": [
        "<domain:math> Integrate x^3 * sin(x) dx step‑by‑step.",
        "<domain:math> Solve the equation 2x^2 ‑ 5x + 3 = 0.",
    ],
    "code": [
        "<domain:code> Write a Python function that returns the nth Fibonacci number using memoisation.",
        "<domain:code> Create a Svelte component that displays the current date and updates every minute.",
    ],
}


rows = []
print("Running sanity‑suite …\n")
for domain, prompts in TESTS.items():
    for p in prompts:
        text, lat, eid = generate(p, max_tokens=256)
        print(f"[{domain.upper()}] ({lat:.0f} ms, expert {eid})\nQ> {p}\nA> {text[:200]}…\n{'-'*60}")
        rows.append({"domain": domain, "prompt": p, "latency_ms": lat, "expert_id": eid})

# — save csv for later use —
with open("sanity_results.csv", "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
print("\n✅ Done. Results saved to sanity_results.csv")
