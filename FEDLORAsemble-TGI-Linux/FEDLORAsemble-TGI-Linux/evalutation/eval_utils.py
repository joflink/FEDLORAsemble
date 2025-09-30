# ─────────────────────────── eval_utils.py ───────────────────────────
import re

# ────────── multiple-choice helpers ──────────
# CHOICE2ID = {
#     "A": 0, "B": 1, "C": 2, "D": 3,"E": 4,
#     "a": 0, "b": 1, "c": 2, "d": 3,"e": 4,
#     "0": 0, "1": 1, "2": 2, "3": 3,"3": 4,
#     0: 0, 1: 1, 2: 2, 3: 3,4:4,
# }

CHOICE2ID = {
    **{k: i for i, k in enumerate("ABCD")},
    **{k.lower(): i for i, k in enumerate("ABCD")},
    **{str(i): i for i in range(10)},   # '0'–'9'
    **{i: i for i in range(10)},        # 0–9 som int
}



def extract_choice(text: str) -> int:
    """Returnerar första ensamma A–D eller sista siffran 0–3, annars -1."""
    m = re.search(r"\b([ABCD])\b", str(text))
    if m:
        return CHOICE2ID[m.group(1)]
    nums = re.findall(r"[0-3]", str(text))
    return CHOICE2ID[int(nums[-1])] if nums else -1

# ────────── numeric answer (GSM8K) ──────────
ANS_RE = re.compile(
    r"(?:boxed|####)?\s*[{]?(?P<ans>-?\d+(?:\.\d+)?)[}]?\s*$",
    flags=re.M,
)
def extract_number(text: str) -> str:
    text = str(text)
    m = ANS_RE.search(text)
    if m:
        return m.group("ans").lstrip("0") or "0"
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1].lstrip("0") if nums else text.strip()
