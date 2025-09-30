import re, math, ast

# ────────── multiple‑choice helpers ──────────
CHOICE2ID = {"A": 0, "B": 1, "C": 2, "D": 3,
             "a": 0, "b": 1, "c": 2, "d": 3,
             0: 0, 1: 1, 2: 2, 3: 3}

def extract_choice(text: str) -> int:
    """Plockar första ensamma A–D eller sista siffran 0–3."""
    m = re.search(r"\b([ABCD])\b", str(text))
    if m:
        return CHOICE2ID[m.group(1)]
    nums = re.findall(r"[0-3]", str(text))
    return CHOICE2ID[int(nums[-1])] if nums else -1

# ────────── numeric answer (GSM8K) ──────────
ANS_RE = re.compile(r"(?:boxed|####)?\s*[{]?(?P<ans>-?\d+(?:\.\d+)?)[}]?\s*$")
def extract_number(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    m = ANS_RE.search(text)
    if m:
        return m.group("ans").lstrip("0") or "0"
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1].lstrip("0") if nums else text.strip()

# ────────── kod‑sanering ──────────
def keep_first_def(txt: str) -> str:
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")
    i = txt.find("def ")
    if i == -1:
        return txt.strip()
    snippet = txt[i:].split("# Example")[0].strip()
    try:
        import ast, textwrap
        mod = ast.parse(snippet)
        first = mod.body[0]
        end = first.end_lineno
        snippet = "\n".join(snippet.splitlines()[:end])
        return textwrap.dedent(snippet)
    except Exception:
        return snippet

