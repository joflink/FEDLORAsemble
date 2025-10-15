# extractor.py
import re
def extract_answer(text: str) -> str:
    m = re.search(r"(?:boxed|####)?\\s*[{]?(\\-?\\d+(?:\\.\\d+)?)[}]?\\s*$", text)
    if m: return m.group(1).lstrip(\"0\") or \"0\"
    nums = re.findall(r\"\\-?\\d+(?:\\.\\d+)?\", text)
    return nums[-1].lstrip(\"0\") if nums else \"\"
