from evals.metric import Metric
from evals.record import record_metrics
from extractor import extract_answer

class MathExactMatch(Metric):
    def __init__(self): self.ok = 0; self.n = 0
    def add(self, example, completion):
        gold = extract_answer(example[\"completion\"])
        pred = extract_answer(completion)
        self.ok += (gold==pred); self.n += 1
    def report(self):
        acc = self.ok/max(self.n,1)
        record_metrics({\"exact_match\":acc})
        return {\"exact_match\":acc}
