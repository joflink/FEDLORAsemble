# oe_compl_fn.py
from tgi_router_client import generate

class TGICompletion:
    def __call__(self, prompt, **_):
        text, lat_ms, eid = generate(prompt)
        return {
            "choices": [{"text": text}],
            "usage": {"latency_ms": lat_ms, "expert_id": eid}
        }
