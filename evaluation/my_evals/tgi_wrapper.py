from evals.completion_fns import CompletionFn
from tgi_router_client import generate

class TGICompletion(CompletionFn):
    def call(self, prompt, **_):
        txt, lat, eid = generate(prompt)
        return {\"choices\":[{\"text\":txt}], \"usage\":{\"latency_ms\":lat,\"expert_id\":eid}}
