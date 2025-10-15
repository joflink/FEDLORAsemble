# tgi_router_client.py
import requests, time, json
from ALBERTRouter import ALBERTRouterHF

from duckduckgo_search import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer as HfTok, BitsAndBytesConfig
import torch
def web_search(query: str, max_len: int = 500) -> str:
    """
    Returnera upp till tre DuckDuckGo-träffar som
    'Title: …\nSnippet: …' och inget annat.
    """
    hits = DDGS().text(query, max_results=3)
    if not hits:
        return "❌ No web results."
    body = "\n\n".join(
        f"Title: {h['title']}\nSnippet: {h['body']}" for h in hits
    )
    return body[:max_len]



SERVER = "http://localhost:8080/v1/completions"
HEADERS = {"Content-Type": "application/json"}

MODEL   = "Qwen2.5-0.5B-Instruct"        # foundation på TGI-servern

# id → alias,  preprompt,          max_tokens
EXPERT_CFG = {
    0: ("reasoning", "Reason step‑by‑step:\n",           256),
    1: ("general",   "You are a helpful assistant:\n",   256),
    2: ("math",      "You are a math expert:\n",         256),
    3: ("code",      "You are a coding expert:\n",      256),
    4: ("web",       "",                                 256),   # web‑search pseudo‑expert
}
router = ALBERTRouterHF(general_expert_id=1)
temperature=0
def generate(prompt, k_vote=3, temp=0.2):
    t0 = time.perf_counter()


    #expert_id = router(prompt, k=k_vote)
    expert_id = router.forward(prompt, k=k_vote)
    # 2 — Web‑search pseudo‑expert
    if expert_id == 4:
        # hämta bakgrund via DuckDuckGo
        background = web_search(prompt)
        web_pre = (
            "Below is background information gathered from a web search. "
            "Use it to answer the question. "
            f"[WEB INFO]\n{background}\n\n[USER QUESTION]\n"
        )

        payload = {
            "model": MODEL,
            "prompt": web_pre + prompt,
            "adapter_id": "general",           # du kan ha egen web‑LoRA om du vill
            "max_tokens": 256,
            "temperature": temperature,
            "stream": False
        }
        r = requests.post(SERVER, headers=HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        answer = r.json()["choices"][0]["text"]
        return answer, (time.perf_counter()-t0)*1000, expert_id

    # 3 — bygga TGI‑payload
    alias, pre, mtok = EXPERT_CFG[expert_id]
    payload = {
        "model": MODEL,
        "prompt": pre + prompt,
        "adapter_id": alias,
        "max_tokens": mtok,
        "temperature": temperature,
        "stream": False
    }
    r = requests.post(SERVER, headers=HEADERS, data=json.dumps(payload))
    r.raise_for_status()
    answer = r.json()["choices"][0]["text"]
    latency_ms = (time.perf_counter() - t0)*1000
    return answer, latency_ms, expert_id
