# tgi_router_client.py
import requests
import time
import json
from duckduckgo_search import DDGS
# from transformers import AutoModelForCausalLM, AutoTokenizer as HfTok, BitsAndBytesConfig # Inte nödvändigt i klienten längre
# import torch # Inte nödvändigt i klienten längre

from ALBERTRouter import ALBERTRouterHF as ALBERTRouter

def web_search(query: str, max_snippet_len: int = 500) -> str:
    """
    Returnera upp till tre DuckDuckGo-träffar som
    'Title: …\nSnippet: …' och inget annat.
    """
    print(f"Performing web search for: {query}")
    hits = DDGS().text(query, max_results=3)
    if not hits:
        return "❌ No web results found."
    
    body_parts = []
    current_len = 0
    for h in hits:
        title_snip = f"Title: {h['title']}\nSnippet: {h['body']}"
        if current_len + len(title_snip) > max_snippet_len and body_parts:
            break # Lägg inte till mer om det överskrider max_len
        body_parts.append(title_snip)
        current_len += len(title_snip) + 2 # +2 för \n\n
    print(body_parts)
    return "\n\n".join(body_parts)


TGI_SERVER_URL = "http://localhost:8080/v1/completions" # Standard TGI endpoint
TGI_HEADERS = {"Content-Type": "application/json"}

# Modellnamnet som används på TGI-servern (foundation model)
TGI_MODEL_NAME = "Qwen2.5-0.5B-Instruct" # Exempel, byt till din modell

# Expertkonfiguration: id -> (alias, preprompt, max_tokens)
# Aliaset används som adapter_id i TGI-anropet.
EXPERT_CFG = {
    0: ("indian", "you are a historian\n", 5),
    1: ("general", "You are a helpful assistant:\n", 5),
    2: ("math", "You are a math expert. Solve systematically:\n", 5),
    3: ("code", """You are a world-class AI coding expert and engineer with deep experience in Python, Ruby, SQL, Java, JavaScript, C#, Go, Rust, and other modern programming languages. You always write clean, efficient, and scalable code following the latest industry standards. When the user asks a question:

1. Carefully read the question and identify the language, framework, and version requested.
2. Provide a concrete, runnable code example with comments explaining each step.
3. Briefly explain why this is the best solution (e.g., design patterns, performance, security).
4. When relevant, link to official documentation or other authoritative resources.

Your tone is educational, professional, and helpful. Always tailor your answer to the user’s skill level and context.\n""", 5), # Ge code lite mer tokens
    4: ("web", "", 256),  # Web-search pseudo-expert. Preprompt hanteras speciellt.
}

# Initiera routern (använder dummy-versionen definierad ovan)
router = ALBERTRouter(general_expert_id=1) # general_expert_id = 1 (alias "general")

def generate(prompt: str, k_vote: int = 3, temp: float = 0.85): # temp default 0.1, k_vote default 3
    """
    Hanterar routing och anrop till TGI-servern.
    Returnerar (svar_text, total_latens_ms, expert_id_int).
    """
    router_latency_ms=0
    llm_latency_ms=0
    total_latency_ms=0
    t_start_router = time.perf_counter()
    t_start_total = time.perf_counter()
    # 1. Router väljer expert
    # (ALBERTRouterHF.forward returnerar ett heltal expert_id)
    expert_id = router.forward(prompt, k=k_vote)
    router_latency_ms = (time.perf_counter() - t_start_router) * 1000

    # 2. Hantera webb-sökning (pseudo-expert)
    if expert_id == 4: # Antag att 4 är ID för "web"
        print("Web expert (ID 4) chosen by router. Performing web search...")
        background_info = web_search(prompt)
        
        # Speciell preprompt för webbresultat
        web_preprompt = (
            "You are a helpful assistant. Based on the following web search results, "
            "provide a comprehensive answer to the user's question. "
            "If the web search results are not relevant, say so.\n\n"
            f"[WEB INFO]\n{background_info}\n\n[USER QUESTION]\n"
        )
        
        # Använd "general" adaptern (eller en specifik "web_summary" adapter om du har en)
        # för att summera/svara baserat på webbinfo. Max_tokens från web expert config.
        _, _, web_max_tokens = EXPERT_CFG[expert_id] # Använd max_tokens från web-config
        
        payload = {
            "model": TGI_MODEL_NAME,
            "prompt": web_preprompt + prompt,
            "adapter_id": "general", # Tvinga general adaptern för webbsök-svar
            "max_tokens": web_max_tokens,
            "temperature": temp,
            "stream": False,
        }
        adapter_alias_for_log = "web_via_general"
    else:
        # 3. Bygg TGI-payload för vanlig LoRA-expert
        if expert_id not in EXPERT_CFG:
            print(f"Warning: Router returned unknown expert_id {expert_id}. Falling back to general.")
            expert_id = router.general_expert_id # Fallback till general expertens ID

        adapter_alias, preprompt_str, max_tokens_val = EXPERT_CFG[expert_id]
        adapter_alias_for_log = adapter_alias
        
        payload = {
            "model": TGI_MODEL_NAME,
            "prompt": preprompt_str + prompt,
            "adapter_id": adapter_alias, # Detta är LoRA-adapterns namn
            "max_tokens": max_tokens_val,
            "temperature": temp,
            "stream": False,
            #"stop": ["###", "\n\n"]
        }

    # Skicka request till TGI-servern
    print(f"Sending payload to TGI for adapter '{adapter_alias_for_log}': {payload['prompt'][:100]}...")
    try:
        t_start_llm = time.perf_counter()
        response = requests.post(TGI_SERVER_URL, headers=TGI_HEADERS, data=json.dumps(payload), timeout=60)
        response.raise_for_status()  # Kasta exception för HTTP-fel (4xx eller 5xx)
        answer_text = response.json()["choices"][0]["text"].strip()
        llm_latency_ms = (time.perf_counter() - t_start_llm) * 1000

    except requests.exceptions.RequestException as e:
        print(f"ERROR: TGI request failed: {e}")
        answer_text = f"Error: Could not get response from TGI server for expert {adapter_alias_for_log}."
    except (KeyError, IndexError) as e:
        print(f"ERROR: Could not parse TGI response: {e} - Response: {response.text}")
        answer_text = f"Error: Invalid response from TGI server for expert {adapter_alias_for_log}."


    total_latency_ms = (time.perf_counter() - t_start_total) * 1000
    #print(f"TGI response received. Total latency (routing+LLM): {total_latency_ms:.1f}ms")
    return answer_text,router_latency_ms, llm_latency_ms,total_latency_ms, expert_id