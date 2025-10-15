# moe_system_simplified.py
import os
import re
import json
import time
import requests # För MoESystems direkta TGI-anrop

import csv # Importera csv-modulen
# Importera från din tgi_router_client.py
import tgi_router_client # Ger tillgång till tgi_router_client.generate, .EXPERT_CFG, etc.

TAG_RE = re.compile(r"<domain:(\w+)>", flags=re.I)

class MoESystem:
    def __init__(self, 
                 tgi_server_url: str = tgi_router_client.TGI_SERVER_URL,
                 tgi_model_name: str = tgi_router_client.TGI_MODEL_NAME,
                 expert_config_from_client: dict = tgi_router_client.EXPERT_CFG,
                 use_tgi_router_by_default: bool = True,
                 default_direct_call_alias: str = "general"):
        
        self.tgi_server_url = tgi_server_url
        self.tgi_model_name = tgi_model_name
        self.use_tgi_router_by_default = use_tgi_router_by_default
        self.default_direct_call_alias = default_direct_call_alias # Används om use_tgi_router_by_default är False

        self.expert_config_by_id = expert_config_from_client
        self.expert_config_by_alias = {
            details[0]: {"preprompt": details[1], "max_tokens": details[2], "id": id_}
            for id_, details in expert_config_from_client.items()
        }
        
        # Säkerställ att default_direct_call_alias finns
        if self.default_direct_call_alias not in self.expert_config_by_alias:
            raise ValueError(f"Default direct call alias '{self.default_direct_call_alias}' not found in expert configuration.")

        self.log_file = "moe_chat_history.csv"
        print(f"MoESystem initialized. Default routing via TGI Client: {self.use_tgi_router_by_default}.")
        if not self.use_tgi_router_by_default:
            print(f"Default direct TGI calls will use adapter: '{self.default_direct_call_alias}'")

    def _make_tgi_api_call(self, payload: dict):
        """Privat metod för att göra direkta TGI API-anrop."""
        try:
            response = requests.post(self.tgi_server_url, headers=tgi_router_client.TGI_HEADERS, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Direct TGI request failed: {e}")
            return f"Error: Could not get response from TGI server."
        except (KeyError, IndexError) as e:
            print(f"ERROR: Could not parse TGI response: {e} - Response: {response.text}")
            return f"Error: Invalid response from TGI server."

    # def _save_history(self, prompt, expert_route_info, response, r_llm_ms,t_llm_ms, t_total_ms):
    #     # t_router_ms är nu inkluderat i t_llm_ms om tgi_router_client används
    #     entry = {
    #         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    #         "prompt": prompt,
    #         "expert_route": expert_route_info,
    #         "response": response,
    #         "router__ms": round(r_llm_ms, 1), # Inkluderar routertid om tgi_client användes
    #         "t_llm__ms": round(t_llm_ms, 1), # Inkluderar routertid om tgi_client användes
    #         "t_total_system_ms": round(t_total_ms, 1),
    #     }
    #     history = []
    #     if os.path.exists(self.log_file):
    #         try:
    #             with open(self.log_file, "r", encoding="utf-8") as f:
    #                 history = json.load(f)
    #         except json.JSONDecodeError:
    #             pass 
    #     history.append(entry)
    #     with open(self.log_file, "w", encoding="utf-8") as f:
    #         json.dump(history, f, ensure_ascii=False, indent=2)

    def _save_history(self, prompt: str, expert_route_info: str, response: str, 
                      router_ms: float, llm_ms: float, total_system_ms: float):
        """
        Spara en enskild interaktionspost till en CSV-fil.
        Skriver header om filen är ny.
        """
        
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "expert_route": expert_route_info,
            "response": response,
            "router_ms": round(router_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_system_ms": round(total_system_ms, 1),
        }
        
        # Definiera fältnamnen för CSV-filen. Ordningen här bestämmer kolumnordningen.
        fieldnames = [
            "timestamp", 
            "prompt", 
            "expert_route", 
            "response", 
            "router_ms", 
            "llm_ms", 
            "total_system_ms"
        ]
        
        # Kontrollera om filen redan existerar för att avgöra om headern ska skrivas
        file_exists = os.path.exists(self.log_file)
        is_empty = False
        if file_exists:
            is_empty = os.path.getsize(self.log_file) == 0
        
        try:
            with open(self.log_file, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Skriv headern endast om filen är ny eller tom
                if not file_exists or is_empty:
                    writer.writeheader()
                    
                writer.writerow(entry)
        except IOError as e:
            print(f"Error writing to CSV log file {self.log_file}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV logging: {e}")

    def forward(self, prompt: str, temperature: float = 0.1):
        t_start_total_system = time.perf_counter()
        final_answer = ""
        llm_time_ms = 0 # Kommer att inkludera routertid om tgi_router_client används
        expert_route_info_for_log = "N/A"
        router_latency_ms=0
        llm_latency_ms =0
        if self.use_tgi_router_by_default:
            print(f"🤖 Using TGI Router Client for: {prompt[:60]}...")
            # Anropet till tgi_router_client.generate inkluderar dess interna routertid + LLM-tid
            final_answer,router_latency,llm_latency, total_client_latency_ms, expert_id_from_router = \
                tgi_router_client.generate(prompt, temp=temperature)
            
            llm_time_ms = total_client_latency_ms 
            router_latency_ms = router_latency 
            llm_latency_ms = llm_latency 
            
            # Hämta alias för loggning
            chosen_alias = self.expert_config_by_id.get(expert_id_from_router, (f"unknown_id:{expert_id_from_router}",'',''))[0]
            expert_route_info_for_log = f"routed:{chosen_alias}(id:{expert_id_from_router})"
        else: # Direkt TGI-anrop (t.ex. till general expert)
            cfg = self.expert_config_by_alias[self.default_direct_call_alias]
            preprompt_str = "You are a helpful assitent, assist in solving this task:"
            adapter_id_str = self.default_direct_call_alias
            max_tokens_val = 512
            
            print(f"📞 Direct TGI call (without adapter) for: {prompt[:60]}...")
            t_llm_start = time.perf_counter()
            payload = {
                "model": self.tgi_model_name,
                "prompt": preprompt_str + prompt,
                "max_tokens": max_tokens_val,
                "temperature": temperature,
                "stream": False
            }
            final_answer = self._make_tgi_api_call(payload)
            llm_time_ms = (time.perf_counter() - t_llm_start) * 1000
            expert_route_info_for_log = f"direct: no adapter"

        total_system_time_ms = (time.perf_counter() - t_start_total_system) * 1000
        
        # Om llm_time_ms är nära total_system_time_ms (vilket det oftast är nu),
        # är det ok. Skillnaden är den lilla overhead i MoESystem.
        print(f"⏱️  LLM/Client Router:  {router_latency_ms:.1f} ms, llm: {llm_latency_ms:.1f} ms | MoESystem Total: {total_system_time_ms:.1f} ms")
        self._save_history(prompt, expert_route_info_for_log, final_answer, router_latency_ms, llm_latency_ms,total_system_time_ms)
        return final_answer

# ────────────────────────────  demo  ─────────────────────────────────────────
if __name__ == "__main__":
    print("--- MoE System Demo (Simplified with TGI Router Client) ---")
    
    # Exempel: Initiera MoESystem för att använda TGI-routern som standard
    moe_with_router = MoESystem(use_tgi_router_by_default=True)
    
    # Exempel: Initiera MoESystem för att göra direkta anrop till "general" som standard
    # moe_direct_general = MoESystem(use_tgi_router_by_default=False, default_direct_call_alias="general")

    # queries = [
    #     "What is the capital of Sweden?", # Should use TGI router (likely general)
    #      "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", # Tagged math
    #      "Explain bubble-sort and give Python code.", # Tagged code
    #      "Name the counterpart of Baniyas in Punjab and Komatis in Golconda during the Mughal period.", # Should be routed to web by tgi_router_client's internal router
    #      "Which regions were involved in trade relations with India through land routes?", # Tagged web,
    #      "Write a short story for me about a dragon that captures a princess.", # Should use TGI router (likely general)
    #      "Explain general relativity in simple terms.", # Should use TGI router (likely reasoning or general)
    # ]

#     queries= [
#         "What marked the period of the Gupta dynasty in terms of progress?",
# "What facilitated the growth of trade in the Sangam economy?",
# "What were some important towns and craft centers during the Sangam period?",
# "What were the characteristics of Vedic literature, and what texts are considered as Vedic?",
# "What are the four separate collections included in the Mantra category, and what is their significance?",
#  "Describe the role of artisans and merchants in town administration during the Gupta period.",
#     ]
    queries= [
    "You are given a matrix of m rows and n columns. Write a function that calculates the transpose of the given matrix. matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
"Create a list comprehension that takes all the elements of list_one and creates a new list where all the elements are doubled. list_one = [1, 2, 3, 4, 5]",
"Construct a SQL query to find all columns in a table called \"customers\" where the first name is equal to 'John'.",
"Write a Java program to find the largest element in a given array.",
 "Create an HTML form with radio buttons for selecting gender.",
"Write a Ruby program to search for a specific item in an array of strings. array = [\"Apple\",\"Banana\",\"Mango\",\"Orange\"]"
   ]
    print("\n--- Testing MoESystem with TGI Router Enabled by Default ---")
    for q in queries:
        print(f"\n{'='*15} QUERY START {'='*15}")
        print(f"❓ Q: {q}")
        response = moe_with_router.forward(q, temperature=0.85)
        print(f"💡 A: {response}")
        print(f"{'='*15} QUERY END {'='*17}\n")

    # print("\n--- Testing MoESystem with Direct 'general' Calls by Default (tags still work) ---")
    # for q in queries:
    # print(f"\n{'='*15} QUERY START {'='*15}")
    #     print(f"❓ Q: {q}")
    #     response = moe_direct_general.forward(q, temperature=0.1)
    # print(f"💡 A: {response}")
    # print(f"{'='*15} QUERY END {'='*17}\n")