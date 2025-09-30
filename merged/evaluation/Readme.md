0=reflect
1= General
2= Math
3= Code
4= websearch

Math test
Accuracy: 54.36%   N=1319


python3 run_eval.py \
  --path cais/mmlu \
  --config all \
  --metric accuracy \
  --fewshot 0 \
  --limit 500


python3 run_eval.py \
  --path gsm8k --config main --split test \
  --q question --a answer --fewshot 8



python3 run_eval.py \
  --path gsm8k --config main --split test \
  --q question --a answer --fewshot 4 




curl 127.0.0.1:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "Hwhat is 5 + 5?",
  "parameters": {
    "max_new_tokens": 40,
    "adapter_id": "math"
  }
}'


docker ps -a

docker compose up -d --force-recreate
docker logs fedlora



websearch:
❓ Q: what is the latest news about the icehockey world cup?
🤖 Using TGI Router Client for: what is the latest news about the icehockey world cup?...
🔎 Top-3 Experts: [4, 1, 0] with probabilities [0.9774433970451355, 0.008445114828646183, 0.005270545836538076]
✅ Selected Expert: 4 with confidence 0.98
Web expert (ID 4) chosen by router. Performing web search...
Performing web search for: what is the latest news about the icehockey world cup?
['Title: Canada shuts out Slovenia to open ice hockey worlds and ... - ABC News\nSnippet: STOCKHOLM -- Canada opened the ice hockey world championship by shutting out newcomer Slovenia 4-0 on Saturday. Bo Horvat scored two power-play goals, Nathan MacKinnon had a goal and two assists ...']
Sending payload to TGI for adapter 'web_via_general': You are a helpful assistant. Based on the following web search results, provide a comprehensive answ...
TGI response received. Total latency (routing+LLM): 15109.6ms
⏱️  LLM/Client Total: 15109.6 ms | MoESystem Total: 15109.9 ms
💡 A: Based on the information provided in the web search results, the latest news about the Ice Hockey World Cup is:

Canada shut out Slovenia in the opening round of the tournament, with a score of 4-0. This was the first time Canada had ever defeated Slovenia in a World Cup match. The win was particularly significant as it marked Canada's first victory in the tournament since 2008, when they lost to the Czech Republic in the quarterfinals. The win also marked Canada's first victory in the tournament since 2010, when they lost to the United States in the quarterfinals. The win was also Canada's first victory in the tournament since 2010, when they lost to the United States in the quarterfinals. The win was also Canada's first victory in the tournament since 2010, when they lost to the United States in the quarterfinals. The win was also Canada's first victory in the tournament since 2010, when they lost to the United States in the quarterfinals. The win was also Canada's first victory in the tournament since 2010, when they lost to the United States in the quarterfinals. The win was also Canada's first



timestamp,prompt,expert_route,response,router_ms,llm_ms,total_system_ms
2025-05-13 09:59:07,What is the capital of Sweden?,routed:general(id:1),"The capital of Sweden is Stockholm. Stockholm is the largest city in Sweden and the capital of the country. It is located in the province of Stockholm County, on the island of Sweden. Stockholm is known for its beautiful architecture, world-class museums, and cultural institutions. It is also home to the Royal Palace, the Parliament building, and the Swedish Parliament. The city is also known for its extensive public transportation system, including the Stockholm Metro and the Stockholm Bus. Stockholm is a major economic center and a major tourist destination in Sweden. It is home to many universities, research institutions, and cultural institutions. The city is also known for its diverse population, with a mix of Swedish, Finnish, and other nationalities. The city is also home to the Swedish Parliament, the Swedish National Museum, and the Swedish National Library. The city is also home to the Swedish National Museum, the Swedish National Library, and the Swedish National Museum. The city is also home to the Swedish National Museum, the Swedish National Library, and the Swedish National Museum. The city is also home to the Swedish National Museum, the Swedish National Library, and the Swedish National Museum. The city is also home to the Swedish National Museum, the Swedish National Library, and the Swedish National Museum. The city is also",39.4,15854.9,15894.4
