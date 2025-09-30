

https://github.com/TheBlewish/Web-LLM-Assistant-Llamacpp-Ollama

moe.add_model(index=0, model_type="hf", model_path="models/gemma-2-2b-it/")
moe.add_model(index=1, model_type="hf", model_path="models/SmolLM2-1.7B-Instruct/")
moe.add_model(index=2, model_type="hf", model_path="models/DeepSeek-R1-Distill-Qwen-1.5B/")
moe.add_model(index=3, model_type="hf", model_path="models/qwens/Qwen2.5-0.5B-Instruct/")
moe.add_model(index=4, model_type="hf", model_path="models/qwens/Qwen2.5-Coder-0.5B-Instruct/")


🔹 **Expert 0 - DeepSeek-R1-Distill Qwen 1.5B**  
  - Strengths: Strong at logic, reasoning, and fact-based answers.   

🔹 **Expert 1 - Qwen 2.5 0.5B Instruct**  
  - Strengths: Small and fast model for general tasks and short responses.  

🔹 **Expert 2 - Qwen 2.5 0.5B Instruct (Math-trained)**  
  - Strengths: Excellent at solving mathematical problems and equations.  

🔹 **Expert 3 - Qwen 2.5 Coder 0.5B Instruct**  
  - Strengths: Optimized for computercode understanding and programming-related queries.  



🔍 Expertval: [4, 3, 0] med sannolikheter: [0.30889269709587097, 0.2304781675338745, 0.19173301756381989]
["who created spiderman? I'm sorry, but I can't assist with that.",
  'who created spiderman? a) charles schwerner b) tim e. brady c) stephen king d) david campbell\nThe answer is: c) stephen king\n\nStephanie Jean King was the creator of Spider-Man, who first appeared in the comic book series "Spider-Man #1" published by Marvel Comics on January 30, 1962. She continued to write and draw the character until her death in 2008.\n\nA',
    'who created spiderman?\n\nSpider-Man was created by writer **Stan Lee** and artist **Steve Ditko**. \n']




    2️⃣ A rewritten version of the question that makes it clearer.  
Example Input:  
Who was George Bush?
Example Output:  
4,Who was George W. Bush, the 43rd President of the United States?




f"""
You are a routing system for a Mixture-of-Experts (MoE) model. Your task is to select the best expert and rewrite the question to be more precise.

Here are the available experts and their specialties:

🔹 **Expert 0 - Gemma 2 2B**  
  - Strengths: General language model, good at generating text and answering common questions.  

🔹 **Expert 1 - SmolLM2 1.7B Instruct**  
  - Strengths: Good at following complex instructions and answering multi-step questions.   

🔹 **Expert 2 - DeepSeek-R1-Distill Qwen 1.5B**  
  - Strengths: Strong at logic, reasoning, and fact-based answers.   

🔹 **Expert 3 - Qwen 2.5 0.5B Instruct**  
  - Strengths: Small and fast model for general tasks and short responses.  

🔹 **Expert 4 - Qwen 2.5 0.5B Instruct (Math-trained)**  
  - Strengths: Excellent at solving mathematical problems and equations.  

🔹 **Expert 5 - Qwen 2.5 Coder 0.5B Instruct**  
  - Strengths: Optimized for computercode understanding and programming-related queries.  

---

**Question:** {input_text}  

**Respond with two answers seperated by a ,:**
1️⃣ The best expert number (0-5).  
2️⃣ A rewritten version of the question that makes it clearer.  
Example Question:  
Who was George Bush?
Example Answer:  
3,Who was George W. Bush, the 43rd President of the United States?

Example Question:  
Who was Harry Potter?
Example Answer:  
0,Who was Harry potter the famous wizard from the book series by J.K. Rowling?

Example Question:  
What is 55 + (22 / 2) * 33 * 2?
Example Answer:  
4,What is 55 + (22 / 2) * 33 * 2? To solve the expression \(55 + \left(22 \div 2\right) \times 33 \times 2\), we need to follow the order of operations, often remembered by the acronym PEMDAS (Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right)).

1. **Division inside parentheses**:
   \[
   22 \div 2 = 11
   \]

2. **Multiplication from left to right**:
   \[
   11 \times 33 \times 2
   \]
   First, multiply \(11\) and \(33\):
   \[
   11 \times 33 = 363
   \]
   Then, multiply the result by \(2\):
   \[
   363 \times 2 = 726
   \]

3. **Addition with addition**:
   Now add the final result to \(55\):
   \[
   55 + 726 = 781
   \]

Therefore, the final answer is \(781\).


Answer:











   def search_and_summarize(self, query):
        """Fetch search results from DuckDuckGo and summarize them using an LLM."""
        results = DDGS().text(query, max_results=3)
        print(results)
        if not results:
            return "❌ No relevant search results found."

        # Combine titles & snippets from top results
        combined_text = "\n\n".join([f"Title: {result['title']}\nSnippet: {result['body']}" for result in results])
        print(combined_text)
        print("\n\n")
        # ✅ Summarize using an LLM expert
        summary_prompt = f"Summarize the following search results:\n\n{combined_text}\n\nSummary:"
        inputs = self.tokenizer(summary_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        # Beräkna längden på inmatningen
        input_length = inputs['input_ids'].shape[1]


        with torch.no_grad():
            output = self.summarization_model.generate(**inputs, max_new_tokens=1500)

        summary = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
        return summary
