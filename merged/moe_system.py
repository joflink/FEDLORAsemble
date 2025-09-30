"""
FEDLORAsemble - TGI-Based Mixture of Experts System
==================================================

This is the main file for the FEDLORAsemble system implementing a 
Mixture of Experts (MoE) architecture with ALBERT-based routing and 
Text Generation Inference (TGI) as the primary inference backend.

The system includes:
- ALBERT-based router for intelligent expert selection
- TGI server integration with LoRA adapters
- Web-based search as fallback expert
- Optimized inference with ONNX quantization
- Comprehensive logging and evaluation

Author: FEDLORAsemble Team
Date: 2025
Version: 2.0 (TGI-focused with English documentation)
"""

import os
import time
import json
import re
import requests
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any

# DuckDuckGo for web search functionality
from duckduckgo_search import DDGS

# Import the TGI router components
import sys
sys.path.append('evaluation')
from ALBERTRouter import ALBERTRouterQuant as ALBERTRouter


def web_search(query: str, max_snippet_len: int = 500) -> str:
    """
    Perform web search using DuckDuckGo and return formatted results.
    
    Args:
        query: Search query string
        max_snippet_len: Maximum length of combined snippets
        
    Returns:
        Formatted search results as string
    """
    print(f"🔍 Performing web search for: {query}")
    try:
        hits = DDGS().text(query, max_results=3)
        if not hits:
            return "❌ No web results found."
        
        body_parts = []
        current_len = 0
        for hit in hits:
            title_snippet = f"Title: {hit['title']}\nSnippet: {hit['body']}"
            if current_len + len(title_snippet) > max_snippet_len and body_parts:
                break  # Don't add more if it exceeds max_len
            body_parts.append(title_snippet)
            current_len += len(title_snippet) + 2  # +2 for \n\n
        
        return "\n\n".join(body_parts)
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return f"Error: Could not perform web search - {str(e)}"


def make_tgi_request(endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[str, bool]:
    """
    Make a request to the TGI server.
    
    Args:
        endpoint: TGI server endpoint URL
        payload: Request payload dictionary
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (response_text, success_flag)
    """
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        
        response_data = response.json()
        answer_text = response_data["choices"][0]["text"].strip()
        return answer_text, True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ TGI request failed: {e}")
        return f"Error: Could not get response from TGI server - {str(e)}", False
    except (KeyError, IndexError) as e:
        print(f"❌ Could not parse TGI response: {e}")
        return f"Error: Invalid response from TGI server", False


class TGIExpert:
    """
    Expert that uses TGI (Text Generation Inference) server with LoRA adapters
    for specialized inference on different domains.
    
    This class represents a specialized expert within the MoE system that
    leverages TGI's optimized inference engine with domain-specific LoRA adapters.
    """
    
    def __init__(self, expert_id: int, adapter_name: str, tgi_endpoint: str, 
                 preprompt: str = "", max_tokens: int = 500, model_name: str = "Qwen2.5-0.5B-Instruct"):
        """
        Initialize TGI Expert.
        
        Args:
            expert_id: Unique identifier for this expert
            adapter_name: Name of the LoRA adapter to use with TGI
            tgi_endpoint: TGI server endpoint URL
            preprompt: Domain-specific preprompt for the expert
            max_tokens: Maximum number of tokens to generate
            model_name: Base model name used by TGI server
        """
        self.expert_id = expert_id
        self.adapter_name = adapter_name
        self.tgi_endpoint = tgi_endpoint
        self.preprompt = preprompt
        self.max_tokens = max_tokens
        self.model_name = model_name
        
        print(f"🔧 Initialized TGI Expert {expert_id} with adapter '{adapter_name}'")

    def forward(self, input_text: str, temperature: float = 0.7) -> str:
        """
        Process input and generate response from the TGI expert.
        
        Args:
            input_text: User's input text
            temperature: Sampling temperature for generation
            
        Returns:
            Generated response from the expert
        """
        prompt = f"{self.preprompt}{input_text}"
        
        # Build TGI payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "adapter_id": self.adapter_name,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        
        print(f"🚀 TGI Expert {self.expert_id} processing with adapter '{self.adapter_name}'")
        
        # Make request to TGI server
        response_text, success = make_tgi_request(self.tgi_endpoint, payload)
        
        if success:
            print(f"✅ TGI Expert {self.expert_id} generated response")
            return response_text
        else:
            print(f"❌ TGI Expert {self.expert_id} failed")
            return response_text  # Error message


class WebSearchExpert:
    """
    Web-based search expert that fetches results via DuckDuckGo and
    summarizes them using TGI with a general adapter.
    
    This expert is activated when the router is uncertain or when
    current information is needed that isn't in the training data.
    """
    
    def __init__(self, tgi_endpoint: str, model_name: str = "Qwen2.5-0.5B-Instruct", 
                 summarizer_adapter: str = "general"):
        """
        Initialize WebSearch Expert.
        
        Args:
            tgi_endpoint: TGI server endpoint URL
            model_name: Base model name used by TGI server
            summarizer_adapter: LoRA adapter name for summarization
        """
        self.tgi_endpoint = tgi_endpoint
        self.model_name = model_name
        self.summarizer_adapter = summarizer_adapter
        
        print(f"🌐 Initialized WebSearch Expert with TGI endpoint: {tgi_endpoint}")

    def search_and_summarize(self, query: str, max_tokens: int = 256) -> str:
        """
        Fetch search results from DuckDuckGo and summarize them using TGI.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens for summarization
            
        Returns:
            Summarized answer based on search results
        """
        try:
            print(f"🔍 Searching for: {query}")
            
            # Get web search results
            background_info = web_search(query, max_snippet_len=1000)
            
            if "❌" in background_info or "Error" in background_info:
                return background_info
            
            # Create summarization prompt
            web_preprompt = (
                "You are a helpful assistant. Based on the following web search results, "
                "provide a comprehensive answer to the user's question. "
                "If the web search results are not relevant, say so.\n\n"
                f"[WEB SEARCH RESULTS]\n{background_info}\n\n[USER QUESTION]\n"
            )
            
            # Build TGI payload for summarization
            payload = {
                "model": self.model_name,
                "prompt": web_preprompt + query,
                "adapter_id": self.summarizer_adapter,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            }
            
            print(f"🚀 Summarizing web results using TGI with '{self.summarizer_adapter}' adapter")
            
            # Make request to TGI server
            response_text, success = make_tgi_request(self.tgi_endpoint, payload)
            
            if success:
                return f"🌍 Web-based answer:\n{response_text}"
            else:
                return f"❌ Could not summarize web results: {response_text}"
                
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return f"Error: Could not perform web search - {str(e)}"


class TGIRouterSystem:
    """
    Router system that uses ALBERT-based classification for selecting
    the best expert for a given task, optimized for TGI integration.
    
    If the model is uncertain (confidence below threshold), it routes to web search.
    This approach provides intelligent routing based on text analysis using
    quantized ONNX models for fast inference.
    """
    
    def __init__(self, router_model_path: str = "evaluation/router_fp32_v2/model.onnx", 
                 general_expert_id: int = 1, fallback_threshold: float = 0.35):
        """
        Initialize ALBERT Router.
        
        Args:
            router_model_path: Path to trained ONNX router model
            general_expert_id: ID of the general expert for fallback
            fallback_threshold: Confidence threshold below which to use fallback
        """
        print(f"🔄 Loading ALBERT Router: {router_model_path}")
        
        try:
            self.router = ALBERTRouter(
                onnx_model_path=router_model_path,
                general_expert_id=general_expert_id,
                fallback_threshold=fallback_threshold
            )
            print("✅ ALBERT Router loaded and ready")
            
        except Exception as e:
            print(f"❌ Could not load ALBERT Router: {e}")
            raise

    def forward(self, input_text: str, k: int = 3) -> int:
        """
        Process input and select the best expert.
        
        Args:
            input_text: Text to analyze for routing
            k: Number of top experts to consider
            
        Returns:
            Expert index (0-3) or 4 for web search if uncertain
        """
        try:
            # Use the ONNX router for fast inference
            selected_expert = self.router.forward(input_text, k=k)
            
            print(f"🎯 Router selected expert: {selected_expert}")
            return selected_expert
            
        except Exception as e:
            print(f"❌ Router error: {e}")
            return self.router.general_expert_id  # Fallback to general expert


class TGIMoESystem:
    """
    Main class for the TGI-based Mixture-of-Experts system.
    
    Coordinates ALBERT router, TGI expert management, and chat logging
    for a complete MoE-based AI system using TGI as the inference backend.
    """
    
    def __init__(self, tgi_endpoint: str = "http://localhost:8080/v1/completions", 
                 base_model_name: str = "Qwen2.5-0.5B-Instruct",
                 router_model_path: str = "evaluation/router_fp32_v2/model.onnx"):
        """
        Initialize the TGI MoE system.
        
        Args:
            tgi_endpoint: TGI server endpoint URL
            base_model_name: Base model name used by TGI server
            router_model_path: Path to ONNX router model
        """
        print("🚀 Initializing FEDLORAsemble TGI MoE System...")
        
        self.tgi_endpoint = tgi_endpoint
        self.base_model_name = base_model_name
        
        # Initialize router
        self.router = TGIRouterSystem(router_model_path)
        
        # Expert configuration: id -> (adapter_name, preprompt, max_tokens)
        self.expert_config = {}
        self.tgi_experts = {}  # Loaded TGI experts

        # WebSearch Expert (always available)
        self.web_expert = WebSearchExpert(tgi_endpoint, base_model_name)

        # Chat logging
        self.chat_log = []
        self.log_file = "chat_history.json"
        
        # Performance measurement
        self.start_time = time.time()
        
        print(f"✅ TGI MoE System initialized with endpoint: {tgi_endpoint}")

    def add_expert(self, expert_id: int, adapter_name: str, preprompt: str = "", max_tokens: int = 500):
        """
        Register a TGI expert with its LoRA adapter configuration.
        
        Args:
            expert_id: Unique expert identifier (0-3 for domain experts, 4 for web search)
            adapter_name: Name of the LoRA adapter to use with TGI
            preprompt: Domain-specific preprompt for the expert
            max_tokens: Maximum tokens to generate
        """
        if expert_id == 4:
            print("⚠️ Expert ID 4 is reserved for web search")
            return
            
        self.expert_config[expert_id] = (adapter_name, preprompt, max_tokens)
        print(f"🔹 Registered Expert {expert_id} with adapter '{adapter_name}'")

    def get_expert(self, expert_id: int) -> Optional[TGIExpert]:
        """
        Get or create a TGI expert for the given ID.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            TGI expert instance or None if not configured
        """
        if expert_id == 4:
            # Web search expert is handled separately
            return None
            
        if expert_id in self.tgi_experts:
            return self.tgi_experts[expert_id]

        if expert_id not in self.expert_config:
            print(f"⚠️ No expert registered for ID {expert_id}")
            return None

        adapter_name, preprompt, max_tokens = self.expert_config[expert_id]
        
        try:
            expert = TGIExpert(
                expert_id=expert_id,
                adapter_name=adapter_name,
                tgi_endpoint=self.tgi_endpoint,
                preprompt=preprompt,
                max_tokens=max_tokens,
                model_name=self.base_model_name
            )
            
            self.tgi_experts[expert_id] = expert
            print(f"✅ Created TGI Expert {expert_id}")
            return expert
            
        except Exception as e:
            print(f"❌ Could not create Expert {expert_id}: {e}")
            return None

    def forward(self, input_text: str, temperature: float = 0.7) -> str:
        """
        Main function that processes input through the entire TGI MoE system.
        
        Flow:
        1. ALBERT router analyzes input
        2. Selects appropriate expert (or web search)
        3. Makes TGI request with specific LoRA adapter
        4. Generates response
        5. Logs interaction
        
        Args:
            input_text: User's input text
            temperature: Sampling temperature for generation
            
        Returns:
            Generated response from selected expert
        """
        request_start = time.time()
        
        # Router selects expert
        selected_expert = self.router.forward(input_text)

        # Handle web search
        if selected_expert == 4:
            print(f"🌍 Web search activated for: '{input_text[:50]}...'")
            response = self.web_expert.search_and_summarize(input_text)
            expert_name = "WebSearch"
        else:
            # Get and use selected TGI expert
            print(f"🤖 ALBERT Router selected Expert {selected_expert}")
            expert = self.get_expert(selected_expert)
            
            if expert is None:
                response = f"❌ Expert {selected_expert} is not available."
                expert_name = f"Expert_{selected_expert}_UNAVAILABLE"
            else:
                response = expert.forward(input_text, temperature)
                expert_name = f"TGI_Expert_{selected_expert}"

        # Calculate processing time
        request_time = time.time() - request_start
        
        # Log interaction
        self.save_chat_history(input_text, selected_expert, response, request_time)
        
        print(f"⏱️ Response generated in {request_time:.2f} seconds")
        
        return response

    def save_chat_history(self, prompt: str, expert: int, response: str, processing_time: float):
        """
        Save chat history to JSON file for analysis and debugging.
        
        Args:
            prompt: User's input
            expert: Selected expert index
            response: Generated response
            processing_time: Time to process the request
        """
        chat_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "expert": expert,
            "response": response,
            "processing_time": round(processing_time, 2),
            "tgi_endpoint": self.tgi_endpoint,
            "base_model": self.base_model_name
        }

        # Add to internal log
        self.chat_log.append(chat_entry)

        try:
            # Load existing history
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        history = []
            else:
                history = []

            # Add new entry
            history.append(chat_entry)

            # Save to file
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"⚠️ Could not save chat history: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics for monitoring.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "num_experts_registered": len(self.expert_config),
            "num_experts_loaded": len(self.tgi_experts),
            "total_interactions": len(self.chat_log),
            "uptime_seconds": time.time() - self.start_time,
            "tgi_endpoint": self.tgi_endpoint,
            "base_model": self.base_model_name
        }


# ----------------------
# Main program and demonstration
# ----------------------
if __name__ == "__main__":
    """
    Demonstrates the TGI MoE system with different types of questions
    to showcase intelligent routing between experts.
    """
    
    print("=" * 60)
    print("🧠 FEDLORAsemble TGI MoE System - Demonstration")
    print("=" * 60)
    
    # Initialize the TGI system
    # Make sure TGI server is running on localhost:8080
    moe = TGIMoESystem(
        tgi_endpoint="http://localhost:8080/v1/completions",
        base_model_name="Qwen2.5-0.5B-Instruct"
    )

    # Register specialized experts with their LoRA adapters
    print("\n📋 Registering TGI experts...")
    
    # Expert 0: Reasoning and analysis  
    moe.add_expert(
        expert_id=0,
        adapter_name="reasoning",
        preprompt="You are an expert in logical reasoning. Analyze this step by step:\n",
        max_tokens=800
    )
    
    # Expert 1: General conversation
    moe.add_expert(
        expert_id=1,
        adapter_name="general",
        preprompt="You are a friendly and helpful AI assistant. Respond informatively and engagingly:\n",
        max_tokens=600
    )
    
    # Expert 2: Mathematics and calculations
    moe.add_expert(
        expert_id=2,
        adapter_name="math",
        preprompt="You are a math expert. Solve this step by step with clear calculations:\n",
        max_tokens=700
    )
    
    # Expert 3: Programming and code
    moe.add_expert(
        expert_id=3,
        adapter_name="code",
        preprompt="You are a programming expert. Write clean, well-commented code and explain the solution:\n",
        max_tokens=1000
    )

    # Test questions to demonstrate routing
    test_questions = [
        "Who is Donald Duck?",  # General question -> Expert 1
        "What is the size of the moon?",  # Facts/web search
        "Calculate the integral of x^2 from 0 to 3",  # Mathematics -> Expert 2
        "Write a Python function to sort a list",  # Programming -> Expert 3
        "Explain why federated learning is important for privacy"  # Reasoning -> Expert 0
    ]

    print(f"\n🧪 Testing system with {len(test_questions)} questions...")
    print("-" * 60)

    # Run tests
    for i, question in enumerate(test_questions, 1):
        print(f"\n💬 Question {i}: {question}")
        print("🔄 Processing...")
        
        try:
            response = moe.forward(question)
            print(f"🤖 Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        
        print("-" * 40)

    # Show system statistics
    print(f"\n📊 System Statistics:")
    stats = moe.get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n✅ Demonstration completed!")
    print(f"📝 Chat history saved to: {moe.log_file}")
    print("\n💡 Note: Make sure TGI server is running with the appropriate LoRA adapters.")
    print("   Start TGI with: docker-compose up -d (in evaluation/ directory)")
    print("=" * 60)
