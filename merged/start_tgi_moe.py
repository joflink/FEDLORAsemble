#!/usr/bin/env python3
"""
FEDLORAsemble TGI MoE System - Easy Startup Script
================================================

This script provides an easy way to start and interact with the TGI-based
Mixture of Experts system. It handles TGI server checks and provides
a simple command-line interface.

Usage:
    python start_tgi_moe.py
    python start_tgi_moe.py --endpoint http://localhost:8080/v1/completions
"""

import argparse
import requests
import time
import sys
from moe_system import TGIMoESystem

def check_tgi_server(endpoint: str, max_retries: int = 5) -> bool:
    """
    Check if TGI server is running and responsive.
    
    Args:
        endpoint: TGI server endpoint
        max_retries: Maximum number of connection attempts
        
    Returns:
        True if server is responsive, False otherwise
    """
    # Convert completions endpoint to health endpoint
    health_endpoint = endpoint.replace('/v1/completions', '/health')
    
    print(f"🔍 Checking TGI server at: {health_endpoint}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(health_endpoint, timeout=5)
            if response.status_code == 200:
                print("✅ TGI server is running and healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_retries - 1:
            print(f"⏳ Attempt {attempt + 1}/{max_retries} failed, retrying in 2 seconds...")
            time.sleep(2)
    
    print("❌ TGI server is not responding")
    return False

def setup_default_experts(moe: TGIMoESystem):
    """
    Set up the default expert configuration.
    
    Args:
        moe: TGI MoE system instance
    """
    print("🔧 Setting up default experts...")
    
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
        preprompt="You are a friendly and helpful AI assistant:\n",
        max_tokens=600
    )
    
    # Expert 2: Mathematics
    moe.add_expert(
        expert_id=2,
        adapter_name="math",
        preprompt="You are a math expert. Solve this step by step:\n",
        max_tokens=700
    )
    
    # Expert 3: Programming
    moe.add_expert(
        expert_id=3,
        adapter_name="code",
        preprompt="You are a coding expert. Provide detailed code solutions:\n",
        max_tokens=1000
    )
    
    print("✅ Default experts configured")

def interactive_mode(moe: TGIMoESystem):
    """
    Run interactive chat mode with the MoE system.
    
    Args:
        moe: TGI MoE system instance
    """
    print("\n🎯 Starting interactive mode...")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'stats' to see system statistics")
    print("💡 Type 'help' for more commands")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🤔 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Good bye!")
                break
            
            if user_input.lower() == 'stats':
                stats = moe.get_system_stats()
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            if user_input.lower() == 'help':
                print("\n📚 Available commands:")
                print("   • quit/exit/q  - Exit the program")
                print("   • stats        - Show system statistics")
                print("   • help         - Show this help message")
                print("   • Any other text will be processed by the MoE system")
                continue
                
            if not user_input:
                continue
            
            print("🔄 Processing...")
            start_time = time.time()
            
            response = moe.forward(user_input)
            
            processing_time = time.time() - start_time
            print(f"\n🤖 Assistant: {response}")
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 Good bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test_mode(moe: TGIMoESystem):
    """
    Run batch test with predefined questions.
    
    Args:
        moe: TGI MoE system instance
    """
    test_questions = [
        "What is the capital of France?",
        "Calculate 15 * 23 + 7",
        "Write a Python function to reverse a string", 
        "Explain the concept of recursion",
        "What are the latest developments in AI?"  # Should trigger web search
    ]
    
    print(f"\n🧪 Running batch test with {len(test_questions)} questions...")
    print("-" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n💬 Question {i}: {question}")
        print("🔄 Processing...")
        
        try:
            start_time = time.time()
            response = moe.forward(question)
            processing_time = time.time() - start_time
            
            print(f"🤖 Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"⏱️  Time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    # Show final statistics
    stats = moe.get_system_stats()
    print("\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FEDLORAsemble TGI MoE System")
    parser.add_argument(
        "--endpoint", 
        default="http://localhost:8080/v1/completions",
        help="TGI server endpoint (default: http://localhost:8080/v1/completions)"
    )
    parser.add_argument(
        "--model", 
        default="Qwen2.5-0.5B-Instruct", 
        help="Base model name (default: Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--router", 
        default="evaluation/router_fp32_v2/model.onnx",
        help="Router model path (default: evaluation/router_fp32_v2/model.onnx)"
    )
    parser.add_argument(
        "--mode", 
        choices=["interactive", "batch"], 
        default="interactive",
        help="Run mode: interactive chat or batch test (default: interactive)"
    )
    parser.add_argument(
        "--skip-health-check", 
        action="store_true",
        help="Skip TGI server health check"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting FEDLORAsemble TGI MoE System")
    print(f"🌐 TGI Endpoint: {args.endpoint}")
    print(f"🤖 Base Model: {args.model}")
    print(f"🧠 Router Model: {args.router}")
    
    # Check TGI server health
    if not args.skip_health_check:
        if not check_tgi_server(args.endpoint):
            print("\n❌ TGI server is not running or not responding")
            print("💡 To start TGI server:")
            print("   cd evaluation")
            print("   docker-compose up -d")
            print("\n💡 Or skip health check with --skip-health-check")
            sys.exit(1)
    
    # Initialize MoE system
    try:
        moe = TGIMoESystem(
            tgi_endpoint=args.endpoint,
            base_model_name=args.model,
            router_model_path=args.router
        )
        
        # Set up default experts
        setup_default_experts(moe)
        
        # Run in selected mode
        if args.mode == "interactive":
            interactive_mode(moe)
        else:
            batch_test_mode(moe)
            
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
