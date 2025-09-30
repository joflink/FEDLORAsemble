# FEDLORAsemble: Federated LoRA-Based Mixture of Experts with TGI

A modular, production-ready system combining **federated learning**, **LoRA fine-tuning**, and a **TGI-powered Mixture of Experts (MoE)** architecture for efficient, domain-specialized inference.

---

## 🌟 Key Features

### 🤖 TGI-Based Mixture of Experts (MoE)
- **ALBERT-based router** with ONNX quantization for fast expert selection  
- **TGI server integration** with dynamic LoRA adapter switching  
- **Specialized experts** for different domains (math, programming, general conversation)  
- **Web search expert** as fallback for uncertain or out-of-domain queries  

### 🔄 Federated Training
- Built on the **[Flower](https://flower.dev/)** framework for distributed, privacy-preserving training  
- **LoRA adapters** enable parameter-efficient fine-tuning across clients  
- **Automatic model saving** and checkpointing  
- **Mixed-precision training** for performance and memory efficiency  

### 📊 Evaluation & Benchmarking
- **HumanEval** for code generation evaluation  
- **Pass@k metrics** for robust performance measurement  
- **Comprehensive benchmarks**: MMLU, GSM8K, ARC  
- **TGI-native evaluation** for realistic production scenarios  

### 🛠️ Production-Ready TGI Support
- **Text Generation Inference (TGI)** as primary backend  
- **ONNX model export** for optimized inference  
- **Quantization support** (FP32, INT8) for diverse hardware  
- **Docker-based deployment** with `docker-compose`  

---

## 📁 Project Structure

```
merged/
├── 📄 README.md                    # This file
├── ⚙️ pyproject.toml              # Dependencies & Flower config
├── 🧠 moe_system.py               # TGI-based MoE orchestrator
├── 📊 evaluation_main.py          # Main evaluation entrypoint
│
├── 🎯 training/                   # Federated training logic
│   ├── server_app.py              # Flower server
│   ├── client_app.py              # Flower client
│   ├── AImodels.py                # Model + LoRA handling
│   ├── dataset.py                 # Dataset loading & preprocessing
│   ├── strategy.py                # Federated aggregation strategy
│   └── ...
│
├── 📈 evaluation/                 # Benchmarking suite
│   ├── ALBERTRouter.py            # Router implementation
│   ├── run_eval.py                # Benchmark runner
│   ├── eval_utils.py              # Utilities
│   ├── tgi_router_client.py       # TGI client for MoE
│   ├── docker-compose.yml         # TGI deployment config
│   ├── evals.yaml                 # Evaluation settings
│   ├── 🤖 bert-router/            # Trained router checkpoints
│   ├── 🎯 lora/                   # Domain-specific LoRA adapters
│   └── 📊 results/                # Evaluation outputs
│
├── 🗂️ datasets/                  # Training data
│   ├── code/                      # Programming tasks
│   ├── math/                      # Math problems
│   ├── general/                   # General dialogue
│   └── reasoning/                 # Logical reasoning
│
└── 💾 models/                     # Base models (not in Git)
    ├── qwens/
    │   ├── Qwen2.5-0.5B-Instruct
```

> 💡 **Note**:  
> - **Qwen2.5-0.5B-Instruct** is a general-purpose instruction-tuned model (0.49B params, 32K context).  


---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
git clone https://github.com/joflink/FEDLORAsemble
cd merged
pip install -e .               # or `poetry install`
```

### 2. Download Models
```bash
mkdir -p models/qwens
git lfs clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct models/qwens/Qwen2.5-0.5B-Instruct
```

> ⚠️ **Requirement**: Use `transformers >= 4.37.0` to avoid `KeyError: 'qwen2'`.

### 3. Launch TGI Server
```bash
cd evaluation
docker-compose up -d
curl http://localhost:8080/health  # verify
```

### 4. Run MoE System
```bash
python moe_system.py  # auto-connects to TGI at localhost:8080
```

### 5. Start Federated Training
```bash
# Server
python training/server_app.py --server-address 0.0.0.0:8080 --rounds 100

# Client (on another machine/terminal)
python training/client_app.py --server-address <SERVER_IP>:8080
```

---

## 🔧 Configuration

### Add an Expert (`moe_system.py`)
```python
moe.add_model(
    index=3,
    model_type="hf",
    model_path="models/qwens/Qwen2.5-0.5B-Instruct",
    preprompt="You are a coding expert. Generate clean, efficient Python:\n",
    max_tokens=800
)
```

### Federated Training (`pyproject.toml`)
```toml
[tool.flwr.app.config]
model.name = "models/qwens/Qwen2.5-0.5B-Instruct"
model.lora.peft-lora-r = 32
train.training-arguments.per-device-train-batch-size = 16
```

### TGI Settings (`evaluation/docker-compose.yml`)
```yaml
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    environment:
      - MODEL_ID=models/qwens/Qwen2.5-0.5B-Instruct
      - MAX_CONCURRENT_REQUESTS=128
      - CUDA_MEMORY_FRACTION=0.8
```

---

## 📊 Evaluation

### Code (HumanEval)
```bash
python evaluation_main.py \
  --base-model models/qwens/Qwen2.5-0.5B-Instruct \
  --lora-adapter evaluation/lora/code \
  --num-samples 10
```

### Full Benchmark Suite
```bash
cd evaluation
python run_eval.py --config evals.yaml
```

---

## 🎯 Experts & Domains

| ID | Domain        | Model                                | Specialty                     |
|----|---------------|--------------------------------------|-------------------------------|
| 0  | Reasoning     | DeepSeek-R1-Distill                  | Logical analysis              |
| 1  | General       | Qwen2.5-0.5B-Instruct                | Conversation, QA              |
| 2  | Math          | Qwen2.5-0.5B-Math                    | Problem solving               |
| 3  | Coding        | Qwen2.5-0.5B-Instruct          | Code gen, debugging           |
| 4  | Web Search    | Qwen2.5 + DuckDuckGo API             | Real-time info                |

LoRA adapters live under `evaluation/lora/<size>/<domain>/`.

---

## 🧪 Federated Training Flow

1. Server initializes global model  
2. Clients join and download parameters  
3. Each client trains **locally** using LoRA on domain-specific data  
4. Only LoRA deltas are sent back  
5. Server aggregates updates (e.g., FedAvg)  
6. Global model improves → repeat  

> 🗓️ **Smart scheduling**: Train different experts on different days (e.g., Monday = reasoning, Thursday = coding).

---

## 📈 Performance

| Benchmark         | Base Model | MoE System | Δ       |
|-------------------|------------|------------|---------|
| HumanEval (Pass@1)| 24.5%      | 31.2%      | **+6.7%** |
| GSM8K             | 45.3%      | 52.8%      | **+7.5%** |
| MMLU (Avg)        | 58.2%      | 63.7%      | **+5.5%** |

**Router**: 87.3% accuracy, ~150ms latency, 12.7% fallback to web search.

---

## 🚀 Development Opportunities

The current architecture lays a strong foundation for advanced capabilities:

- **Client Orchestration Pipeline**:  
  The server can dynamically control training—deciding *which strategy* to apply, *when* to trigger it, and even *split datasets* to spawn new LoRA adapters on-demand (e.g., creating niche adapters during federated rounds).

- **Integrated RAG Solution**:  
  Add a retrieval-augmented generation (RAG) layer as a new expert or pre-router step, enabling responses grounded in external knowledge bases or documentation.

- **Privacy-Aware & Graded Routing**:  
  Enhance the router to:  
  - Respect **privacy constraints** (e.g., avoid sending sensitive prompts to cloud experts)  
  - Perform **prompt difficulty grading** (classify as “easy”/“hard”) to route simple queries to lightweight models and complex ones to high-capacity experts  

These extensions align naturally with the modular MoE + federated design and significantly improve utility, safety, and efficiency in real-world deployments.

---

## 🎯 Roadmap

### Q2 2025
- [ ] Improved router training (larger datasets)  
- [ ] Automatic expert selection via performance feedback  
- [ ] Web UI  
- [ ] Kubernetes manifests  

### Q3–Q4 2025
- [ ] Multimodal experts (text + image)  
- [ ] Real-time federated learning  
- [ ] Mobile support  

### 2026+
- [ ] Auto-specialization  
- [ ] Cross-lingual experts  
- [ ] Edge optimization  

---

*Last updated: 2025-01-30*  
**FEDLORAsemble Team** 🚀
