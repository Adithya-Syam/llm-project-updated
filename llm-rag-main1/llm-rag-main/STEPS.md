# 🚀 Legal LLM Fine-tuning and Deployment Steps

## 📋 **Table of Contents**
1. [System Requirements](#system-requirements)
2. [Initial Setup](#initial-setup)
3. [Fine-tuning Steps](#fine-tuning-steps)
4. [Post-Training Steps](#post-training-steps)
5. [Web Interface Launch](#web-interface-launch)
6. [GPU vs CPU Deployment](#gpu-vs-cpu-deployment)
7. [Troubleshooting](#troubleshooting)

---

## 🖥️ **System Requirements**

### **Minimum Requirements (CPU/Mac)**
- **RAM**: 8GB+ 
- **Storage**: 20GB free space
- **Python**: 3.8+
- **Time**: 1-2 hours for training

### **Recommended Requirements (GPU)**
- **GPU**: 8GB+ VRAM (RTX 3070, 4060, or better)
- **RAM**: 16GB+
- **Storage**: 50GB free space
- **CUDA**: 11.8+ or 12.x
- **Time**: 15-30 minutes for training

---

## 🔧 **Initial Setup**

### **Step 1: Environment Setup**
```bash
# Clone/navigate to project directory
cd law-llm-finetune-rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **Step 2: Data Preparation**
```bash
# Process legal documents dataset
python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed --clean_text

# Verify data processing
ls data/processed/
# Should show: train_split.json, validation.json, test.json
```

---

## 🎯 **Fine-tuning Steps**

### **Option A: CPU/Mac Training (Current Setup)**
```bash
# Train with CPU/MPS optimized settings
python scripts/train_summarization.py

# Expected output:
# - Device: Apple Silicon (MPS) or CPU
# - Precision: fp32
# - Batch size: 1
# - Training time: 45-90 minutes
```

### **Option B: GPU Training (For Deployment)**
```bash
# Train with GPU optimized settings
python scripts/train_summarization.py --config config/training_config_gpu.yaml

# Expected output:
# - Device: CUDA (GPU name)
# - Precision: fp16
# - Batch size: 8
# - Training time: 10-20 minutes
```

### **Training Progress Indicators**
```
✅ Model Loading: Llama-3.2-1B-Instruct downloaded (~2.5GB)
✅ LoRA Setup: ~5.6M trainable parameters (0.45% of total)
✅ Data Loading: 6,178 training + 1,545 validation samples
✅ Training Progress: Loss decreasing over epochs
✅ Model Saving: Final model saved to ./results/
```

---

## 📊 **Post-Training Steps**

### **Step 1: Build RAG Index**
```bash
# Build vector database for legal document search
python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index

# Expected output:
# - Embeddings model loaded
# - 7,823+ documents processed
# - FAISS index created
# - Index saved to data/rag_index/
```

### **Step 2: Test the System**
```bash
# Quick system test
python -c "
from src.models.summarizer import LegalSummarizer
from src.rag.legal_rag import LegalRAG

# Test fine-tuned model
summarizer = LegalSummarizer.from_pretrained('./results')
print('✅ Fine-tuned model loaded')

# Test RAG system
rag = LegalRAG.load('data/rag_index')
results = rag.search('property tax', top_k=3)
print(f'✅ RAG system: {len(results)} results found')
"
```

### **Step 3: Verify Model Performance**
```bash
# Run evaluation (optional)
python src/utils/evaluation.py

# Check model files
ls results/
# Should show: adapter_config.json, adapter_model.safetensors, etc.
```

---

## 🌐 **Web Interface Launch**

### **Method 1: Streamlit (Recommended)**
```bash
# Launch web application
streamlit run web_app/app.py

# Expected output:
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

### **Method 2: Using Makefile**
```bash
# Alternative launch method
make webapp
```

### **Method 3: Production Deployment**
```bash
# For production servers
streamlit run web_app/app.py --server.port 8501 --server.address 0.0.0.0
```

### **Web Interface Features**
- **💬 Chat Tab**: Interactive legal AI assistant
- **📄 Summarization Tab**: Upload and summarize legal documents
- **🔍 Search Tab**: Legal precedent and case search
- **📊 Analytics Tab**: System statistics and export options

---

## ⚖️ **GPU vs CPU Deployment**

### **🖥️ CPU/Mac Deployment (Current)**
```bash
# Configuration used
Config: config/training_config.yaml
Batch Size: 1
Sequence Length: 512
Precision: fp32
Memory Usage: 4-8GB
Training Time: 45-90 minutes
Inference Speed: 2-5 seconds per query

# Launch command
python scripts/train_summarization.py
streamlit run web_app/app.py
```

### **🚀 GPU Deployment (Production)**
```bash
# Configuration for GPU systems
Config: config/training_config_gpu.yaml
Batch Size: 8
Sequence Length: 1024
Precision: fp16
Memory Usage: 8-16GB VRAM
Training Time: 10-20 minutes
Inference Speed: 0.5-1 second per query

# Launch commands
python scripts/train_summarization.py --config config/training_config_gpu.yaml
streamlit run web_app/app.py
```

### **🔄 Migration from CPU to GPU**
```bash
# 1. Copy trained model and data
scp -r results/ user@gpu-server:/path/to/project/
scp -r data/ user@gpu-server:/path/to/project/

# 2. On GPU server, install dependencies
pip install -r requirements.txt

# 3. Optional: Retrain with GPU config for better performance
python scripts/train_summarization.py --config config/training_config_gpu.yaml

# 4. Launch web interface
streamlit run web_app/app.py
```

---

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Memory Errors**
```bash
# Error: "MPS backend out of memory"
# Solution: Reduce batch size further
# Edit config/training_config.yaml:
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

#### **2. CUDA Not Available**
```bash
# Error: "CUDA not available"
# Solution: Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **3. Model Loading Errors**
```bash
# Error: "No module named 'torch'"
# Solution: Use correct Python environment
which python
conda activate base  # or your environment
```

#### **4. Port Already in Use**
```bash
# Error: "Port 8501 is already in use"
# Solution: Use different port
streamlit run web_app/app.py --server.port 8502
```

#### **5. Slow Training**
```bash
# Issue: Training too slow on CPU
# Solution: Use smaller dataset or switch to GPU
# Reduce epochs in config:
num_train_epochs: 1
```

---

## 📈 **Performance Optimization**

### **For CPU/Mac Systems**
```yaml
# Optimized settings for limited resources
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_length: 256
num_train_epochs: 1
```

### **For GPU Systems**
```yaml
# Optimized settings for GPU
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
max_length: 1024
num_train_epochs: 3
fp16: true
```

---

## 🎯 **Quick Start Commands**

### **Complete Setup (CPU/Mac)**
```bash
# One-time setup
pip install -r requirements.txt
python scripts/preprocess_data.py
python scripts/train_summarization.py
python scripts/build_rag_index.py
streamlit run web_app/app.py
```

### **Complete Setup (GPU)**
```bash
# One-time setup for GPU
pip install -r requirements.txt
python scripts/preprocess_data.py
python scripts/train_summarization.py --config config/training_config_gpu.yaml
python scripts/build_rag_index.py
streamlit run web_app/app.py
```

### **Quick Launch (After Training)**
```bash
# Just launch the web interface
streamlit run web_app/app.py
```

---

## 📞 **Support & Next Steps**

### **Verification Checklist**
- [ ] ✅ Dependencies installed
- [ ] ✅ Data preprocessed (7,823 documents)
- [ ] ✅ Model fine-tuned (results/ folder exists)
- [ ] ✅ RAG index built (data/rag_index/ exists)
- [ ] ✅ Web interface accessible (http://localhost:8501)
- [ ] ✅ Chat functionality working
- [ ] ✅ Document summarization working
- [ ] ✅ Legal search working

### **Performance Targets**
| Metric | CPU/Mac | GPU |
|--------|---------|-----|
| Training Time | 45-90 min | 10-20 min |
| Inference Speed | 2-5 sec | 0.5-1 sec |
| Memory Usage | 4-8 GB | 8-16 GB |
| ROUGE-L Score | 0.35+ | 0.40+ |

### **Project Delivery**
```bash
# Package for delivery
tar -czf legal-llm-system.tar.gz \
  --exclude='dataset' \
  --exclude='wandb' \
  --exclude='__pycache__' \
  .

# Include these key files:
# - results/ (trained model)
# - data/rag_index/ (vector database)
# - config/ (configurations)
# - src/ (source code)
# - web_app/ (interface)
# - requirements.txt
# - STEPS.md (this file)
```

---

## 🏆 **Success! Your Legal LLM System is Ready**

Your system now includes:
- ✅ **Fine-tuned Llama 1B** for legal document summarization
- ✅ **RAG System** with 7,823+ legal documents
- ✅ **Interactive Web Interface** with chat, search, and analytics
- ✅ **GPU/CPU Compatibility** for flexible deployment
- ✅ **Production Ready** architecture

**Access your system at: http://localhost:8501** 🎉
