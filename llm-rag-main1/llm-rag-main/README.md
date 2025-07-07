# 🏛️ Legal LLM Fine-tuning and RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A comprehensive AI system for legal document analysis, combining fine-tuned Large Language Models with Retrieval-Augmented Generation (RAG) for intelligent legal assistance.

## 🎯 **Features**

- **🔥 Fine-tuned Llama 3.2-1B** - Specialized for legal document summarization
- **🔍 RAG System** - Semantic search through 7,823+ legal documents
- **🤖 AI Legal Assistant** - Interactive chatbot with conversation memory
- **🌐 Web Interface** - Complete Streamlit application with multiple interaction modes
- **⚖️ Multi-jurisdiction** - Support for Indian and UK legal documents
- **🖥️ Hardware Flexible** - Compatible with CPU, MPS (Apple Silicon), and CUDA

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- 20GB+ free storage

### **Installation**
```bash
# Clone repository
git clone https://github.com/rabieHs/llm-rag.git
cd llm-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **Setup and Launch**
```bash
# 1. Prepare data (requires legal dataset)
python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed

# 2. Build RAG index
python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index

# 3. Optional: Fine-tune model
python scripts/train_summarization.py

# 4. Launch web interface
streamlit run web_app/app.py
```

Access the application at: **http://localhost:8501**

## 📊 **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Legal Dataset │    │  Fine-tuned LLM │    │   RAG System    │
│   7,823+ docs   │───▶│  Llama 3.2-1B   │───▶│ FAISS + Vector │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────────────────────────┐
                       │        Web Interface                │
                       │  💬 Chat  📄 Summarize  🔍 Search  │
                       └─────────────────────────────────────┘
```

## 🛠️ **Components**

### **Data Processing**
- **Legal Document Loader** - Handles multiple jurisdictions and formats
- **Text Preprocessor** - Cleans and normalizes legal text
- **Dataset Splitter** - Creates train/validation/test splits

### **Model Fine-tuning**
- **LoRA Implementation** - Efficient parameter-efficient fine-tuning
- **Legal Prompt Engineering** - Specialized prompts for legal tasks
- **Multi-device Support** - CPU, MPS, and CUDA compatibility

### **RAG System**
- **Vector Database** - FAISS-based semantic search
- **Embeddings** - Sentence-transformers for document encoding
- **Retrieval Engine** - Similarity-based document retrieval

### **Web Interface**
- **Interactive Chat** - AI-powered legal assistant
- **Document Summarization** - Upload and summarize legal documents
- **Legal Search** - Find relevant cases and precedents
- **Analytics Dashboard** - System statistics and performance metrics

## 📈 **Performance**

| Component | CPU/Mac | GPU |
|-----------|---------|-----|
| **Training Time** | 45-90 min | 10-20 min |
| **Inference Speed** | 2-5 sec | 0.5-1 sec |
| **Memory Usage** | 4-8 GB | 8-16 GB |
| **RAG Search** | <1 sec | <0.5 sec |

## 🎯 **Use Cases**

- **Legal Research** - Find relevant cases and precedents
- **Document Analysis** - Summarize complex legal documents
- **Case Preparation** - Research similar cases and legal principles
- **Legal Education** - Interactive learning tool for law students
- **Compliance** - Understand legal requirements and regulations

## 📚 **Documentation**

- **[Complete Documentation](DOCUMENTATION.md)** - Detailed setup and usage guide
- **[Training Steps](STEPS.md)** - Fine-tuning and deployment instructions
- **[API Reference](src/)** - Code documentation and examples

## 🔧 **Configuration**

### **Training Configuration**
```yaml
# config/training_config.yaml
training:
  per_device_train_batch_size: 1  # Adjust for your hardware
  learning_rate: 0.0003
  num_train_epochs: 3
  
model:
  base_model: "meta-llama/Llama-3.2-1B-Instruct"
  max_length: 2048
```

### **RAG Configuration**
```yaml
# config/rag_config.yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  
retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Hugging Face** - For transformers and model hosting
- **Meta AI** - For the Llama model family
- **Sentence Transformers** - For embedding models
- **FAISS** - For efficient vector search
- **Streamlit** - For the web interface framework

## 📞 **Support**

- **Issues** - Report bugs and request features via GitHub Issues
- **Discussions** - Join community discussions in GitHub Discussions
- **Documentation** - Comprehensive guides in the docs folder

## 🔗 **Related Projects**

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://github.com/streamlit/streamlit)

---

**⭐ Star this repository if you find it helpful!**

**🔗 [Live Demo](http://localhost:8501)** | **📖 [Documentation](DOCUMENTATION.md)** | **🚀 [Quick Start](#quick-start)**
