# ⚖️ LLM-RAG System for Indian Jurisprudence

> Bridging Legal Expertise and Public Access

This project presents a powerful Legal AI system tailored for the Indian legal ecosystem. By combining Retrieval-Augmented Generation (RAG), multilingual NLP, and fine-tuned LLMs, our system addresses the challenges of court overload, language barriers, hallucinated outputs, and the lack of transparent reasoning in legal AI.



## 🚀 Key Functionalities

- 🧠 **Chain-of-Thought Case Analysis**  
  Transparent step-by-step reasoning for judgments and legal arguments.

- 📝 **Dual-View Summarization**  
  Generates both *formal* and *layman-friendly* case summaries.

- ⚖️ **Bias & Hallucination Detector**  
  Detects AI-generated biases and invalid citations in responses.

- 🌐 **Multilingual Legal Chatbot**  
  Converses in 10+ Indian languages to bridge accessibility gaps.

- 📘 **Statutory Section Predictor**  
  Automatically maps fact patterns to relevant IPC/CrPC sections.

- 📊 **Integrated Performance Evaluator**  
  Evaluates output on accuracy, readability, bias, and section validity.

## 🧩 Motivation

- **40M+** pending court cases in India and thousands of new judgments weekly.
- Majority of the population cannot interpret legal documents written in English or legal jargon.
- Existing AI tools are not explainable, hallucinate information, and are rarely evaluated for factual correctness.
- Need for **low-cost**, **auditable**, and **multilingual** legal AI tools for public empowerment.

## 🛠️ System Architecture

```
Input (Query/Case)
      ↓
RAG Pipeline (Legal Docs + LLM)
      ↓
Chain-of-Thought Generator
      ↓
Bias & Hallucination Checker
      ↓
Summarization (Formal + Layman)
      ↓
Statutory Section Prediction
      ↓
Multilingual Chat Interface + Quality Metrics
```

## 🔬 Training & Fine-Tuning

- **Model**: LoRA fine-tuned 1B-parameter LLM
- **Hardware**: 8 GB GPU (optimized for cost efficiency)
- **Precision**: FP16
- **Training Time**: ~10–20 minutes
- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
  - Target Modules: Query, Key, Value, Output Projections

## 📊 Evaluation Metrics

| Module                      | Metric                        | Status |
|----------------------------|-------------------------------|--------|
| Case Outcome Prediction    | Accuracy, F1-score            | ✅     |
| Summarization              | ROUGE, BLEU                   | ✅     |
| Bias & Hallucination Check | Citation validity, Bias flag  | ✅     |
| Section Prediction         | Statute match score           | ✅     |
| Chatbot Responses          | Multilingual fluency, Coherence | ✅  |

## 📚 Dataset

- Curated from **Indian Supreme Court judgments**
- Includes multilingual text samples, legal labels, and statutory mappings.
- Preprocessed to clean headers, annotations, and OCR errors.

## 💡 Novelty

- Four-step **Chain-of-Thought** reasoning engine
- **Dual-view summarizer** (formal + simplified)
- Real-time **bias and hallucination detection**
- **Multilingual legal chatbot** spanning 10 Indian languages
- **Statutory section prediction** using contextual embeddings
- Quantitative performance dashboard for legal AI trustworthiness

## 📦 Setup

```bash
git clone https://github.com/Adithya-Syam/llm-project-updated.git
cd llm-indian-legal-rag
pip install -r requirements.txt
streamlit run app/main.py
```

## 🏁 Future Work

- Expand to High Court and District Court judgments.
- Integrate Explainable AI (XAI) for prediction rationale.
- Enable document upload and live Q&A from PDFs.
- Open-source model and data with public APIs.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It does not constitute legal advice. Always consult a certified legal professional for real-world legal decisions.

## 🙏 Acknowledgments

- [Indian-LegalGPT](https://github.com/jfontestad/Indian-LegalGPT)
- [LangChain](https://www.langchain.com/)
- [LoRA by Hugging Face](https://huggingface.co/docs/peft/index)
