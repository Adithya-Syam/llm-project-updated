# ⚖️ LLM-Based Court Case Judgment Prediction System

An AI-powered legal assistant that leverages large language models (LLMs) to analyze, summarize, and predict outcomes of Indian Supreme Court cases. This project aims to assist lawyers, legal researchers, and students with faster insights and better decision-making using modern NLP techniques.

## 🚀 Features

- 🔮 **Case Outcome Prediction**  
  Predicts the judgment (favorable/unfavorable) of a given legal case using fine-tuned LLMs.

- 📚 **Legal Precedent Retrieval**  
  Fetches the most relevant past judgments to support current cases using semantic similarity and vector search.

- 📝 **Case Summarization**  
  Generates concise and accurate summaries from lengthy court documents.

- 🧠 **Bias & Consistency Check**  
  Identifies potential biases in legal arguments and ensures consistency across multiple case judgments.

- 🤖 **AI-Powered Chatbot**  
  A chatbot trained on Indian legal documents to answer queries based on legal context.

## 🧠 Tech Stack

- **Language Models:** GPT-4, Legal-BERT, Indian-LegalGPT (Fine-tuned)
- **Frameworks & Libraries:** Hugging Face Transformers, LangChain, FAISS, PyTorch
- **Backend:** FastAPI / Flask (optional for deployment)
- **Frontend:** Streamlit (for demo)
- **Storage:** Pinecone / FAISS for vector DB
- **Deployment:** Google Colab / Local GPU / Streamlit Cloud
- **Data Sources:** Indian Supreme Court judgments (sourced from [Indian Kanoon](https://indiankanoon.org) / SCC Online)

## 🧪 Methodology

1. **Data Collection**: 
   - Curated a dataset of Indian Supreme Court judgments.
2. **Preprocessing**: 
   - Cleaned text, removed headers, handled OCR issues.
3. **Fine-Tuning**:
   - Used Legal-BERT and GPT-based models fine-tuned on Indian legal corpus.
4. **RAG-based System**:
   - Combined retrieval (using FAISS) and generation (via LLM) for better contextual responses.
5. **Evaluation**:
   - F1-score, Accuracy for classification.
   - BLEU and ROUGE for summarization quality.

## 📊 Results

| Functionality         | Model Used        | Accuracy / Score |
|----------------------|------------------|------------------|
| Case Outcome Prediction | LegalBERT + SVM | 81.4% Accuracy   |
| Summarization        | GPT-3.5-turbo     | ROUGE-L: 0.62    |
| Precedent Retrieval  | FAISS + Embedding | Top-5: 93% match |
| Chatbot              | RAG with LangChain| Domain-specific coherence |

## 📂 Directory Structure

