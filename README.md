# Bhagavad-Gita-ChatBot
A **RAG-LLM–based** AI-powered chatbot that delivers wisdom and insights from the sacred **Bhagavad Gita**. Built using Streamlit for the UI, LangChain for semantic search orchestration, and Hugging Face models for intelligent retrieval and response generation.

# ✨ Features

**Multi-Modal Retrieval:** Combines semantic search, theme-based retrieval, and keyword matching for comprehensive answers

**Multiple Response Styles:** Choose from balanced, student-friendly, philosophical, or devotional answer styles

**Interactive Web Interface:** Clean, intuitive Streamlit interface with real-time chat functionality

**Verse Context:** Shows relevant verses that inform each answer for transparency

**Theme Classification:** Automatically categorizes verses by spiritual themes (dharma, karma, moksha, etc.)

**Keyword Extraction:** TF-IDF based keyword extraction for enhanced searchability



# 🛠️ Technology Stack

## AI/ML:

Hugging Face Transformers

LangChain for RAG (Retrieval-Augmented Generation)

ChromaDB for vector storage

Sentence Transformers for embeddings

## NLP: 
NLTK, scikit-learn (TF-IDF)

## Data Processing: 
Pandas, NumPy, Google Sheets

## Frontend:
Streamlit + ( With the help of GenAI tools) 

## Model Configuration

**Embedding Model:** sentence-transformers/all-mpnet-base-v2

**LLM:** openai/gpt-oss-20b (via Hugging Face)

**Vector Store:** ChromaDB (temporary storage)


# 🎯 How It Works
## 1. Data Processing

Extracts verses from CSV with metadata

Performs keyword extraction using TF-IDF

Classifies verses into spiritual themes


## 2. Multi-Modal Retrieval

**Semantic Search:** Vector similarity using sentence transformers

**Theme-Based:** Matches questions to spiritual themes

**Keyword Matching:** TF-IDF based keyword relevance


## 3. Answer Generation

Combines retrieved verses with weighted scoring

Generates contextual prompts based on selected style

Uses Hugging Face models for natural language generation

## 💡 Example Questions

"What is dharma according to the Bhagavad Gita?"

"How should one perform karma ?"

"What does Krishna say about detachment?"

"What is the path to self-realization?"

"How to control the mind through meditation?"


## 🎨 Answer Styles

**Balanced:** Well-rounded responses balancing depth with accessibility

**Student:** Clear, simple explanations perfect for learners

**Philosophical:** Deep, contemplative insights with universal principles

**Devotional:** Heart-centered responses emphasizing love and surrender
 

# Usage

**Upload Data:** Upload your Bhagavad Gita CSV file using the sidebar ("GitaText - Data_for_Gita_ChatBot_Cleaned.csv" or in formate mentioned)

**Configure:** Enter your Hugging Face API token

**Initialize:** Click "Initialize ChatBot" to set up the AI models  ( may takes time while initialization !! ... Will improve)

**Ask Questions:** Start asking questions about the Gita's teachings

**Choose Style:** Select your preferred answer style (balanced, student, philosophical, devotional)

# Upcoming Updates:

**Follow-up Questions:**  Suggest related questions based on conversation flow

**Conversation History:**  Persistent chat history across sessions

### ⭐ If this project helped you, please give it a star!
