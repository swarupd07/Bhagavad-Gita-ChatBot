# Bhagavad-Gita-ChatBot
AI-powered chatbot that provides wisdom and insights from the sacred Bhagavad Gita. Built with Streamlit, LangChain, and Hugging Face models for semantic search and intelligent responses

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
Pandas, NumPy

## Frontend:
Leveraged by GenAI tools 

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



### ⭐ If this project helped you, please give it a star!

### NOTE:
Use "GitaText - Data_for_Gita_ChatBot_Cleaned" file or similarly formated .csv file

!! ChatBot may takes time while initialization !! ...( Will improve)
