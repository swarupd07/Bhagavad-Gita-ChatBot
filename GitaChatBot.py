import streamlit as st
from typing import List, Dict
import pandas as pd
import re
import os
from collections import Counter
import nltk
from dotenv import load_dotenv
import tempfile
import zipfile
from pathlib import Path

# Downloading NLTK data
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Loading environment variables
load_dotenv()


from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma

# TfidfVectorizer for keyword extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

THEMES = {
    'dharma': ['duty', 'righteousness', 'moral', 'ethics', 'right action'],
    'karma': ['action', 'work', 'deed', 'consequence', 'result'],
    'moksha': ['liberation', 'freedom', 'salvation', 'enlightenment'],
    'devotion': ['bhakti', 'love', 'surrender', 'devotion', 'worship'],
    'knowledge': ['jnana', 'wisdom', 'understanding', 'truth', 'reality'],
    'meditation': ['dhyana', 'concentration', 'focus', 'mindfulness'],
    'detachment': ['vairagya', 'renunciation', 'letting go', 'non-attachment'],
    'self': ['atma', 'soul', 'self-realization', 'consciousness', 'identity', 'dhyan', 'mind', 'discipline', 'control', 'meditation', 'focus', 'concentration'],
    'yoga': ['union', 'discipline', 'practice', 'path', 'spirituality', 'asana', 'pranayama'],
    'war': ['battle', 'conflict', 'struggle', 'fight', 'courage', 'valor', 'heroism'],
}

class GitaVerse:
    def __init__(self):
        self.chapter = ""
        self.verse_no = ""
        self.sanskrit_transliteration = ""
        self.translation_in_english = ""
        self.meaning_in_english = ""
        self.meaning_in_hindi = ""
        self.keywords = []
        self.theme = ""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class GitaChatBot:
    def __init__(self):
        self.verses = []
        self.collections = None
        self.model = None
        self.embedder = None
        self.tfidf_vectorizer = None
        self.english_stopwords = stopwords.words('english')
        self.tfidf_matrix = None
        
    def initialize_models(self):

        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
        llm = HuggingFaceEndpoint(
            endpoint_url="openai/gpt-oss-20b",
            task="text-generation",
            huggingfacehub_api_token=api_token,
            max_new_tokens=200,
        )
        self.model = ChatHuggingFace(llm=llm)
        
        return True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def extract_verse(self, data: pd.DataFrame):
        self.verses = []
        
        for index, row in data.iterrows():
            verse = GitaVerse()
            verse.chapter = str(row['chapter'])
            verse.verse_no = str(row['verse'])
            verse.sanskrit_transliteration = row['sanskrit_verse_transliteration']
            verse.translation_in_english = row['translation_in_english']
            verse.meaning_in_english = row['meaning_in_english']
            verse.meaning_in_hindi = row['meaning_in_hindi']
            verse.keywords = self.get_keywords(row)
            verse.theme = self.get_theme(row)
            self.verses.append(verse)


    def get_keywords(self, row):
        keywords = []
        words = re.findall(r'\b[A-Za-z]{4,}\b', row['meaning_in_english'])
        for word in words:
            if word.lower() not in self.english_stopwords:
                keywords.append(word)
        
        keywords = [word for word, freq in Counter(keywords).most_common(10)]
        return keywords
    
    def get_theme(self, row):
        text = (row['meaning_in_english']).lower()
        theme_score = {}
        for theme, keywords in THEMES.items():
            score = sum(text.count(keyword) for keyword in keywords)
            theme_score[theme] = score
            
        return max(theme_score, key=theme_score.get) if max(theme_score.values()) > 0 else 'general'
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create_vector_store(self):
        # temporary directory for ChromaDB
        temp_dir = tempfile.mkdtemp()
        
        self.collections = Chroma(
            persist_directory=temp_dir,
            embedding_function=self.embedder,
            collection_name="gita_verse"
        )
        
        # dividing verses in batches for getting better speed
        batch_size = 36
        for i in range(0, len(self.verses), batch_size):
            batch = self.verses[i:min(i+batch_size, len(self.verses))]
            
            texts = []
            metadatas = []
            ids = []
            
            for verse in batch:
                text = f"Chapter {verse.chapter}, Verse {verse.verse_no}:\n{verse.sanskrit_transliteration}\n{verse.translation_in_english}\n{verse.meaning_in_english}"
                texts.append(text)
                
                metadata = {
                    'chapter': verse.chapter,
                    'verse': verse.verse_no,
                    'keywords': ', '.join(verse.keywords),
                    'theme': verse.theme
                }
                metadatas.append(metadata)
                ids.append(f"chapter_{verse.chapter}_verse_{verse.verse_no}")
            
            self.collections.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        # Fit TF-IDF vectorizer on all verse keywords
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([' '.join(verse.keywords) for verse in self.verses])
        return True
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def retrieve_content(self, question: str, top_k: int = 3) -> List[Dict]:
        semantic_results = self.semantic_retrieval(question, top_k)
        theme_results = self.theme_based_retrieval(question, top_k)
        keyword_results = self.keyword_based_retrieval(question, top_k)
        return self.combine_results(semantic_results, theme_results, keyword_results)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def semantic_retrieval(self, question: str, top_k: int) -> List[Dict]:
        verses = self.collections.similarity_search(question, k=top_k)
        results = []
        
        for verse in verses:
            results.append({
                'chapter': verse.metadata.get('chapter'),
                'verse': verse.metadata.get('verse'),
                'content': verse.page_content
            })
        return results
    
    def keyword_based_retrieval(self, question: str, top_k: int) -> List[Dict]:
        question_lower = question.lower()
        question_words = re.findall(r'\b[A-Za-z]{4,}\b', question_lower)
        filtered_words = [word for word in question_words if word not in self.english_stopwords]
        if not filtered_words:
            return []
        
        question_tfidf = self.tfidf_vectorizer.transform([' '.join(filtered_words)])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        dense = question_tfidf.todense()
        denselist = dense.tolist()
        tfidf_scores = dict(zip(feature_names, denselist[0]))
        
        verse_scores = []
        for idx, verse in enumerate(self.verses):
            score = sum(tfidf_scores.get(keyword.lower(), 0) for keyword in verse.keywords)
            if score > 0:
                verse_scores.append((verse, score))

        verse_scores.sort(key=lambda x: x[1], reverse=True)
        top_verses = verse_scores[:top_k]
        results = []
        for verse, score in top_verses:
            results.append({
                'chapter': verse.chapter,
                'verse': verse.verse_no,
                'content': f"{verse.sanskrit_transliteration} {verse.translation_in_english} {verse.meaning_in_english}"
            })
        
        return results




    def theme_based_retrieval(self, question: str, top_k: int) -> List[Dict]:
        question_lower = question.lower()
        theme_score = {}
        
        for theme, keywords in THEMES.items():
            score = sum(question_lower.count(keyword) for keyword in keywords)
            theme_score[theme] = score
        
        relevant_verses = []
        sorted_themes = sorted(theme_score.items(), key=lambda item: item[1], reverse=True)
        
        for theme, score in sorted_themes:
            if score > 0:
                for verse in self.verses:
                    if verse.theme == theme:
                        relevant_verses.append(verse)
        
        def verse_score(verse):
            score = 0
            if verse.theme in THEMES:
                for keyword in THEMES[verse.theme]:
                    score += verse.meaning_in_english.lower().count(keyword)
            return score
        
        relevant_verses.sort(key=verse_score, reverse=True)
        
        results = []
        for verse in relevant_verses[:top_k]:
            result = {
                'chapter': verse.chapter,
                'verse': verse.verse_no,
                'content': f"{verse.sanskrit_transliteration} {verse.translation_in_english} {verse.meaning_in_english}"
            }
            results.append(result)
        
        return results
    
    def combine_results(self, semantic_results: List[Dict], theme_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        importance_score = {}
    
        for result in semantic_results:
            key = (result['chapter'], result['verse'])
            importance_score[key] = importance_score.get(key, 0) + 4
        
        for result in theme_results:
            key = (result['chapter'], result['verse'])
            importance_score[key] = importance_score.get(key, 0) + 1
        
        for result in keyword_results:
            key = (result['chapter'], result['verse'])
            importance_score[key] = importance_score.get(key, 0) + 2
            
        sorted_results = sorted(importance_score.items(), key=lambda x: x[1], reverse=True)
        combined_results = []
        seen_verses = set()
        
        for (chapter, verse), score in sorted_results:
            if (chapter, verse) not in seen_verses:
                for result_list in [semantic_results, theme_results]:
                    for result in result_list:
                        if result['chapter'] == chapter and result['verse'] == verse:
                            combined_results.append(result)
                            seen_verses.add((chapter, verse))
                            break
                    if (chapter, verse) in seen_verses:
                        break
        
        return combined_results[:3]  # Returning top 3 verses
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def generate_prompt(self, question: str, retrieved_content: List[Dict], style: str) -> str:

        context = '\n\n'.join(f"{verse['content']}" for verse in retrieved_content)
        
        if style == "student":
            return f"""
You are a guided teacher of the Bhagavad Gita. Use the following verses as context to answer the student's question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Give output after summarizing it
- Answer clearly and compassionately, in simple and scholarly tone.
- First, explain the meaning of each relevant verse briefly.
- Then, synthesize them into a clear response that addresses the student's question.
- Provide bullet-point or short-paragraph explanations.
- If the answer isn't fully supported by the context, say "I don't have enough information to answer that"—rather than guessing.

Answer:
"""
        
        elif style == "philosophical":
            return f"""
You are a wise sage well-versed in the Bhagavad Gita, speaking with the depth of ancient wisdom.

Student's Question: {question}

Sacred Teachings from Bhagavad Gita: {context}

Provide a comprehensive summarized response that:
- Give output after summarizing it
1. Addresses the philosophical essence of the question
2. Explains the deeper meaning from the Gita's perspective
3. Connects to universal spiritual principles
4. Offers contemplative insights for self-reflection

Speak with wisdom, clarity, and reverence for the sacred teachings.
"""
        
        elif style == "devotional":
            return f"""
You are a devoted teacher sharing Krishna's teachings with love and reverence.

Devotee's Question: {question}

Lord Krishna's Teachings: {context}

Share a heartfelt summarized response that:
- Give output after summarizing it
1. Honors the divine wisdom in Krishna's words
2. Speaks to the heart and soul
3. Encourages devotion and surrender
4. Inspires spiritual growth through love

Respond with devotion, compassion, and spiritual warmth.
"""
        
        else:  # balanced
            return f"""
You are a knowledgeable teacher of the Bhagavad Gita, offering balanced wisdom.

Question: {question}

Gita's Wisdom: {context}

Provide a well-rounded summarized response that:
- Give output after summarizing it
1. Explains the teaching clearly and accurately
2. Offers both philosophical depth and practical relevance
3. Shows the relevance to modern life
4. Maintains respect for the sacred text

Balance wisdom with accessibility, depth with clarity.
"""
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_answer(self, question: str, retrieved_content: List[Dict], style: str = "balanced") -> str:
        """Generate answer using the AI model"""
        if not retrieved_content:
            return "I couldn't find relevant verses to answer your question. Please try rephrasing or ask about a different topic."
        
        prompt = self.generate_prompt(question, retrieved_content, style)
        answer = self.model.invoke(prompt)
        return answer.content

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initializing the chatbot
@st.cache_resource
def get_chatbot():
    return GitaChatBot()

def main():
    st.set_page_config(
        page_title="Gita ChatBot",
        page_icon="🕉️",
        layout="wide"
    )
    
    st.title("🕉️ Bhagavad Gita ChatBot")
    st.markdown("*Ask questions about the sacred teachings of the Bhagavad Gita*")
    
    chatbot = get_chatbot()
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Gita CSV file",
            type=['csv'],
            help="Upload your Bhagavad Gita dataset CSV file"
        )
        
        # API Token input
        api_token = st.text_input(
            "HuggingFace API Token",
            type="password",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            help="Enter your HuggingFace API token"
        )
        
        if api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        
        # Initialize button
        if st.button("Initialize ChatBot", type="primary"):
            if uploaded_file and api_token:
                with st.spinner("Initializing ChatBot..."):
                    # Load data
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} verses from CSV")
                    
                    # Initialize models
                    if chatbot.initialize_models():
                        st.success("AI models initialized successfully")
                        
                        # Process data
                        chatbot.extract_verse(df)
                        st.success(f"Processed {len(chatbot.verses)} verses")
                        
                        # Create vector store
                        if chatbot.create_vector_store():
                            st.success("Vector store created successfully")
                            st.session_state.chatbot_ready = True
                        else:
                            st.error("Failed to create vector store")
                    else:
                        st.error("Failed to initialize AI models")
            else:
                st.warning("Please upload CSV file and provide API token")
    
    # Main chat interface
    if st.session_state.get('chatbot_ready', False):
        st.success("✅ ChatBot is ready!")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question about the Bhagavad Gita:",
                placeholder="What does the Gita say about dharma?"
            )
        
        with col2:
            style = st.selectbox(
                "Answer Style:",
                ["balanced", "student", "philosophical", "devotional"]
            )
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Searching sacred texts..."):
                # Retrieve content
                retrieved_content = chatbot.retrieve_content(question, top_k=3)
                
                if retrieved_content:
                    # Display retrieved verses
                    with st.expander("📖 Retrieved Verses", expanded=False):
                        for i, verse in enumerate(retrieved_content, 1):
                            st.markdown(f"**Verse {i}: Chapter {verse['chapter']}, Verse {verse['verse']}**")
                            st.text(verse['content'][:200] + "..." if len(verse['content']) > 200 else verse['content'])
                            st.markdown("---")
                    
                    # Generate and display answer
                    answer = chatbot.generate_answer(question, retrieved_content, style)
                    
                    st.markdown("### 💫 Answer:")
                    st.markdown(answer)
                else:
                    st.warning("No relevant verses found. Please try rephrasing your question.")
        
        # Sample questions
        '''with st.expander("💡 Sample Questions", expanded=False):
            sample_questions = [
                "What is dharma according to the Bhagavad Gita?",
                "How should one perform karma yoga?",
                "What does Krishna say about detachment?",
                "What is the path to moksha?",
                "How to control the mind through meditation?",
                "What is bhakti yoga?",
                "What does the Gita teach about war and duty?",
                "How to achieve self-realization?"
            ]
            
            for q in sample_questions:
                if st.button(q, key=f"sample_{q}"):
                    st.session_state.sample_question = q
                    st.rerun()
            
            if 'sample_question' in st.session_state:
                question = st.session_state.sample_question
                del st.session_state.sample_question'''
    
    else:
        st.info("👈 Please configure and initialize the ChatBot using the sidebar")
        
        # Show sample CSV format
        with st.expander("📋 Required CSV Format", expanded=True):
            st.markdown("""
            Your CSV file should contain the following columns:
            - `chapter`: Chapter number
            - `verse`: Verse number  
            - `sanskrit_verse_transliteration`: Sanskrit verse in Roman script
            - `translation_in_english`: English translation
            - `meaning_in_english`: Detailed meaning in English
            - `meaning_in_hindi`: Meaning in Hindi (optional)
            """)
            
            # Sample data
            sample_data = pd.DataFrame({
                'chapter': [1, 1],
                'verse': [1, 2],
                'sanskrit_verse_transliteration': ['dhrtarastra uvaca...', 'sanjaya uvaca...'],
                'translation_in_english': ['Dhritarashtra said...', 'Sanjaya said...'],
                'meaning_in_english': ['King Dhritarashtra inquired...', 'Sanjaya replied...'],
                'meaning_in_hindi': ['राजा धृतराष्ट्र ने पूछा...', 'संजय ने उत्तर दिया...']
            })
            
            st.dataframe(sample_data)

if __name__ == "__main__":
    main()