import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

# Import renamed modules
from pipeline import RAGPipeline
from web import TechNewsRetriever
from vector import VectorStore

@dataclass
class TechUpdate:
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    summary: str

class TechRAGApp:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.news_retriever = TechNewsRetriever()
        self.vector_store = VectorStore()

    def initialize_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'latest_updates' not in st.session_state:
            st.session_state.latest_updates = []
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'response_style' not in st.session_state:
            st.session_state.response_style = 'Structured'

    def fetch_latest_tech_news(self):
        with st.spinner("Fetching latest tech updates..."):
            sources = ["hackernews", "techcrunch", "github_trending", "reddit_programming"]
            all_updates = []
            for source in sources:
                try:
                    updates = self.news_retriever.fetch_from_source(source)
                    all_updates.extend(updates)
                except Exception as e:
                    st.error(f"Error fetching from {source}: {e}")

            if all_updates:
                self.vector_store.add_documents(all_updates)
                st.session_state.latest_updates = all_updates
                st.session_state.vector_store_ready = True
                st.success(f"Loaded {len(all_updates)} tech updates.")

    def process_query(self, query: str) -> str:
        if not st.session_state.vector_store_ready:
            return "Please fetch latest tech updates first."
        relevant_docs = self.vector_store.similarity_search(query, k=5)

        if st.session_state.response_style == 'Structured':
            return self.rag_pipeline.generate_response(query, relevant_docs)
        else:
            return self.rag_pipeline.generate_conversational_response(query, relevant_docs)

    def display_tech_updates(self):
        with st.sidebar:
            st.header("Latest Tech Updates")
            if st.button("Refresh Updates"):
                self.fetch_latest_tech_news()

            if st.session_state.latest_updates:
                for update in st.session_state.latest_updates[:10]:
                    with st.expander(f"{update.source} - {update.title[:50]}..."):
                        st.write(f"Source: {update.source}")
                        st.write(f"Time: {update.timestamp.strftime('%H:%M')} | Date: {update.timestamp.strftime('%Y-%m-%d')}")
                        st.write(f"Summary: {update.summary}")
                        st.link_button("Read Full", update.url)

            st.radio(
                "Choose Response Style",
                options=["Structured", "Conversational"],
                index=0,
                key="response_style"
            )

    def main_chat_interface(self):
        st.header("Tech Updates Q&A Assistant")
        st.write("Ask about the latest tech news, frameworks, tools, or any tech topic.")

        query = st.text_input(
            "Your question:",
            placeholder="What are the latest AI developments?",
            key="query_input"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ask"):
                if query.strip():
                    self.handle_query(query)
        with col2:
            st.write("Powered by real-time tech data")

    def handle_query(self, query: str):
        st.session_state.chat_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "response": None
        })

        with st.spinner("Analyzing latest tech updates..."):
            response = self.process_query(query)
            st.session_state.chat_history[-1]["response"] = response

        self.display_chat_history()

    def display_chat_history(self):
        st.subheader("Conversation")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"You: {chat['query']}")
            if chat['response']:
                st.markdown(f"Assistant: {chat['response']}")
            st.markdown("---")

    def run(self):
        st.set_page_config(page_title="Tech Updates RAG System", layout="wide")
        st.title("Real-Time Tech Updates RAG System")
        st.write("Get AI-powered answers based on the latest tech news and developments.")

        self.initialize_session_state()

        col1, col2 = st.columns([3, 1])
        with col1:
            self.main_chat_interface()
            if not st.session_state.vector_store_ready:
                self.fetch_latest_tech_news()
            if st.session_state.chat_history:
                self.display_chat_history()

        with col2:
            self.display_tech_updates()

if __name__ == "__main__":
    app = TechRAGApp()
    app.run()
