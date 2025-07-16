import os
from openai import OpenAI  # ✅ For OpenAI v1+
from dotenv import load_dotenv
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

load_dotenv()

@dataclass
class RAGContext:
    query: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    sources: List[str]
    timestamp: datetime

class RAGPipeline:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = OpenAI()  # ✅ Automatically uses OPENAI_API_KEY from .env

        self.system_prompt = """You are an expert AI assistant specialized in providing accurate, up-to-date information about technology developments, programming, software engineering, and tech industry news.

Your role is to:
1. Analyze the provided context from recent tech sources
2. Give comprehensive, accurate answers based on the retrieved information
3. Cite specific sources when providing information
4. Highlight the most recent and relevant developments
5. Explain technical concepts clearly
6. Provide actionable insights when appropriate

Guidelines:
- Always ground your responses in the provided context
- When citing sources, mention the source name and recency
- If information is incomplete, acknowledge limitations
- For technical topics, provide both overview and details
- Stay focused on technology, programming, and industry developments
- Be concise but comprehensive"""

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "I don't have any recent tech updates to answer your question. Please refresh the tech updates first."

        context = self._prepare_context(retrieved_docs)
        user_prompt = self._create_user_prompt(query, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            generated_response = response.choices[0].message.content
            return self._format_response_with_sources(generated_response, retrieved_docs)

        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your OpenAI API key and try again."

    def generate_conversational_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "I couldn't find any recent updates to answer your question. Try refreshing the tech news."

        context = self._prepare_context(retrieved_docs)
        prompt = f"""Here are some recent updates:\n{context}\n\nUser question: {query}\n\nPlease answer the question in a natural and conversational tone, keeping it simple and easy to understand."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You're a helpful tech assistant who answers in a friendly tone."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_context = f"""Document {i}:
Title: {doc.get('title', 'No title')}
Source: {doc.get('source', 'Unknown')}
Content: {doc.get('content', '')[:800]}...
Relevance Score: {doc.get('similarity_score', 0):.3f}
---"""
            context_parts.append(doc_context)
        return "\n".join(context_parts)

    def _create_user_prompt(self, query: str, context: str) -> str:
        return f"""
Based on the following recent tech updates and information, please answer the user's question:

USER QUESTION: {query}

RECENT TECH CONTEXT:
{context}

Please provide a comprehensive answer based on the provided context. Include:
1. Direct answer to the question
2. Relevant details from the sources
3. Recent developments mentioned in the context
4. Technical insights and implications
5. Actionable recommendations if applicable

Focus on information from the provided context and cite sources appropriately.
"""

    def _format_response_with_sources(self, response: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        sources_section = "\n\n📚 **Sources:**\n"
        unique_sources = {}

        for doc in retrieved_docs:
            source = doc.get('source', 'Unknown')
            title = doc.get('title', 'No title')
            url = doc.get('url', '')
            if source not in unique_sources:
                unique_sources[source] = []
            unique_sources[source].append({'title': title, 'url': url})

        for source, articles in unique_sources.items():
            sources_section += f"- **{source.title()}**: "
            links = []
            for article in articles[:3]:
                if article['url']:
                    links.append(f"[{article['title'][:50]}...]({article['url']})")
                else:
                    links.append(article['title'][:50] + "...")
            sources_section += ", ".join(links) + "\n"

        return response + sources_section
