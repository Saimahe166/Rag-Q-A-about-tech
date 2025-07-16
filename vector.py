import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime
from typing import List, Dict , Any , Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


class VectorStore:
    def __init__(self, collection_name:str = "tech_updates", persist_directory:str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path = persist_directory,
            settings = Settings(
                anonymized_telemetry = False,
                allow_reset = True
            )
        )
        self.embedding_model = SentenceTransformer('all-MiniLm-L6-v2')
        self.collection = self._get_or_create_collection()
        print(f"Vector store initialized with {self.collection.count()} documents")


    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name = self.collection_name)
            print(f"using existing collection name: {self.collection_name}")

        except:
            collection = self.client.create_collection(
                name = self.collection_name,
                metadata = {"hnsw:space": "cosine"}
            )
            print(f" created new collection: {self.collection_name}")

        return collection
    
    def add_documents(self, tech_updates:List[any])-> None:
        if not tech_updates:
            return
        documents = []
        metadatas=[]
        ids=[]

        for update in tech_updates:
            doc_text =f"{update.title}\n\n{update.content}\n\nSource:{update.source}"
            doc_id = self.create_doc_id(update.url, update.timestamp)
            if self._document_exists(doc_id):
                continue
            documents.append(doc_text)
            metadatas.append({
                "title":update.title,
                "url":update.url,
                "source":update.source,
                "timestamp":update.timestamp.isoformat(),
                "summary": update.summary
            })
            ids.append(doc_id)
        if documents:
            try:
                self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
                print(f"{len(documents)}Added documents to the vector store")
            except Exception as e:
                print("documents are not added to vector store")

    def _create_doc_id(self, url: str, timestamp: datetime) -> str:
        """Create unique document ID"""
        unique_string = f"{url}_{timestamp.isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()


    def _document_exists(self, doc_id: str) -> bool:
        """Check if document already exists"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'title': metadata.get('title', 'No title'),
                        'source': metadata.get('source', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'summary': metadata.get('summary', '')
                    })
            
            return formatted_results
        
        except Exception as e:
            print(f" Error in similarity search: {e}")
            return []
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent documents"""
        try:
            # Get all documents
            all_docs = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_docs['documents']:
                return []
            
            # Sort by timestamp (newest first)
            docs_with_metadata = []
            for i, doc in enumerate(all_docs['documents']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                timestamp_str = metadata.get('timestamp', '')
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.min
                except:
                    timestamp = datetime.min
                
                docs_with_metadata.append({
                    'content': doc,
                    'metadata': metadata,
                    'timestamp': timestamp,
                    'title': metadata.get('title', 'No title'),
                    'source': metadata.get('source', 'Unknown'),
                    'url': metadata.get('url', ''),
                    'summary': metadata.get('summary', '')
                })
            
            # Sort by timestamp and return top results
            docs_with_metadata.sort(key=lambda x: x['timestamp'], reverse=True)
            return docs_with_metadata[:limit]
        
        except Exception as e:
            print(f" Error getting recent documents: {e}")
            return []
    
    def get_documents_by_source(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents from specific source"""
        try:
            results = self.collection.get(
                where={"source": source},
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'title': metadata.get('title', 'No title'),
                        'source': metadata.get('source', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'summary': metadata.get('summary', '')
                    })
            
            return formatted_results
        
        except Exception as e:
            print(f" Error getting documents by source: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            total_docs = self.collection.count()
            
            # Get source distribution
            all_docs = self.collection.get(include=['metadatas'])
            source_counts = {}
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                'total_documents': total_docs,
                'source_distribution': source_counts,
                'collection_name': self.collection_name
            }
        
        except Exception as e:
            print(f" Error getting stats: {e}")
            return {'total_documents': 0, 'source_distribution': {}, 'collection_name': self.collection_name}

    def clear_collection(self) -> None:
        """Clear all documents from collection"""
        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f" Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f" Error clearing collection: {e}")

    def delete_old_documents(self, days_old: int = 7) -> None:
        """Delete documents older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get all documents
            all_docs = self.collection.get(include=['metadatas'])
            
            old_doc_ids = []
            if all_docs['metadatas']:
                for i, metadata in enumerate(all_docs['metadatas']):
                    timestamp_str = metadata.get('timestamp', '')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_date:
                            old_doc_ids.append(all_docs['ids'][i])
                    except:
                        continue
            
            # Delete old documents
            if old_doc_ids:
                self.collection.delete(ids=old_doc_ids)
                print(f" Deleted {len(old_doc_ids)} old documents")
        
        except Exception as e:
            print(f" Error deleting old documents: {e}")

# Example usage
if __name__ == "__main__":
    # Test vector store
    vector_store = VectorStore()
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
    
    # Test similarity search
    results = vector_store.similarity_search("artificial intelligence", k=3)
    print(f"\nFound {len(results)} similar documents for 'artificial intelligence'")
    
    for result in results:
        print(f"- {result['title']} (Score: {result['similarity_score']:.3f})")
        print(f"  Source: {result['source']}")
        print(f"  Summary: {result['summary'][:100]}...")
        print()







    

