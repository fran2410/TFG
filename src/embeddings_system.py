import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from data_loader import Email


class MultilingualEmbedder:
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        print(f"Modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])
        
        texts = [t if t else " " for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)


class EmailVectorDB:
    def __init__(self, 
                 db_path: str = "data/vectordb",
                 collection_name: str = "emails",
                 embedder: Optional[MultilingualEmbedder] = None):

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedder = embedder or MultilingualEmbedder()
        
        print(f"Inicializando ChromaDB en {self.db_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Colección '{collection_name}' cargada. Documentos: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Email search collection"}
            )
            print(f"Nueva colección '{collection_name}' creada")
            
        self.chunk_size = 300 
        self.chunk_overlap = 50  
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[Tuple[str, int, int]]:
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [(text, 0, len(text))]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            start_pos = len(' '.join(words[:i])) if i > 0 else 0
            end_pos = start_pos + len(chunk_text)
            
            chunks.append((chunk_text, start_pos, end_pos))
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def index_emails(self, emails: List[Any], batch_size: int = 100):

        documents = []
        metadatas = []
        ids = []
        
        for email in tqdm(emails, desc="Preparando documentos"):

            email_docs, email_metas, email_ids = self._prepare_email_documents(email)
            documents.extend(email_docs)
            metadatas.extend(email_metas)
            ids.extend(email_ids)
        
        if not documents:
            print("ERROR No hay documentos para indexar")
            return
        
        print(f"Total de chunks a indexar: {len(documents)}")
        
        print("Generando embeddings...")
        embeddings = self.embedder.encode(documents, batch_size=batch_size)
        
        print("Insertando en ChromaDB...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Insertando batches"):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"Total documentos en DB: {self.collection.count()}")
    
    def _prepare_email_documents(self, email: Any) -> Tuple[List[str], List[Dict], List[str]]:
        documents = []
        metadatas = []
        ids = []
        
        base_metadata = {
            "email_id": email.id,
            "message_id": email.message_id,
            "from": email.from_address,
            "to": ", ".join(email.to_addresses) if email.to_addresses else "",
            "cc": ", ".join(email.cc_addresses) if email.cc_addresses else "",
            "bcc": ", ".join(email.bcc_addresses) if email.bcc_addresses else "",
            "date": email.date or "",
            "subject": email.subject or "",
            "body": email.body or "",
            "x_from": email.x_from or "",
            "x_to": ", ".join(email.x_to) if email.x_to else "",
            "x_cc": ", ".join(email.x_cc) if email.x_cc else "",
            "x_bcc": ", ".join(email.x_bcc) if email.x_bcc else "",
            "x_folder": email.x_folder or "",
            "x_origin": email.x_origin or "",
            "x_filename": email.x_filename or "",
        }
        
        if email.subject:
            doc_id = f"{email.id}_subject"
            documents.append(f"Email Subject: {email.subject}")
            
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_type": "subject",
                "chunk_index": 0
            })
            metadatas.append(metadata)
            ids.append(doc_id)
        
        if email.body:
            preview = email.body[:500]  
            summary = f"Subject: {email.subject}\n\n{preview}..." if email.subject else preview
            
            doc_id = f"{email.id}_summary"
            documents.append(summary)
            
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_type": "summary",
                "chunk_index": 0
            })
            metadatas.append(metadata)
            ids.append(doc_id)
        
        if email.body: 
            chunks = self.chunk_text(email.body, self.chunk_size, self.chunk_overlap)
            
            for idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
                doc_id = f"{email.id}_body_{idx}"
                documents.append(chunk_text)
                
                metadata = base_metadata.copy()
                metadata.update({
                    "chunk_type": "body",
                    "chunk_index": idx,
                    "chunk_start": start_pos,
                    "chunk_end": end_pos,
                    "total_chunks": len(chunks)
                })
                metadatas.append(metadata)
                ids.append(doc_id)
        
        return documents, metadatas, ids
    
    def search(self, query: str, n_results: int = 10, filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        
        query_embedding = self.embedder.encode_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_metadata,
            include=["metadatas", "documents", "distances"]
        )
        
        processed_results = self._process_search_results(results, query)
        
        return processed_results
    
    def _process_search_results(self, results: Dict, query: str) -> Dict[str, Any]:

        if not results['ids'][0]:
            return {"query": query, "results": [], "total": 0}
        
        email_results = {}
        
        for idx, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][idx]
            document = results['documents'][0][idx]
            distance = results['distances'][0][idx]
            
            email_id = metadata['email_id']
            
            if email_id not in email_results:
                email_results[email_id] = {
                    "email_id": email_id,
                    "from": metadata.get('from', ''),
                    "to": metadata.get('to', ''),
                    "subject": metadata.get('subject', ''),
                    "date": metadata.get('date', ''),
                    "chunks": [],
                    "best_distance": distance
                }
            
            email_results[email_id]['chunks'].append({
                "text": document,
                "chunk_type": metadata.get('chunk_type', 'unknown'),
                "chunk_index": metadata.get('chunk_index', 0),
                "distance": distance
            })
            
            if distance < email_results[email_id]['best_distance']:
                email_results[email_id]['best_distance'] = distance
        
        sorted_results = sorted(
            email_results.values(), 
            key=lambda x: x['best_distance']
        )
        
        return {
            "query": query,
            "results": sorted_results,
            "total": len(sorted_results)
        }
    
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            print(f"Colección '{self.collection.name}' eliminada")
        except Exception as e:
            print(f"Error eliminando colección: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        
        if count > 0:
            sample = self.collection.get(limit=100, include=["metadatas"])
            
            chunk_types = {}
            for m in sample['metadatas']:
                ct = m.get('chunk_type', 'unknown')
                chunk_types[ct] = chunk_types.get(ct, 0) + 1
        else:
            chunk_types = {}
        
        return {
            "total_chunks": count,
            "chunk_distribution": chunk_types,
            "db_path": str(self.db_path),
            "embedding_dim": self.embedder.embedding_dim
        }


def test_embeddings_and_db():
    
    json_path = "../data/processed/enron_sample_1000.json"
    with open(json_path, "r", encoding="utf-8") as f:
        emails_data = json.load(f)

    emails = [Email(**e) for e in emails_data]

    db = EmailVectorDB(db_path="../data/test_vectordb")

    print(f"Cargando emails desde {json_path} ({len(emails)} encontrados)")
    db.index_emails(emails)

    results = db.search("bike", n_results=5)
    
    print(f"\n Resultados de búsqueda:")
    print(f"Query: '{results['query']}'")
    print(f"Emails encontrados: {results['total']}")
    
    for i, result in enumerate(results['results'][:5]):
        print(f"\n Email {i+1}:")
        print(f"  - Email ID: {result['email_id']}")
        print(f"  - Subject: {result['subject']}")
        print(f"  - From: {result['from']}")
        print(f"  - Best distance: {result['best_distance']:.3f}")
        print(f"  - Chunks encontrados: {len(result['chunks'])}")

    stats = db.get_stats()
    print(f"\n Estadísticas de la DB:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Distribución: {stats['chunk_distribution']}")

    db.delete_collection()
    
    return db


if __name__ == "__main__":
    test_embeddings_and_db()