import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from data_loader import Email
import argparse
import statistics
import shutil


class MultilingualEmbedder:
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):

        print(f"Modelo: {model_name} (device: {device})")
        self.model = SentenceTransformer(model_name, device=device)
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

        subject = email.subject or ""
        body = email.body or ""

        # Si no hay body pero sí subject: indexamos un único chunk con solo el asunto.
        if not body and subject:
            doc_id = f"{email.id}_chunk_0"
            doc_text = f"Subject: {subject}"
            documents.append(doc_text)

            metadata = base_metadata.copy()
            metadata.update({
                "chunk_type": "subject_body",   # ahora todos los chunks usan este tipo
                "chunk_index": 0,
                "chunk_start": 0,
                "chunk_end": 0,
                "total_chunks": 1
            })
            metadatas.append(metadata)
            ids.append(doc_id)
            return documents, metadatas, ids

        # Si tampoco hay subject ni body: no hacemos nada.
        if not body and not subject:
            return documents, metadatas, ids

        # Si hay body (sea corto o largo), lo dividimos en chunks de palabras
        chunks = self.chunk_text(body, self.chunk_size, self.chunk_overlap)
        total_chunks = len(chunks) if chunks else 1

        for idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            # Cada chunk siempre comienza con el asunto (aunque esté vacío)
            if subject:
                doc_text = f"Subject: {subject}\n\n{chunk_text}"
            else:
                # Si no hay asunto, solo ponemos el trozo de body
                doc_text = chunk_text

            doc_id = f"{email.id}_chunk_{idx}"
            documents.append(doc_text)

            metadata = base_metadata.copy()
            metadata.update({
                "chunk_type": "subject_body",   # único tipo de chunk ahora
                "chunk_index": idx,
                "chunk_start": start_pos,       # posiciones dentro del body original
                "chunk_end": end_pos,
                "total_chunks": total_chunks
            })
            metadatas.append(metadata)
            ids.append(doc_id)

        return documents, metadatas, ids
        
    def _is_duplicate_content(self, email_result: Dict) -> bool:
        body = email_result.get('body', '')
        subject = email_result.get('subject', '')
        
        forward_patterns = [
            r'-{20,}\s*Forwarded by',
            r'From:.*\nSent:.*\nTo:',
            r'-----Original Message-----',
        ]
        
        has_prefix = bool(re.match(r'^\s*(re|fw|fwd):', subject, re.IGNORECASE))
        has_forward_pattern = any(re.search(p, body[:1000], re.IGNORECASE) for p in forward_patterns)
        
        return has_prefix and has_forward_pattern
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        unique_results = []
        seen_subjects = {}
        
        for result in results:
            if self._is_duplicate_content(result):
                continue
            
            subject = result.get('subject', '')
            norm_subject = re.sub(r'^\s*(re|fw|fwd):\s*', '', subject, flags=re.IGNORECASE).strip().lower()
            
            if norm_subject in seen_subjects:
                continue
            
            seen_subjects[norm_subject] = True
            unique_results.append(result)
        
        return unique_results
    
    def search(self, query: str, n_results: int = 10, 
               filter_metadata: Optional[Dict] = None,
               deduplicate: bool = True) -> Dict[str, Any]:

        search_n = n_results * 3 if deduplicate else n_results
        
        query_embedding = self.embedder.encode_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=search_n,
            where=filter_metadata,
            include=["metadatas", "documents", "distances"]
        )
        
        processed_results = self._process_search_results(results, query)
        if deduplicate and processed_results['results']:
            processed_results['results'] = self._deduplicate_results(processed_results['results'])[:n_results] 
            processed_results['total'] = len(processed_results['results'])
        
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
    
    def delete_db(self):
        """Elimina completamente la base de datos"""
        try:
            self.client.delete_collection(self.collection.name)
            print(f"Colección '{self.collection.name}' eliminada")
        except Exception as e:
            print(f"Error eliminando colección: {e}")
        
        try:
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
                print(f"Base de datos eliminada: {self.db_path}")
        except Exception as e:
            print(f"Error eliminando directorio de BD: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        
        if count > 0:
            try:
                sample = self.collection.get(limit=min(count, 2000), include=["metadatas"]) 
                metas = sample.get('metadatas', [])
            except Exception:
                sample = self.collection.get(limit=1000, include=["metadatas"]) 
                metas = sample.get('metadatas', [])
            
            chunk_types = {}
            for m in metas:
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


 

def load_tests_from_tsv(test_path: str) -> List[Tuple[str, str]]:
    tests = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
             
            if '\t' in line:
                expected_id, question = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    expected_id, question = parts
                else:
                    continue
            tests.append((expected_id.strip(), question.strip()))
    return tests


def get_email_ids_in_collection(db: EmailVectorDB) -> set:
     
    try:
        total = db.collection.count()
        sample = db.collection.get(limit=max(1000, total), include=['metadatas'])
    except Exception:
        sample = db.collection.get(limit=1000, include=['metadatas'])
    metas = sample.get('metadatas', [])
    ids = {m.get('email_id') for m in metas if m.get('email_id')}
    return ids


def run_retrieval_tester(db: EmailVectorDB, test_file: str, topk: int = 3, n_results: int = 10) -> Dict[str, Any]:
    tests = load_tests_from_tsv(test_file)
    if not tests:
        raise ValueError(f"No tests found in {test_file}")

    present_ids = get_email_ids_in_collection(db)

    results_per_query = []
    p_at_1 = []
    mrr_scores = []
    recall_at_k = []
    missing_in_db = 0

    for expected_id, question in tqdm(tests, desc="Ejecutando tests"):
        if expected_id not in present_ids:
             
            missing_in_db += 1
            p_at_1.append(0)
            mrr_scores.append(0)
            recall_at_k.append(0)
            results_per_query.append({
                'expected': expected_id,
                'question': question,
                'found': False,
                'rank': None,
                'top_ids': []
            })
            continue

        search_res = db.search(question, n_results=n_results, deduplicate=True)
        predicted_ids = [r['email_id'] for r in search_res.get('results', [])]

        rank = None
        if expected_id in predicted_ids:
            rank = predicted_ids.index(expected_id) + 1
        
         
        p1 = 1 if predicted_ids and predicted_ids[0] == expected_id else 0
        p_at_1.append(p1)

         
        rr = 1.0 / rank if rank else 0.0
        mrr_scores.append(rr)

         
        r_at_k = 1 if expected_id in predicted_ids[:topk] else 0
        recall_at_k.append(r_at_k)

        results_per_query.append({
            'expected': expected_id,
            'question': question,
            'found': rank is not None,
            'rank': rank,
            'top_ids': predicted_ids[:max(topk, 10)]
        })

    metrics = {
        'n_queries': len(tests),
        'n_missing_in_db': missing_in_db,
        'precision_at_1': statistics.mean(p_at_1) if p_at_1 else 0.0,
        'mrr': statistics.mean(mrr_scores) if mrr_scores else 0.0,
        f'recall_at_{topk}': statistics.mean(recall_at_k) if recall_at_k else 0.0,
    }

    summary = {
        'metrics': metrics,
        'per_query': results_per_query
    }

    return summary


def test_multiple_models(json_path: str, test_file: str, topk: int = 3, n_results: int = 10, device: str = "cpu"):
    import torch
    import gc

    models_to_test = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    print(f"\n{'='*80}")
    print(f"Cargando emails desde {json_path}")
    print(f"{'='*80}\n")

    with open(json_path, "r", encoding="utf-8") as f:
        emails_data = json.load(f)
    emails = [Email(**e) for e in emails_data]
    print(f"Total emails cargados: {len(emails)}")
    print(f"Dispositivo: {device.upper()}\n")

    all_results = {}

    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"PROBANDO MODELO: {model_name}")
        print(f"{'='*80}\n")

        db_path = f"../data/test_vectordb_{model_name.replace('/', '_').replace('-', '_')}"

        db = None
        embedder = None
        start_time = time.perf_counter()   

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            embedder = MultilingualEmbedder(model_name=model_name, device=device)
            db = EmailVectorDB(db_path=db_path, embedder=embedder)

            print(f"\nIndexando {len(emails)} emails...")
            db.index_emails(emails, batch_size=32)

            print(f"\nEjecutando tests desde {test_file}...")
            summary = run_retrieval_tester(db, test_file, topk=topk, n_results=n_results)

            elapsed_time = time.perf_counter() - start_time   

            all_results[model_name] = summary['metrics']
            all_results[model_name]['execution_time_sec'] = round(elapsed_time, 2)   

            print(f"\n--- Resultados para {model_name} ---")
            m = summary['metrics']
            print(f"Tiempo total: {elapsed_time:.2f} s")
            print(f"Queries: {m['n_queries']}")
            print(f"Missing expected IDs in DB: {m['n_missing_in_db']}")
            print(f"Precision@1: {m['precision_at_1']:.3f}")
            print(f"MRR: {m['mrr']:.3f}")
            print(f"Recall@{topk}: {m[f'recall_at_{topk}']:.3f}")

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            all_results[model_name] = {
                'error': str(e),
                'execution_time_sec': round(elapsed_time, 2),
                'precision_at_1': 0.0,
                'mrr': 0.0,
                f'recall_at_{topk}': 0.0
            }
            print(f"\n❌ Error con modelo {model_name}: {e}")
            print(f"Tiempo antes del fallo: {elapsed_time:.2f} s")

        finally:
            try:
                print(f"\nLimpiando base de datos temporal...")
                if db is not None:
                    db.delete_db()
            except Exception as e:
                print(f"Error al limpiar BD: {e}")

            del embedder
            del db
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
     
    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO DE TODOS LOS MODELOS")
    print(f"{'='*80}\n")
    
    print(f"{'Modelo':<50} {'P@1':<8} {'MRR':<8} {'R@{topk}':<8}")
    print(f"{'-'*80}")
    
    for model_name, metrics in all_results.items():
        if 'error' not in metrics:
            print(f"{model_name:<50} {metrics['precision_at_1']:<8.3f} {metrics['mrr']:<8.3f} {metrics[f'recall_at_{topk}']:<8.3f}")
        else:
            print(f"{model_name:<50} ERROR")
    
     
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if valid_results:
        best_model_mrr = max(valid_results.items(), 
                             key=lambda x: x[1].get('mrr', 0))
        best_model_p1 = max(valid_results.items(), 
                            key=lambda x: x[1].get('precision_at_1', 0))
        
        print(f"\n{'='*80}")
        print(f"🏆 Mejor modelo por MRR: {best_model_mrr[0]} (MRR: {best_model_mrr[1].get('mrr', 0):.3f})")
        print(f"🏆 Mejor modelo por P@1: {best_model_p1[0]} (P@1: {best_model_p1[1].get('precision_at_1', 0):.3f})")
        print(f"{'='*80}\n")
    else:
        print("\n⚠️  Ningún modelo se ejecutó correctamente\n")
    
     
    output_file = "../data/processed/model_comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Resultados guardados en: {output_file}")
    
    return all_results


 


def test_embeddings_and_db():
    json_path = "../data/processed/enron_sample_1000+60.json"
    with open(json_path, "r", encoding="utf-8") as f:
        emails_data = json.load(f)

    emails = [Email(**e) for e in emails_data]

    db = EmailVectorDB(db_path="../data/test_vectordb")

    print(f"Cargando emails desde {json_path} ({len(emails)} encontrados)")
    db.index_emails(emails)

    results = db.search("bike", n_results=10, deduplicate=False)
    
    print(f"\n Resultados de búsqueda:")
    print(f"Query: '{results['query']}'")
    print(f"Emails encontrados: {results['total']}")
    

    stats = db.get_stats()
    print(f"\n Estadísticas de la DB:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Distribución: {stats['chunk_distribution']}")

    # db.delete_collection()
    
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-tester', action='store_true', help='Ejecutar el tester de recuperación usando test_preguntas.txt')
    parser.add_argument('--test-models', action='store_true', help='Probar múltiples modelos de embeddings')
    parser.add_argument('--test-file', type=str, default='test_preguntas.txt', help='Ruta del archivo de tests (id \t pregunta)')
    parser.add_argument('--json-path', type=str, default='../data/processed/enron_sample_1000+60.json', help='Ruta al archivo JSON con emails')
    parser.add_argument('--topk', type=int, default=3, help='Valor K para recall@K')
    parser.add_argument('--n-results', type=int, default=10, help='Número de resultados a recuperar por consulta')
    parser.add_argument('--db-path', type=str, default='../data/test_vectordb', help='Ruta al vectordb de Chroma')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Dispositivo para los modelos (cpu/cuda)')
    args = parser.parse_args()

    if args.test_models:
        test_multiple_models(
            json_path=args.json_path,
            test_file=args.test_file,
            topk=args.topk,
            n_results=args.n_results,
            device=args.device
        )
    elif args.run_tester:
        db = EmailVectorDB(db_path=args.db_path)
        summary = run_retrieval_tester(db, args.test_file, topk=args.topk, n_results=args.n_results)

        print('\n--- Tester summary ---')
        m = summary['metrics']
        print(f"Queries: {m['n_queries']}")
        print(f"Missing expected IDs in DB: {m['n_missing_in_db']}")
        print(f"Precision@1: {m['precision_at_1']:.3f}")
        print(f"MRR: {m['mrr']:.3f}")
        print(f"Recall@{args.topk}: {m[f'recall_at_{args.topk}']:.3f}")

    else:
         
        test_embeddings_and_db()