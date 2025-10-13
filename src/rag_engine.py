import json
import requests
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from data_loader import Email

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: str
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "timestamp": datetime.now().isoformat()
        }

class OllamaHandler:

    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 240):

        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout        
         
        self.temperature = 0.3   
        self.max_tokens = 1000
            
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
             
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Error en Ollama: {response.status_code}")
                return "Lo siento, no pude generar una respuesta."
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return "Error al generar la respuesta."


class EmailRAGEngine:

    def __init__(self,
                 vector_db,
                 ollama_handler: Optional[OllamaHandler] = None,
                 use_reranking: bool = True):

        self.vector_db = vector_db
        self.ollama = ollama_handler or OllamaHandler()
        self.use_reranking = use_reranking
        self.top_k_retrieval: int = 5   
        self.min_similarity_threshold: float = 0.3   
        self.language: str = "es"   
             
    def _create_prompt(self, query: str, context: str) -> str:

        if self.language == "es":
            system_message = """Eres un asistente experto en búsqueda y análisis de correos electrónicos.

Tu tarea es responder preguntas basándote ÚNICAMENTE en los emails proporcionados como contexto, en los emails puedes enontrar información como su id, quien lo manda, quien lo recibe,
la fecha, el asunto y el cuerpo del email.
            
Reglas importantes:
1. Responde SOLO con información que aparece en los emails proporcionados
2. SIEMPRE cita la fuente: indica qué EMAIL(s) contiene(n) la información
3. Si la información no está en los emails, di claramente "No encontré esta información en los emails proporcionados"
4. Sé conciso pero completo
5. Si múltiples emails tienen información relevante, menciónalos todos
6. Usa un tono profesional pero amigable

Formato de respuesta:
- Respuesta directa a la pregunta
- Fuentes: [EMAIL X, EMAIL Y]
- (Opcional) Contexto adicional relevante"""

            user_prompt = f"""Contexto de emails relevantes:
{context}

Pregunta del usuario: {query}

Por favor, responde la pregunta basándote en los emails anteriores. Recuerda citar las fuentes."""

        else:   
            system_message = """You are an expert assistant for email search and analysis.

Your task is to answer questions based ONLY on the emails provided as context.

Important rules:
1. Answer ONLY with information from the provided emails
2. ALWAYS cite sources: indicate which EMAIL(s) contain the information
3. If information is not in the emails, clearly state "I did not find this information in the provided emails"
4. Be concise but complete
5. If multiple emails have relevant information, mention all of them
6. Use a professional but friendly tone

Response format:
- Direct answer to the question
- Sources: [EMAIL X, EMAIL Y]
- (Optional) Additional relevant context"""

            user_prompt = f"""Context from relevant emails:
{context}

User question: {query}

Please answer the question based on the above emails. Remember to cite sources."""
        
         
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    def query(self, 
              question: str,
              n_results: int = 5,
              filters: Optional[Dict] = None) -> RAGResponse:

        start_time = datetime.now()         
        search_results = self.vector_db.search(
            query=question,
            n_results=n_results * 2 if self.use_reranking else n_results,
            filter_metadata=filters
        )
        if not search_results['results']:
            return RAGResponse(
                query=question,
                answer="No encontré emails relevantes para tu pregunta.",
                sources=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_used=self.ollama.model_name
            )
         
         
        relevant_results = search_results['results']
        context = self._build_context(question, relevant_results)
        
         
        answer = self._generate_answer(question, context)
         
        sources = self._prepare_sources(relevant_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            model_used=self.ollama.model_name
        )
    
    def _build_context(self, query: str, retrieved_emails: List[Dict]) -> str:

        if not retrieved_emails:
            return "No se encontraron emails relevantes."
        
        context_parts = []
        
        for idx, email in enumerate(retrieved_emails, 1):
             
            email_info = f"[ID: {email['email_id']}\n"
            email_info += f"De: {email['from']}\n"
            email_info += f"Para: {email['to']}\n"
            email_info += f"Asunto: {email['subject']}\n"
            email_info += f"Fecha: {email['date']}\n"
            email_info += f"Similitud: {email['best_distance']}\n"            
             
            email_info += "\nContenido relevante:\n"
             
            sorted_chunks = sorted(email['chunks'], key=lambda x: x['distance'])
             
            for chunk in sorted_chunks[:2]:   
                chunk_preview = chunk['text'][:300]   
                email_info += f"- {chunk_preview}...\n"
            context_parts.append(email_info)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
            prompt = self._create_prompt(question, context)
            answer = self.ollama.generate(prompt)        
            return answer    
        
    def _prepare_sources(self, results: List[Dict]) -> List[Dict[str, Any]]:
        sources = []
        
        for result in results:   
            source = {
                "email_id": result['email_id'],
                "subject": result['subject'],
                "from": result['from'],
                "to": result['to'],
                "date": result['date'],
                "relevance_score": round(result.get('rerank_score', result['best_distance']), 3),
            }
            sources.append(source)
        
        return sources
    
if __name__ == "__main__":

    db_path = "../data/test_vectordb"
    try:
        from embeddings_system import EmailVectorDB   
        db = EmailVectorDB(db_path)
        print("EmailVectorDB cargado correctamente.")
    except Exception as e:
        print(f"No se pudo inicializar EmailVectorDB en '{db_path}': {e}")
        print("Asegúrate de tener el módulo embeddings y la ruta correcta.")
        db = None


    ollama_instance = OllamaHandler(model_name="llama3.2:3b")
     
    try:
        rag = EmailRAGEngine(vector_db=db, ollama_handler=ollama_instance) if db is not None else None
    except Exception as e:
        print(f"Error creando EmailRAGEngine: {e}")
        rag = None

    if rag is not None:
        print("\nEscribe tu consulta o 'exit' para salir.")
        try:
            while True:
                q = input("\nPregunta > ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "salir"):
                    break

                try:
                    resp = rag.query(q)
                        
                    print(f"\nRespuesta (modelo={resp.model_used}) en {resp.processing_time} segundos:\n{resp.answer}\n")
                    if resp.sources:
                        print("Fuentes Query:")
                        for idx, s in enumerate(resp.sources):
                            print(f" -EMAIL Id: {s.get('email_id')} | Fecha: {s.get('date')}\n      Asunto: {s.get('subject')} ")
                except Exception as e:
                    print(f"Error procesando la consulta: {e}")
        except KeyboardInterrupt:
            print("\nSaliendo...")