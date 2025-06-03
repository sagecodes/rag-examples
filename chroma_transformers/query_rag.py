from datasets import load_dataset
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="bioasq_passages")

# -----------------------------------------------------------
# Simple retrieval wrapper
# -----------------------------------------------------------
def retrieve(question: str, k: int = 5):
    res = collection.query(query_texts=[question], n_results=k)
    return res["documents"][0]

# -----------------------------------------------------------
# Generation with FLAN-T5
# -----------------------------------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=500
)

def rag_answer(question: str, k: int = 5) -> str:
    context_chunks = retrieve(question, k)
    prompt = "Answer the biomedical question factually using ONLY the context.\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        prompt += f"[Context {i}]\n{chunk}\n\n"
    prompt += f"Question: {question}"
    answer = generator(prompt, truncation=True)[0]["generated_text"]
    return f" prompt: {prompt.strip()} \n\n Answer: {answer.strip()}"


# -----------------------------------------------------------
# Demo
# -----------------------------------------------------------
# sample_q = qa_ds[0]["question"]
# print("📝 Question:", "What is Hirschsprung disease?")
print("💡 RAG answer:", rag_answer("Is cystic fibrosis caused by a single gene?"))

# python chroma_transformers/query_rag.py