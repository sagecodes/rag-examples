# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------
from datasets import load_dataset
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from chromadb.config import Settings


# -----------------------------------------------------------
# Load BioASQ mini  (PubMed abstracts as passages)
# -----------------------------------------------------------
corpus_ds = load_dataset(
    "rag-datasets/rag-mini-bioasq",
    "text-corpus",
    split="passages"
)
qa_ds = load_dataset(
    "rag-datasets/rag-mini-bioasq",
    "question-answer-passages",
    split="test"
)

print(corpus_ds.column_names)
print(corpus_ds[0])


# -----------------------------------------------------------
# Token-aware chunking helper
# -----------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text_tokenwise(text: str, size: int = 256, overlap: int = 32):
    tokens = enc.encode(text)
    step = size - overlap
    slices = [tokens[i:i+size] for i in range(0, len(tokens), step)]
    return [enc.decode(s) for s in slices]

# Chunk every passage
docs, ids = [], []
for row in corpus_ds:
    for i, chunk in enumerate(chunk_text_tokenwise(row["passage"])):
        docs.append(chunk)
        ids.append(f'{row["id"]}_{i}')

print(f"🔹 Prepared {len(docs):,} chunks for embedding")


# -----------------------------------------------------------
# Build local Chroma vector store with manual GPU embeddings
# -----------------------------------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda")
vectors = embed_model.encode(docs, device="cuda", show_progress_bar=True)

client = chromadb.Client(Settings(allow_reset=True))
client.reset()  # Optional: clears all collections

collection = client.get_or_create_collection(name="bioasq_passages")

batch = 1024
for i in range(0, len(docs), batch):
    collection.add(
        documents=docs[i:i+batch],
        embeddings=vectors[i:i+batch],  # ✅ Pass embeddings directly
        ids=ids[i:i+batch]
    )

print("✅ Chroma collection ready!")


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
    max_new_tokens=128
)

def rag_answer(question: str, k: int = 5) -> str:
    context_chunks = retrieve(question, k)
    prompt = "Answer the biomedical question factually using ONLY the context.\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        prompt += f"[Context {i}]\n{chunk}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return generator(prompt, truncation=True)[0]["generated_text"]


# -----------------------------------------------------------
# Demo
# -----------------------------------------------------------
sample_q = qa_ds[0]["question"]
print("📝 Question:", sample_q)
print("💡 RAG answer:", rag_answer(sample_q))

# python chroma_transformers/rag.py