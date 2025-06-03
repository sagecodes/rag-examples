import weaviate
import os
from weaviate.embedded import EmbeddedOptions
from sentence_transformers import SentenceTransformer

# Load the same model used in build_rag.py
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to the persistent embedded Weaviate instance
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        persistence_data_path="./weaviate_bioasq"
    )
)
client.connect()

collection = client.collections.get("BioChunks")

def retrieve(question: str, k: int = 5):
    vector = embed_model.encode(question).tolist()
    results = collection.query.near_vector(target_vector=vector, limit=k).fetch_objects()
    return [obj.properties["text"] for obj in results.objects]

def rag_answer(question: str, k: int = 5):
    chunks = retrieve(question, k)
    context = "\n".join(chunks)
    return f"Question: {question}\n---\nContext:\n{context}"

if __name__ == "__main__":
    question = "What are the causes of acute myeloid leukemia?"
    answer = rag_answer(question)
    print(answer)

    client.close()
