# build_rag.py

import os
import uuid
import weaviate
import tiktoken
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.config import Property, DataType, Configure
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Connect to persistent embedded Weaviate
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        persistence_data_path="./weaviate_bioasq"
    )
)
client.connect()

collection_name = "BioChunks"

# Create schema if it doesn't exist
if not client.collections.exists(collection_name):
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),  # manual vector insert
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
        ]
    )

collection = client.collections.get(collection_name)

# Tokenizer and embedding model
enc = tiktoken.get_encoding("cl100k_base")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text_tokenwise(text: str, size: int = 256, overlap: int = 32):
    tokens = enc.encode(text)
    step = size - overlap
    slices = [tokens[i:i+size] for i in range(0, len(tokens), step)]
    return [enc.decode(s) for s in slices]

# Only embed if empty
if collection.aggregate.over_all(total_count=True).total_count == 0:
    print("🚀 Building vector store from BioASQ...")
    corpus_ds = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    for row in corpus_ds:
        chunks = chunk_text_tokenwise(row["passage"])
        for chunk in chunks:
            embedding = embed_model.encode(chunk).tolist()
            collection.data.insert(
                properties={"text": chunk, "doc_id": str(row["id"])},
                vector=embedding,
                uuid=uuid.uuid4()
            )
    print("✅ Vector store built and saved.")
else:
    print("✅ Vector store already exists. Skipping rebuild.")

client.close()

# python weaviate_embed/build_rag.py




