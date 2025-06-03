# app.py

import os
import weaviate
import gradio as gr
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to embedded Weaviate instance
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(persistence_data_path="./weaviate_bioasq")
)
client.connect()

collection = client.collections.get("BioChunks")


def retrieve(question, k=3):
    vector = embed_model.encode(question).tolist()
    results = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        return_metadata=MetadataQuery(distance=True),
        return_properties=["text"],
    )
    return [obj.properties["text"] for obj in results.objects]


def rag_answer(question, k):
    chunks = retrieve(question, k)
    context = "\n\n".join(chunks)
    return f"📄 Top {k} retrieved chunks:\n\n{context}"


# Gradio UI
demo = gr.Interface(
    fn=rag_answer,
    inputs=[
        gr.Textbox(label="Ask a biomedical question"),
        gr.Slider(minimum=1, maximum=10, value=3, label="Top K Documents"),
    ],
    outputs=gr.Textbox(label="Retrieved Passages"),
    title="🧬 BioASQ RAG Demo",
    description="Ask a biomedical question and retrieve relevant passages from BioASQ using Weaviate + MiniLM.",
)

if __name__ == "__main__":
    demo.launch()
    client.close()  # Ensure we close the client connection when done