# app.py

import os
import weaviate
import gradio as gr
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load models
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=500)

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

def rag_answer(question, k, show_chunks):
    chunks = retrieve(question, k)
    context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"Answer the biomedical question using only the context.\n\n{context}\n\nQuestion: {question}"
    result = generator(prompt)[0]["generated_text"]

    if show_chunks:
        return f"📄 Top {k} Retrieved Chunks:\n\n{context}\n\n🤖 Answer:\n{result.strip()}"
    else:
        return result.strip()

# Gradio UI
demo = gr.Interface(
    fn=rag_answer,
    inputs=[
        gr.Textbox(label="Ask a biomedical question"),
        gr.Slider(minimum=1, maximum=10, value=3, label="Top K Documents"),
        gr.Checkbox(label="Show Retrieved Chunks", value=True),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🧬 BioASQ RAG Demo",
    description="Ask a biomedical question and retrieve relevant passages from BioASQ using Weaviate + MiniLM + FLAN-T5.",
)

if __name__ == "__main__":
    demo.launch()
    client.close()
