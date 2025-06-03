import gradio as gr
import chromadb
from chromadb.config import Settings
from transformers import pipeline

# Load Chroma vector DB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="bioasq_passages")

# Retrieval
def retrieve(question: str, k: int = 5):
    res = collection.query(query_texts=[question], n_results=k)
    return res["documents"][0]

# Generation (FLAN-T5)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=500
)

# RAG Pipeline
def rag_answer(question: str, k: int = 5):
    context_chunks = retrieve(question, k)
    prompt = "Answer the biomedical question factually using ONLY the context.\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        prompt += f"[Context {i}]\n{chunk}\n\n"
    prompt += f"Question: {question}"
    answer = generator(prompt, truncation=True)[0]["generated_text"]
    return answer.strip()

# Gradio UI
demo = gr.Interface(
    fn=rag_answer,
    inputs=gr.Textbox(placeholder="Ask a biomedical question...", lines=2, label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="🧬 BioRAG: Biomedical Question Answering",
    description="Ask a question based on biomedical literature (BioASQ + PubMed abstracts). Powered by Chroma + FLAN-T5."
)

if __name__ == "__main__":
    demo.launch()

#python chroma_transformers/app.py