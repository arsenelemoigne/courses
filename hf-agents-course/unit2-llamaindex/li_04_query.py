"""UNIT 2.2 · COMPONENTS · STAGE 4 — QUERYING

CHANGED FROM THE COURSE:
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
That is a remote, billed call to a 32B model. Below is your local Ollama.

Settings.llm / Settings.embed_model are set as well as passed explicitly:
anything inside LlamaIndex that reaches for a default otherwise tries OpenAI
and dies with 'No API key found'.

Run li_03_store.py first - this reads the database it wrote.
"""
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

MODEL = "qwen2.5-coder:3b"

llm = Ollama(model=MODEL, request_timeout=300.0, context_window=16384, temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path="./alfred_chroma_db")
vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("alfred"))
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")

QUESTIONS = [
    "What was served at the Gotham Historical Society dinner, and how was it received?",
    "Who in the household is vegetarian, and what must every menu include?",
    "What is happening on 11 April?",
]

for q in QUESTIONS:
    print("=" * 78)
    print("Q:", q)
    response = query_engine.query(q)
    print("\nA:", response, "\n")
    print("retrieved from:")            # the R in RAG, made visible
    for n in response.source_nodes:
        print(f"   score {n.score:.3f}  {n.metadata.get('file_name','?'):18} {n.text[:80]!r}")
    print()
