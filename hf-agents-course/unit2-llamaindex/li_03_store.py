"""UNIT 2.2 · COMPONENTS · STAGE 3+4 — STORING AND INDEXING

Same pipeline as stage 2, plus a vector_store. That one extra argument is the
whole difference: the nodes now land in Chroma on disk instead of a Python list.

CHANGED FROM THE COURSE: chunk_size=25. Twenty-five tokens is about one short
sentence - it makes a nice screenshot and terrible retrieval, because no chunk
ever contains enough context to answer anything. 256 below.

Needs:  uv pip install chromadb llama-index-vector-stores-chroma
"""
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DB_PATH = "./alfred_chroma_db"

db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

documents = SimpleDirectoryReader(input_dir="./data").load_data()

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,          # <-- the only new line vs. stage 2
)
nodes = pipeline.run(documents=documents)
print(f"ingested {len(nodes)} nodes into {DB_PATH}")
print(f"collection now holds {chroma_collection.count()} vectors")

# Indexing: wrap the store so it can be searched. Same embedding model as
# ingestion - if these differ, the query lands in a different vector space
# and retrieval returns noise.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Prove retrieval works before involving any LLM at all.
retriever = index.as_retriever(similarity_top_k=3)
hits = retriever.retrieve("What was served at the Gotham Historical Society dinner?")
print("\nas_retriever() - pure vector search, NO language model involved:\n")
for h in hits:
    print(f"  score {h.score:.3f}  {h.metadata.get('file_name','?'):18} {h.text[:90]!r}")
