"""UNIT 2.2 · COMPONENTS · STAGE 2 — CHUNKING + EMBEDDING (no storage yet)

TWO CHANGES FROM THE COURSE, both explained:

1. The course writes  `nodes = await pipeline.arun(documents=[Document.example()])`
   `await` at the top level is legal ONLY inside IPython/Jupyter/Colab.
   In a .py file it is a SyntaxError. The sync twin is `.run()`, used below.

2. The course ingests `Document.example()` - a built-in dummy - even though it
   just taught you to load YOUR documents. We ingest the real ones, so the
   progression actually connects.

First run downloads BAAI/bge-small-en-v1.5 (~130 MB). After that it is cached.
"""
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents = SimpleDirectoryReader(input_dir="./data").load_data()
print(f"in : {len(documents)} documents")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes = pipeline.run(documents=documents)          # .run(), not await .arun()
print(f"out: {len(nodes)} Node objects\n")

for i, n in enumerate(nodes[:3]):
    print(f"--- node {i} ---")
    print(f"  from      : {n.metadata.get('file_name','?')}")
    print(f"  embedding : {len(n.embedding)} floats, first three "
          f"{[round(x, 4) for x in n.embedding[:3]]}")
    print(f"  text      : {n.text[:160]!r}\n")

print("That vector IS the 'numerical representation of meaning' the page describes.")
print("384 numbers per chunk. Nothing else about the chunk is searchable.")
