"""UNIT 2.2 · SIDE-QUEST — LOOK AT THE GEOMETRY DIRECTLY

Not in the course. This is the step that makes stage 3 concrete.

We embed your three chunks and several questions with the SAME model, then
print the cosine similarity between every question and every chunk.

Cosine similarity = the cosine of the angle between two 384-dimensional
vectors. 1.0 = same direction, 0.0 = perpendicular (unrelated). It is the
ONLY thing a vector database ever computes. Retrieval is: embed, compare,
sort, take the top k. There is nothing else in it.
"""
import numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

docs = SimpleDirectoryReader(input_dir="./data").load_data()
names = [d.metadata["file_name"].replace(".md", "") for d in docs]
chunk_vecs = np.array([embed.get_text_embedding(d.text) for d in docs])

QUESTIONS = [
    "What did we eat at the historical society dinner?",
    "Who is vegetarian?",
    "What is happening on 11 April?",
    "What battles took place in New York in the American Revolution?",
]

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"\nvector shape: {chunk_vecs.shape}   ({len(docs)} chunks x 384 dimensions)\n")

print("SIMILARITY OF EACH QUESTION TO EACH CHUNK")
print("=" * 78)
print(f"{'':52}" + "".join(f"{n[:11]:>13}" for n in names))
for q in QUESTIONS:
    sims = [cos(np.array(embed.get_query_embedding(q)), v) for v in chunk_vecs]
    best = sims.index(max(sims))
    row = "".join(f"{s:>13.3f}" for s in sims)
    print(f"{q[:50]:52}{row}   -> {names[best]}")

print("\n\nSIMILARITY OF THE CHUNKS TO EACH OTHER")
print("=" * 78)
print(f"{'':22}" + "".join(f"{n[:11]:>13}" for n in names))
for i, n in enumerate(names):
    print(f"{n:22}" + "".join(f"{cos(chunk_vecs[i], v):>13.3f}" for v in chunk_vecs))

print("\n\nTHE CONTROL — a question with no answer anywhere in your files:")
q = QUESTIONS[-1]
sims = [cos(np.array(embed.get_query_embedding(q)), v) for v in chunk_vecs]
print(f"  {q!r}")
print(f"  best match: {names[sims.index(max(sims))]}  at {max(sims):.3f}")
print("\n  Note it still returns a winner. Cosine similarity has no concept of")
print("  'nothing here is relevant'. It ranks. It never abstains. Remember this")
print("  when you run li_05.")
