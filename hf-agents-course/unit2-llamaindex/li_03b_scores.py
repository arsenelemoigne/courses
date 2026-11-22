"""UNIT 2.2 · SIDE-QUEST — WHY CHROMA'S 'SCORE' IS NOT YOUR COSINE

li_02b said 0.752 for the best match. li_03 said 0.582. Same corpus, same
model. Two separate reasons, and this script pins down both.

REASON 1 - they were not the same question.
    li_02b : "What did we eat at the historical society dinner?"
    li_03  : "What was served at the Gotham Historical Society dinner?"
    'Gotham' appears in calendar.md, which is why calendar climbed.

REASON 2 - they are not the same MEASURE.
    Chroma's default index space is L2 (squared euclidean), not cosine.
    LlamaIndex then converts, at llama_index/vector_stores/chroma/base.py:472

        similarity_score = math.exp(-distance)

    So the number printed by as_retriever() is exp(-d^2), not cos.
    For unit-length vectors  d^2 = 2 - 2*cos, therefore

        printed score = exp(2*cos - 2)

    Below we ask ONE identical question both ways and check that formula.
"""
import math
import numpy as np
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

Q = "What was served at the Gotham Historical Society dinner?"

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="./alfred_chroma_db")
collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed)

print(f"question: {Q!r}")
print(f"chroma index space: {collection.metadata}\n")

qv = np.array(embed.get_query_embedding(Q))

raw = collection.get(include=["embeddings", "metadatas", "documents"])
by_id = {i: (np.array(e), (md or {}).get("file_name", "?"))
         for i, e, md in zip(raw["ids"], raw["embeddings"], raw["metadatas"])}

hits = index.as_retriever(similarity_top_k=3).retrieve(Q)

print(f"{'file':20}{'cosine':>10}{'exp(2cos-2)':>14}{'chroma score':>15}{'match':>8}")
print("-" * 70)
for h in hits:
    vec, name = by_id.get(h.node.node_id, (None, h.metadata.get("file_name", "?")))
    if vec is None:
        continue
    cos = float(qv @ vec / (np.linalg.norm(qv) * np.linalg.norm(vec)))
    pred = math.exp(2 * cos - 2)
    ok = "yes" if abs(pred - h.score) < 0.01 else "NO"
    print(f"{name:20}{cos:>10.4f}{pred:>14.4f}{h.score:>15.4f}{ok:>8}")

print("""
WHAT TO TAKE FROM THIS

1. A similarity score is only comparable to other scores from the SAME store,
   the SAME metric and the SAME question. Never treat it as an absolute.
2. exp(-d) squashes everything toward 0, so Chroma's numbers always look
   lower and more spread out than raw cosine. Nothing got worse between
   li_02b and li_03; the ruler changed.
3. This is exactly why a fixed cutoff ("reject below 0.55") is unsafe. The
   cutoff would have to be re-derived for every store and metric you use.""")
