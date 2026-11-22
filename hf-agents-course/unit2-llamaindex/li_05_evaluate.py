"""UNIT 2.2 · COMPONENTS · STAGE 5 — EVALUATION

The course snippet is not runnable Python:

    query_engine = # from the previous section
    llm = # from the previous section

An assignment whose right-hand side is only a comment is a SyntaxError. Those
two lines mean 'scroll up and re-run the earlier cell'. Rebuilt properly below.

Its example question ("What battles took place in New York City in the American
Revolution?") has no answer in our documents - which is useful, because a
faithfulness evaluator SHOULD fail an answer with no support. Both are asked.

FaithfulnessEvaluator = a second LLM call that reads the answer and the
retrieved chunks and judges whether the answer is supported. On a 3B model that
judgement is itself unreliable. That is the honest caveat the page omits.
"""
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
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

evaluator = FaithfulnessEvaluator(llm=llm)

for q in ["What was served at the Gotham Historical Society dinner?",
          "What battles took place in New York City in the American Revolution?"]:
    print("=" * 78)
    print("Q:", q)
    response = query_engine.query(q)
    print("A:", response)
    result = evaluator.evaluate_response(response=response)
    print(f"\n  passing  : {result.passing}")
    print(f"  score    : {result.score}")
    print(f"  feedback : {str(result.feedback)[:400]}\n")

print("Compare this to grounding_check in masterclass.py. Same hook in the pipeline,")
print("opposite method: yours does string containment and cannot be wrong about it;")
print("this one asks a language model, and can be.")
