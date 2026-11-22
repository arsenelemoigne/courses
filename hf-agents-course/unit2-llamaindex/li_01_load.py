"""UNIT 2.2 · COMPONENTS · STAGE 1 — LOADING
Course snippet, with the placeholder path replaced by a real one.

    reader = SimpleDirectoryReader(input_dir="path/to/directory")   <-- placeholder, always fails
"""
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

print(f"loaded {len(documents)} Document objects\n")
for d in documents:
    name = d.metadata.get("file_name", "?")
    print(f"  {name:20} {len(d.text):5} chars   id={d.id_[:8]}…")

print("\nfirst 300 characters of the first document:")
print("-" * 60)
print(documents[0].text[:300])
