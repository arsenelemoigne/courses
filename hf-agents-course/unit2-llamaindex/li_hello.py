"""Smallest possible LlamaIndex call, on your local Ollama. No token, no network."""
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="qwen2.5-coder:3b",
    temperature=0.7,
    request_timeout=180.0,     # default is 30s — too short while Ollama loads the model
    context_window=16384,
    additional_kwargs={"num_predict": 100},   # Ollama's name for max_tokens
)

response = llm.complete("Hello, how are you?")
print(response)
