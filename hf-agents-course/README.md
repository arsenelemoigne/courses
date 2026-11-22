# AI Agents course

Hugging Face's [AI Agents course](https://huggingface.co/learn/agents-course). I ran everything on small local models with [Ollama](https://ollama.com) (mostly `qwen3:1.7b`) instead of a hosted API, to see what an agent does with a weak model.

| Folder | Course unit |
|---|---|
| [unit1-dummy-agent](unit1-dummy-agent) | Unit 1: an agent loop written by hand, without a framework |
| [unit1-first-agent](unit1-first-agent) | Unit 1: the first `smolagents` agent with tools and a Gradio UI |
| [unit2-llamaindex](unit2-llamaindex) | Unit 2: LlamaIndex, step by step from loading documents to a multi-agent workflow |
| [unit2-langgraph](unit2-langgraph) | Unit 2: LangGraph (email sorting and document analysis for "Alfred") |
| [unit3-agentic-rag](unit3-agentic-rag) | Unit 3: agentic RAG for the gala guest list |

To run a script: install Ollama, `ollama pull qwen3:1.7b`, then `pip install` the libraries it imports.
