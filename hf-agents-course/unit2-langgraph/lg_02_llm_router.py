"""UNIT 2.3 · THE SAME GRAPH, BUT YOUR MODEL DECIDES

lg_01 is the course's version: decide_mood() is random.random() < 0.5. A coin
flip. No model anywhere.

Here the coin flip is replaced by a real call to your local Ollama. Same graph,
same three nodes, same edges - the ONLY change is that the conditional edge
asks qwen instead of the random module. That single substitution is what turns
a state machine into an agent, and it is worth seeing in isolation.

    uv pip install langgraph langchain-ollama grandalf
    python lg_02_llm_router.py
"""
from typing import Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

MODEL = "qwen2.5-coder:3b"
R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
Y = "\033[38;5;220m"; G = "\033[38;5;41m"; V = "\033[38;5;170m"; E = "\033[38;5;203m"

llm = ChatOllama(model=MODEL, temperature=0.0, num_ctx=4096)


class State(TypedDict):
    graph_state: str        # the sentence we are building
    mood: str               # what the model decided, kept for inspection
    why: str                # the model's raw reply, kept for inspection


def node_1(state: State):
    print(f"  {DIM}---Node 1---{R}")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state: State):
    print(f"  {G}---Node 2 (happy)---{R}")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state: State):
    print(f"  {V}---Node 3 (sad)---{R}")
    return {"graph_state": state["graph_state"] + " sad!"}


def classify(state: State):
    """A NODE that calls the model and writes its decision into state.

    Deliberately split from the router below. The node does the work and
    records it; the router is a pure function of state. Keeping them apart
    means the decision is inspectable and checkpointable - if the router
    itself called the LLM, re-running the graph would re-run the model."""
    prompt = (
        "Classify the emotional tone of this message. "
        "Answer with exactly one word, either happy or sad. No punctuation.\n\n"
        f"Message: {state['graph_state']}"
    )
    print(f"  {Y}[LLM]{R} asking {MODEL} to classify the tone…")
    reply = llm.invoke(prompt).content.strip()
    mood = "happy" if "happy" in reply.lower() else "sad"
    print(f"  {Y}[LLM]{R} raw reply: {reply[:70]!r}  →  parsed: {B}{mood}{R}")
    return {"mood": mood, "why": reply}


def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    """The router. Pure function of state, no side effects, no model call."""
    return "node_2" if state.get("mood") == "happy" else "node_3"


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("classify", classify)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "classify")
builder.add_conditional_edges("classify", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()


if __name__ == "__main__":
    print("\n" + "=" * 74)
    print("  THE GRAPH")
    print("=" * 74)
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"  needs grandalf: uv pip install grandalf  ({e})")
    open("lg_graph_llm.mmd", "w").write(graph.get_graph().draw_mermaid())
    print("  mermaid written to lg_graph_llm.mmd")

    TESTS = [
        "Hi, this is Lance. I just got the promotion I wanted.",
        "Hi, this is Lance. My dog died this morning.",
        "Hi, this is Lance.",
    ]
    for t in TESTS:
        print("\n" + "=" * 74)
        print(f"  INPUT: {t}")
        print("=" * 74)
        out = graph.invoke({"graph_state": t, "mood": "", "why": ""})
        print(f"\n  {B}RESULT{R} {out['graph_state']}")
        print(f"  {DIM}state carried: mood={out['mood']!r}{R}")

    print(f"""
  {B}WHAT CHANGED FROM lg_01{R}
  One function. decide_mood() went from random.random() to reading a value the
  model wrote into state. Everything else - nodes, edges, State, compile - is
  identical.

  {B}AND ONE DESIGN CHOICE WORTH COPYING{R}
  The model call lives in a NODE (classify), not in the router. The router is a
  pure function of state. That is why LangGraph can checkpoint and resume: the
  expensive, non-deterministic part is a recorded step, and the routing decision
  can be replayed from the recorded value without calling the model again.

  Put the LLM call in the router and you lose that - which is the mistake the
  course's structure would lead you into.""")
