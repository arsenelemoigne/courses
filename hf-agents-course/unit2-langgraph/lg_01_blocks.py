"""UNIT 2.3 · BUILDING BLOCKS OF LANGGRAPH — the course's graph, runnable

CHANGES FROM THE COURSE

1. `display(Image(graph.get_graph().draw_mermaid_png()))` does two things that
   break outside a notebook: `display` is IPython-only, and draw_mermaid_png()
   POSTS your graph to the remote mermaid.ink service to render it. We use
   draw_mermaid() (returns text, entirely local) and draw_ascii() instead.

2. Nothing else. This page is already framework-honest.

NOTE, because it matters: NO LANGUAGE MODEL RUNS IN THIS FILE. LangGraph is a
state machine, not an agent. Nodes are plain Python functions. See
lg_02_llm_router.py for the version where a model actually decides.

    uv pip install langgraph grandalf
    python lg_01_blocks.py
"""
import random
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ── 1 · STATE — one dict, threaded through every node ───────────────────────
class State(TypedDict):
    graph_state: str


# ── 2 · NODES — state in, a PARTIAL update out ─────────────────────────────
# A node returns only the keys it changes. LangGraph merges them into the
# state. That is the whole contract.
def node_1(state: State):
    print("  ---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state: State):
    print("  ---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state: State):
    print("  ---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}


# ── 3 · EDGES — a router is a function returning the NAME of the next node ──
# The Literal[...] return type is not decoration: LangGraph reads it to know
# which edges to draw. Exactly like the type hints in LlamaIndex Workflows.
def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    if random.random() < 0.5:
        return "node_2"
    return "node_3"


# ── 4 · STATEGRAPH — wire it up ────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()


if __name__ == "__main__":
    print("\n" + "=" * 74)
    print("  THE GRAPH  (drawn locally — no network, unlike draw_mermaid_png)")
    print("=" * 74)
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"  ascii art needs grandalf:  uv pip install grandalf   ({e})")

    mermaid = graph.get_graph().draw_mermaid()
    open("lg_graph.mmd", "w").write(mermaid)
    print("\n  mermaid source (also written to lg_graph.mmd — paste at mermaid.live):\n")
    for line in mermaid.strip().split("\n"):
        print("   ", line)

    print("\n" + "=" * 74)
    print("  FIVE RUNS  (the conditional edge is a coin flip)")
    print("=" * 74)
    for i in range(5):
        out = graph.invoke({"graph_state": "Hi, this is Lance."})
        print(f"  {i+1}. {out['graph_state']}")

    print("""
  Map it onto what you already know:
    State      = the dict threaded through everything   (smolagents: memory.steps)
    Node       = one step                               (smolagents: one pass)
    Edge       = who runs next                          (smolagents: hardcoded loop)
    Conditional edge = a FUNCTION deciding the next node

  The difference from every loop you have read this fortnight: control flow is
  DATA here - a graph you declare - not a `while` statement. That is what makes
  it checkpointable and resumable.""")
