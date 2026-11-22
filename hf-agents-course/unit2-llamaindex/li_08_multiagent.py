"""UNIT 2.2 · MULTI-AGENT WORKFLOW — two agents, a handoff, and shared state

The course's final example, run locally. Uses ReActAgent EXPLICITLY, which is
the branch that actually worked on your machine (li_06b proved FunctionAgent
silently fails with qwen2.5-coder:3b on Ollama - the model types a correct
tool call and the server never marks it as one).

THE INTERESTING BIT: root_agent is multiply_agent, but the question is an
ADDITION. So the run must HAND OFF to add_agent. Handoff is not magic - it is
a tool named `handoff` that the framework injects into every agent. Watch for
it in the trace.

Everything is traced: which agent is speaking, every tool call, every handoff,
and the function-call counter kept in shared state.

    uv pip install pyvis          # for the drawing at the end
    python li_08_multiagent.py
"""
import asyncio

from llama_index.core.agent.workflow import (
    AgentWorkflow, ReActAgent, AgentStream, AgentInput, AgentOutput,
    ToolCall, ToolCallResult,
)
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama

MODEL = "qwen2.5-coder:3b"
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
C="\033[38;5;39m"; Y="\033[38;5;220m"; G="\033[38;5;41m"; V="\033[38;5;170m"


# ── tools. They take ctx, so they can read and write the shared state. ─────
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    s = await ctx.store.get("state")
    s["num_fn_calls"] += 1
    await ctx.store.set("state", s)
    print(f"\n  {G}{B}>>> PYTHON RAN: add({a}, {b}) = {a+b}"
          f"   (call #{s['num_fn_calls']}){R}\n")
    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    s = await ctx.store.get("state")
    s["num_fn_calls"] += 1
    await ctx.store.set("state", s)
    print(f"\n  {G}{B}>>> PYTHON RAN: multiply({a}, {b}) = {a*b}"
          f"   (call #{s['num_fn_calls']}){R}\n")
    return a * b


llm = Ollama(model=MODEL, request_timeout=300.0, context_window=16384, temperature=0.1)

multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

add_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

workflow = AgentWorkflow(
    agents=[multiply_agent, add_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)


async def ask(question: str):
    print(f"\n{C}{'='*82}\n  {B}Q: {question}{R}")
    print(f"{C}  root agent: multiply_agent   |   available: multiply_agent, add_agent{R}")
    print(f"{C}{'='*82}{R}")

    ctx = Context(workflow)
    handler = workflow.run(user_msg=question, ctx=ctx)

    current = None
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            continue
        if isinstance(ev, AgentInput):
            name = getattr(ev, "current_agent_name", None)
            if name != current:
                current = name
                print(f"\n  {V}{B}[ AGENT: {name} ]{R} {DIM}is now the one being prompted{R}")
        elif isinstance(ev, ToolCall):
            tag = "HANDOFF" if ev.tool_name == "handoff" else "TOOL"
            print(f"  {Y}[{tag}]{R} {ev.tool_name}({ev.tool_kwargs})")
        elif isinstance(ev, ToolCallResult):
            out = str(ev.tool_output)
            print(f"  {G}[RESULT]{R} {out[:160]}")

    response = await handler
    state = await ctx.store.get("state")
    print(f"\n  {B}FINAL:{R} {response}")
    print(f"  {B}num_fn_calls in shared state:{R} {state['num_fn_calls']}")


async def main():
    # Three questions, chosen to isolate one variable: DIFFICULTY.
    await ask("Can you add 5 and 3?")               # handoff, easy   -> counter?
    await ask("Can you add 48293 and 71856?")       # handoff, hard   -> counter?
    await ask("What is 1234 multiplied by 4567?")   # no handoff, hard -> counter?

    print(f"\n{C}{'='*82}\n  DRAW THE AGENT WORKFLOW{R}")
    try:
        try:
            from llama_index.utils.workflow import draw_all_possible_flows      # correct
        except ImportError:
            from llama_index.core.workflow.drawing import draw_all_possible_flows  # broken shim
            print("  !! falling back to the DEPRECATED drawer, which finds no steps")
            print("  !! and draws an empty graph. Run: uv pip install llama-index-utils-workflow")
        draw_all_possible_flows(workflow, filename="flow_agents.html")
        print(f"  {G}wrote flow_agents.html{R}")
        print("  This is the AgentWorkflow's OWN event graph - setup, run_agent_step,")
        print("  call_tool, aggregate results - not a picture of your two agents.")
        print("  AgentWorkflow is itself just a Workflow, which is the point.")
    except ImportError as e:
        print(f"  needs pyvis:  uv pip install pyvis   ({e})")


asyncio.run(main())
