"""UNIT 2.2 · USING AGENTS IN LLAMAINDEX — run locally

THREE CHANGES FROM THE COURSE

1. HuggingFaceInferenceAPI(Qwen2.5-Coder-32B)  ->  local Ollama.

2. The course snippet BUILDS an agent and never RUNS it. It stops at
   `agent = AgentWorkflow.from_tools_or_functions(...)`. Nothing happens.
   AgentWorkflow is async-only, so running it needs `await agent.run(...)`,
   which in a .py file must be wrapped in asyncio.run(). (In IPython/Colab
   the bare `await` works, which is why the course never noticed.)

3. THE TRAP. Read from_tools_or_functions in multi_agent_workflow.py:875 -

       agent_cls = FunctionAgent if llm.metadata.is_function_calling_model
                   else ReActAgent

   and LlamaIndex's Ollama class sets `is_function_calling_model: bool = True`
   BY DEFAULT, for every model, whether or not it can actually call functions.
   So you silently always get FunctionAgent. If your local model's Ollama
   template has no tool support, it fails or invents an answer.

   Set FORCE_REACT = True below to take the other branch.

   This is exactly the Unit 2.1 distinction you already know:
       FunctionAgent = ToolCallingAgent  -> native JSON tool calls, needs
                                            model + server support
       ReActAgent    = the text protocol -> Thought/Action/Observation parsed
                                            out of plain text, works anywhere
"""
import asyncio

from llama_index.core.agent.workflow import (
    AgentWorkflow, FunctionAgent, ReActAgent, ToolCall, ToolCallResult, AgentStream,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

MODEL       = "qwen2.5-coder:3b"
FORCE_REACT = False        # flip to True to use the text protocol instead


# The tool. Type hints, function name and docstring are ALL parsed into the
# schema the model sees - exactly like @tool in smolagents.
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


llm = Ollama(
    model=MODEL,
    request_timeout=300.0,
    context_window=16384,
    temperature=0.1,
    is_function_calling_model=not FORCE_REACT,
)

agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm,
)

inner = agent.agents["Agent"]
print(f"model            : {MODEL}")
print(f"is_function_...  : {llm.metadata.is_function_calling_model}")
print(f"agent class CHOSEN: {type(inner).__name__}")
print(f"tools            : {[t.metadata.name for t in inner.tools]}\n")


async def main():
    question = "What is 1234 multiplied by 4567? Use the tool."
    print("=" * 70)
    print("Q:", question, "\n")

    handler = agent.run(user_msg=question)

    # Stream the events so the loop is visible instead of a black box.
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"  [TOOL CALL]   {ev.tool_name}({ev.tool_kwargs})")
        elif isinstance(ev, ToolCallResult):
            print(f"  [TOOL RESULT] {ev.tool_output}")
        elif isinstance(ev, AgentStream) and ev.delta:
            print(ev.delta, end="", flush=True)

    response = await handler
    print("\n\nFINAL:", response)
    print("\n  1234 * 4567 = 5635678 — check whether it actually called the tool")
    print("  or just did the arithmetic in its head. Both are possible, and")
    print("  only one of them is an agent.")


asyncio.run(main())
