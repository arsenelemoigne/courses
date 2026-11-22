"""UNIT 2.2 · THE AGENT LOOP WITH THE LID OFF

Same question as li_06_agent.py, run twice - once as FunctionAgent (native
JSON tool calling) and once as ReActAgent (the text protocol) - with every
stage printed: the full message history sent to Ollama, the tool schemas
attached to the request, the raw reply, what the framework PARSED out of it,
the tool execution, the observation, and the event sequence.

Instrumentation is done by SUBCLASSING Ollama. (Monkeypatching the class
attributes does not work here: llama_index wraps these methods with wrapt,
which binds `self` itself, so calling the retrieved function with an explicit
self raises "missing a required argument: 'tools'".)

    python li_06b_trace.py
"""
import asyncio, json, textwrap
from typing import Any, Sequence

from llama_index.core.agent.workflow import (
    AgentWorkflow, AgentStream, ToolCall, ToolCallResult,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

MODEL    = "qwen2.5-coder:3b"
QUESTION = "What is 1234 multiplied by 4567? Use the tool."

R="\033[0m"; B="\033[1m"; DIM="\033[2m"
SRC="\033[38;5;39m"; LLM="\033[38;5;214m"; IN="\033[38;5;220m"
OUT="\033[38;5;41m"; TOOL="\033[38;5;170m"; ERR="\033[38;5;203m"

N = [0]


def rule(c, title, sub=""):
    print(f"\n{c}{'─'*94}{R}")
    print(f"{c}{B}  {title}{R}" + (f"   {DIM}{sub}{R}" if sub else ""))
    print(f"{c}{'─'*94}{R}")


def wrapped(c, text, indent="  ", width=88):
    for line in str(text).split("\n"):
        for piece in (textwrap.wrap(line, width) or [""]):
            print(f"{indent}{c}│{R} {piece}")


def show_messages(msgs):
    print(f"  {IN}│{R} {B}{len(msgs)} messages, in order:{R}")
    for i, m in enumerate(msgs):
        role = getattr(m, "role", "?")
        content = str(getattr(m, "content", "") or "")
        extra = getattr(m, "additional_kwargs", {}) or {}
        print(f"  {IN}│{R}")
        print(f"  {IN}│{R} {B}[{i}] role={role}{R}"
              + (f"  {DIM}additional_kwargs={list(extra)}{R}" if extra else ""))
        wrapped(IN, content or "(empty)", indent="  ")


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    print(f"\n  {OUT}{B}>>> THE REAL PYTHON FUNCTION RAN: multiply({a}, {b}) = {a*b}{R}\n")
    return a * b


class TracingOllama(Ollama):
    """Subclass, not monkeypatch. super() keeps wrapt's binding intact."""

    def _prepare_chat_with_tools(self, tools: Sequence[Any], **kwargs: Any):
        payload = super()._prepare_chat_with_tools(tools, **kwargs)
        N[0] += 1
        rule(IN, f"LLM CALL {N[0]} — WHAT IS SENT TO OLLAMA",
             "rebuilt from scratch every pass; nothing is remembered server-side")
        show_messages(payload.get("messages", []))
        specs = payload.get("tools") or []
        print(f"\n  {TOOL}│{R} {B}tool schemas in the HTTP 'tools' field: {len(specs)}{R}"
              f"   {DIM}(a separate JSON field, NOT prompt text){R}")
        for s in specs:
            wrapped(TOOL, json.dumps(s, indent=2), width=88)
        return payload

    def get_tool_calls_from_response(self, response, **kwargs: Any):
        raw = str(getattr(response.message, "content", "") or "")
        native = (getattr(response.message, "additional_kwargs", {}) or {}).get("tool_calls")
        rule(LLM, f"LLM REPLY {N[0]} — WHAT CAME BACK")
        print(f"  {LLM}│{R} {B}message.content{R} {DIM}(the text the model generated){R}")
        wrapped(LLM, raw or "(empty)")
        print(f"\n  {LLM}│{R} {B}additional_kwargs['tool_calls']{R} "
              f"{DIM}(the NATIVE tool-call channel){R}")
        print(f"  {LLM}│{R}   " + (str(native) if native
              else f"{ERR}None — Ollama returned no native tool call{R}"))
        try:
            calls = super().get_tool_calls_from_response(response, **kwargs)
        except Exception as e:
            rule(ERR, "PARSE FAILED", type(e).__name__)
            wrapped(ERR, e)
            raise
        rule(TOOL, "WHAT THE FRAMEWORK PARSED OUT OF THAT REPLY")
        if calls:
            for c in calls:
                print(f"  {TOOL}│{R} {B}{c.tool_name}{R}({c.tool_kwargs})   -> WILL BE EXECUTED")
        else:
            print(f"  {ERR}│{R} {B}ZERO tool calls parsed.{R}")
            print(f"  {ERR}│{R} If the text above looks like a tool call, the model TYPED")
            print(f"  {ERR}│{R} one but the server never marked it as one. Text is not action.")
        return calls

    async def astream_chat(self, messages, **kwargs: Any):
        # ReActAgent never calls _prepare_chat_with_tools - it uses plain chat,
        # with the tools described in English inside the system prompt.
        if not kwargs.pop("_traced", False):
            N[0] += 1
            rule(IN, f"LLM CALL {N[0]} — WHAT IS SENT TO OLLAMA (plain chat, no tools field)")
            show_messages(messages)
        return await super().astream_chat(messages, **kwargs)


async def run(force_react: bool):
    N[0] = 0
    mode = "ReActAgent — the TEXT protocol" if force_react else "FunctionAgent — NATIVE JSON tool calling"
    print(f"\n\n{SRC}{'█'*94}{R}")
    print(f"{SRC}█  {B}{mode}{R}")
    print(f"{SRC}{'█'*94}{R}")

    llm = TracingOllama(model=MODEL, request_timeout=300.0, context_window=16384,
                        temperature=0.1, is_function_calling_model=not force_react)
    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(multiply)], llm=llm)
    inner = agent.agents["Agent"]
    print(f"\n  agent class chosen: {B}{type(inner).__name__}{R}")
    print(f"  {DIM}multi_agent_workflow.py:875 — FunctionAgent if "
          f"is_function_calling_model else ReActAgent{R}")

    handler = agent.run(user_msg=QUESTION)
    seq = []
    try:
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                continue
            seq.append(type(ev).__name__)
            if isinstance(ev, ToolCall):
                rule(TOOL, "EVENT: ToolCall", "the framework is about to run your Python")
                print(f"  {TOOL}│{R} {ev.tool_name}({ev.tool_kwargs})")
            elif isinstance(ev, ToolCallResult):
                rule(OUT, "EVENT: ToolCallResult", "this becomes the next Observation")
                wrapped(OUT, ev.tool_output)
        response = await handler
    except Exception as e:
        rule(ERR, "RUN FAILED", type(e).__name__)
        wrapped(ERR, e)
        response = f"<failed: {type(e).__name__}>"

    rule(SRC, "EVENT SEQUENCE FOR THE WHOLE RUN")
    print("  " + ("  ->  ".join(seq) if seq else "(none)"))
    rule(OUT, "FINAL ANSWER")
    wrapped(OUT, response)
    ran = "ToolCallResult" in seq
    ok  = "5635678" in str(response).replace(",", "")
    print(f"\n  LLM round-trips        : {B}{N[0]}{R}")
    print(f"  tool actually executed : {B}{'YES' if ran else 'NO'}{R}")
    print(f"  answer contains 5635678: {B}{'YES' if ok else 'NO'}{R}")


async def main():
    await run(force_react=False)
    await run(force_react=True)
    print(f"""
{'='*94}
WHAT TO COMPARE BETWEEN THE TWO RUNS

 1. WHERE THE TOOL LIVES. FunctionAgent puts the schema in a separate "tools"
    field of the HTTP request. ReActAgent has no such field at all - it writes
    the tool description into the system prompt as English. Scroll up.

 2. additional_kwargs['tool_calls']. That one field is the entire difference
    between the model TYPING a call and the framework EXECUTING one.

 3. Whether ">>> THE REAL PYTHON FUNCTION RAN" ever appeared, and how many
    LLM round-trips each mode needed.
{'='*94}""")


asyncio.run(main())
