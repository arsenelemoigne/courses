"""UNIT 2.2 · CREATING AGENTIC WORKFLOWS — the Workflow class

FOUR FIXES TO THE COURSE CODE

1. THE @step DECORATOR IS MISSING FROM EVERY SNIPPET. The page says "decorating
   your functions with @step" and then never writes it. Without it the class
   has no steps at all and validation fails. (Same bug as the missing @tool in
   Unit 2.1 - the prose is right, the code is wrong.)

2. `result = await w.run()` - top-level await, notebook only. asyncio.run here.

3. The import path for the drawing helper. The course writes
       from llama_index.utils.workflow import draw_all_possible_flows
   That module does not exist in this version. It is
       from llama_index.core.workflow.drawing import draw_all_possible_flows

4. draw_all_possible_flows needs pyvis:   uv pip install pyvis

A Workflow is NOT an agent. No model is called anywhere in this file. It is a
plain event-driven state machine: steps are triggered by the TYPE of event
they receive, and emit events that trigger other steps. The type hints are
not documentation - they are the wiring.
"""
import asyncio, random

from llama_index.core.workflow import (
    Context, Event, StartEvent, StopEvent, Workflow, step,
)

B="\033[1m"; R="\033[0m"; C="\033[38;5;39m"; G="\033[38;5;41m"; Y="\033[38;5;220m"


def title(t):
    print(f"\n{C}{'='*78}\n  {B}{t}{R}{C}\n{'='*78}{R}")


# ── 1. one step ────────────────────────────────────────────────────────────
class MyWorkflow(Workflow):
    @step                                   # <- the course omits this line
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")


# ── 2. two steps, joined by a custom event ─────────────────────────────────
class ProcessingEvent(Event):
    intermediate_result: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        print(f"  {Y}step_one{R} received {type(ev).__name__}, emitting ProcessingEvent")
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        print(f"  {Y}step_two{R} received {type(ev).__name__}")
        return StopEvent(result=f"Finished processing: {ev.intermediate_result}")


# ── 3. a loop, via the union operator ──────────────────────────────────────
class LoopEvent(Event):
    loop_output: str


class LoopWorkflow(Workflow):
    # StartEvent | LoopEvent on the INPUT and ProcessingEvent | LoopEvent on the
    # OUTPUT is what creates the cycle. The engine reads these annotations.
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print(f"  {Y}step_one{R}: bad thing happened -> looping back to myself")
            return LoopEvent(loop_output="Back to step one.")
        print(f"  {G}step_one{R}: good thing happened -> moving on")
        return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        return StopEvent(result=f"Finished processing: {ev.intermediate_result}")


# ── 4. shared state via Context ────────────────────────────────────────────
class StatefulWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        await ctx.store.set("query", "What is the capital of France?")
        q = await ctx.store.get("query")
        print(f"  stored then read back from the shared store: {q!r}")
        return StopEvent(result=f"answered: {q}")


async def main():
    title("1 · SINGLE STEP")
    print(await MyWorkflow(timeout=10, verbose=False).run())

    title("2 · TWO STEPS, CONNECTED BY A CUSTOM EVENT")
    print(await MultiStepWorkflow(timeout=10, verbose=False).run())

    title("3 · A LOOP  (random - run it a few times)")
    print(await LoopWorkflow(timeout=30, verbose=False).run())

    title("4 · SHARED STATE VIA Context")
    print(await StatefulWorkflow(timeout=10, verbose=False).run())

    title("5 · DRAW THE GRAPH")
    try:
        try:
            from llama_index.utils.workflow import draw_all_possible_flows      # correct
        except ImportError:
            from llama_index.core.workflow.drawing import draw_all_possible_flows  # broken shim
            print("  !! falling back to the DEPRECATED drawer, which finds no steps")
            print("  !! and draws an empty graph. Run: uv pip install llama-index-utils-workflow")
        draw_all_possible_flows(LoopWorkflow(), filename="flow_loop.html")
        draw_all_possible_flows(MultiStepWorkflow(), filename="flow_multistep.html")
        print(f"  {G}wrote flow_loop.html and flow_multistep.html{R}")
        print("  open them in a browser - nodes are events, edges are steps.")
        print("  the self-edge on step_one IS the `| LoopEvent` in the annotation.")
    except ImportError as e:
        print(f"  drawing needs pyvis:  uv pip install pyvis   ({e})")


asyncio.run(main())
