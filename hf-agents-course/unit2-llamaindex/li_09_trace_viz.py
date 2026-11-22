"""UNIT 2.2 · AN INTERACTIVE TRACE OF A REAL RUN

pyvis draws the graph a workflow COULD take, from type hints. This draws the
path one run ACTUALLY took, with every payload attached.

Runs the multi-agent workflow, captures every stage, and writes a
self-contained HTML: a timeline on the left, click any node to see the full
detail on the right - the entire message list sent to Ollama, the raw reply,
the parsed Thought / Action / Action Input, the tool's input and output, the
Observation, and the rebuilt context for the next pass.

    python li_09_trace_viz.py
    open run_trace.html
"""
import asyncio, html, json, re, time
from typing import Any, Sequence

from llama_index.core.agent.workflow import (
    AgentWorkflow, ReActAgent, AgentStream, AgentInput, AgentOutput,
    ToolCall, ToolCallResult,
)
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama

MODEL    = "qwen2.5-coder:3b"
QUESTION = "Can you add 48293 and 71856, then multiply the result by 3?"

REC: list = []
CUR  = {"agent": "?"}
T0   = time.time()


def rec(kind, title, body="", meta=None, messages=None):
    REC.append({
        "i": len(REC), "kind": kind, "title": title, "agent": CUR["agent"],
        "t": round(time.time() - T0, 2), "body": str(body),
        "meta": meta or {}, "messages": messages or [],
    })


# ── tools ──────────────────────────────────────────────────────────────────
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    s = await ctx.store.get("state"); s["num_fn_calls"] += 1
    await ctx.store.set("state", s)
    print(f"  >>> PYTHON RAN add({a}, {b}) = {a+b}")
    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    s = await ctx.store.get("state"); s["num_fn_calls"] += 1
    await ctx.store.set("state", s)
    print(f"  >>> PYTHON RAN multiply({a}, {b}) = {a*b}")
    return a * b


# ── capture what goes to and comes back from the model ─────────────────────
class TracingOllama(Ollama):
    async def astream_chat(self, messages, **kwargs: Any):
        msgs = [{"role": str(getattr(m, "role", "?")),
                 "content": str(getattr(m, "content", "") or "")} for m in messages]
        chars = sum(len(m["content"]) for m in msgs)
        rec("prompt", f"PROMPT SENT — {len(msgs)} messages, {chars:,} chars",
            body="", meta={"messages": len(msgs), "chars": chars}, messages=msgs)
        gen = await super().astream_chat(messages, **kwargs)

        async def wrap():
            last = None
            async for chunk in gen:
                last = chunk
                yield chunk
            if last is not None:
                raw = str(getattr(last.message, "content", "") or "")
                rec("reply", "MODEL REPLY (raw, complete)", body=raw,
                    meta={"chars": len(raw)})
                parse_react(raw)
        return wrap()


def parse_react(raw: str):
    th = re.search(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Answer:)|\Z)", raw, re.S)
    if th and th.group(1).strip():
        rec("thought", "THOUGHT — the model's stated reasoning", th.group(1).strip())
    ac = re.search(r"Action:\s*(\S+)", raw)
    ai = re.search(r"Action Input:\s*(\{.*?\})\s*(?:\n|\Z)", raw, re.S)
    if ac:
        rec("action", f"ACTION — the model asks for `{ac.group(1)}`",
            body=(ai.group(1) if ai else "(no Action Input parsed)"),
            meta={"tool": ac.group(1),
                  "note": "still only TEXT at this point. nothing has run."})
    an = re.search(r"Answer:\s*(.*)", raw, re.S)
    if an and an.group(1).strip():
        rec("answer", "ANSWER — the model says it is done", an.group(1).strip())


llm = TracingOllama(model=MODEL, request_timeout=300.0,
                    context_window=16384, temperature=0.1)

multiply_agent = ReActAgent(name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply], llm=llm)
add_agent = ReActAgent(name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add], llm=llm)

workflow = AgentWorkflow(agents=[multiply_agent, add_agent],
    root_agent="multiply_agent", initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}")


KIND_CSS = {
    "task":     ("#6d28d9", "TASK"),
    "agent":    ("#0e7490", "AGENT"),
    "prompt":   ("#b45309", "PROMPT"),
    "reply":    ("#c2410c", "REPLY"),
    "thought":  ("#7c3aed", "THINK"),
    "action":   ("#a16207", "ACT"),
    "tool":     ("#1d4ed8", "TOOL"),
    "result":   ("#15803d", "RESULT"),
    "handoff":  ("#be185d", "HANDOFF"),
    "answer":   ("#166534", "ANSWER"),
    "final":    ("#166534", "FINAL"),
}


def build_html(records, question, summary):
    """A vis-network graph of the ACTUAL run. Same library pyvis wraps
    (vis-network 9.1.2 from cdnjs — the exact URL pyvis itself uses), driven by
    real trace data instead of by @step type hints, with every payload attached
    to its node."""
    nodes, edges = [], []
    for r in records:
        colour, label = KIND_CSS.get(r["kind"], ("#555", r["kind"]))
        short = (r["body"] or "").replace("\n", " ")[:38]
        nodes.append({
            "id": r["i"],
            "label": f"{label}\n{short}" if short else label,
            "color": {"background": colour, "border": "#0b0d12",
                      "highlight": {"background": colour, "border": "#7dd3fc"}},
            "shape": "box" if r["kind"] in ("tool", "handoff", "result") else "ellipse",
            "borderWidth": 3 if r["kind"] in ("tool", "handoff") else 1,
        })
        if r["i"] > 0:
            edges.append({"from": r["i"] - 1, "to": r["i"],
                          "color": {"color": "#3b4453", "highlight": "#7dd3fc"}})

    data = json.dumps(records).replace("</", "<\\/")
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{lab}</span>'
        for k, (c, lab) in KIND_CSS.items() if k != "final")

    tpl = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Agent run graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 -apple-system,"Segoe UI",sans-serif;background:#0f1115;color:#e6e3dd;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:12px 20px;border-bottom:1px solid #2a2f3a;background:#151922;flex:none}
h1{margin:0 0 3px;font-size:15px}
.q{color:#9aa3b2;font-size:12.5px}
.sum{margin-top:7px;display:flex;gap:18px;flex-wrap:wrap;font-size:11.5px;color:#9aa3b2}
.sum b{color:#e6e3dd;font-variant-numeric:tabular-nums}
.legend{margin-top:8px;display:flex;gap:11px;flex-wrap:wrap;font-size:10.5px;color:#9aa3b2;align-items:center}
.lg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px}
button{background:#262c37;color:#e6e3dd;border:1px solid #3b4453;border-radius:4px;
       padding:3px 10px;font:11px inherit;cursor:pointer}
button:hover{background:#31394a}
main{flex:1;display:flex;min-height:0}
#graph{flex:1;min-width:0;background:#0f1115}
#detail{width:460px;flex:none;overflow-y:auto;padding:18px 22px;border-left:1px solid #2a2f3a;background:#12161e}
#detail h2{margin:0 0 4px;font-size:14px}
#detail .meta{color:#7c8697;font-size:11.5px;margin-bottom:15px}
pre{background:#151922;border:1px solid #262c37;border-left:3px solid #3b4453;border-radius:4px;
    padding:10px 12px;white-space:pre-wrap;word-break:break-word;
    font:11.5px/1.55 ui-monospace,Menlo,monospace;color:#cfd6e1;margin:0 0 11px}
h3{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:#7c8697;margin:17px 0 7px}
.msg{border:1px solid #262c37;border-radius:4px;margin-bottom:8px;overflow:hidden}
.msg .r{background:#1a1f29;padding:5px 10px;font-size:10.5px;letter-spacing:.04em;color:#9aa3b2}
.msg pre{border:0;border-radius:0;margin:0;background:#12161e;max-height:260px;overflow:auto}
.note{border-left:3px solid #a16207;background:#1c1a12;padding:9px 12px;border-radius:3px;
      color:#e0cfa0;font-size:12px;margin-bottom:13px}
.hint{color:#7c8697;font-size:12px}
</style></head><body>
<header>
  <h1>Agent run graph — the path this run actually took</h1>
  <div class="q">__QUESTION__</div>
  <div class="sum">__SUMMARY__</div>
  <div class="legend">__LEGEND__
    <button id="phys">freeze layout</button>
    <button id="fit">fit</button>
    <button id="hier">line it up</button>
  </div>
</header>
<main>
  <div id="graph"></div>
  <div id="detail"><p class="hint">Drag the nodes around. Click one to see everything that
  passed through it. Scroll to zoom.</p></div>
</main>
<script>
const DATA=__DATA__;
const nodes=new vis.DataSet(__NODES__), edges=new vis.DataSet(__EDGES__);
const opts={
  nodes:{font:{color:"#f2f4f8",size:13,face:"-apple-system,Segoe UI,sans-serif",multi:false},
         margin:9,widthConstraint:{maximum:190}},
  edges:{arrows:{to:{enabled:true,scaleFactor:.6}},smooth:{type:"cubicBezier",roundness:.5},width:1.6},
  physics:{enabled:true,barnesHut:{gravitationalConstant:-9000,springLength:150,springConstant:.03,
           avoidOverlap:.25},stabilization:{iterations:220}},
  interaction:{hover:true,dragNodes:true,dragView:true,zoomView:true,tooltipDelay:150},
  layout:{improvedLayout:true}
};
const net=new vis.Network(document.getElementById("graph"),{nodes,edges},opts);
const detail=document.getElementById("detail");
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function show(i){
  const d=DATA[i]; if(!d) return;
  let h='<h2>'+esc(d.title)+'</h2><div class="meta">step '+(i+1)+' of '+DATA.length
       +' &nbsp;·&nbsp; agent: '+esc(d.agent)+' &nbsp;·&nbsp; t+'+d.t+'s</div>';
  if(d.meta&&d.meta.note) h+='<div class="note">'+esc(d.meta.note)+'</div>';
  if(d.body) h+='<pre>'+esc(d.body)+'</pre>';
  if(d.messages&&d.messages.length){
    h+='<h3>the complete context sent to the model &mdash; rebuilt from scratch, every time</h3>';
    d.messages.forEach((m,k)=>{h+='<div class="msg"><div class="r">['+k+'] '+esc(m.role)+' &middot; '
      +m.content.length.toLocaleString()+' chars</div><pre>'+esc(m.content)+'</pre></div>';});
  }
  detail.innerHTML=h; detail.scrollTop=0;
  net.selectNodes([i]);
}
net.on("click",p=>{ if(p.nodes.length) show(p.nodes[0]); });
let physOn=true;
document.getElementById("phys").onclick=e=>{
  physOn=!physOn; net.setOptions({physics:{enabled:physOn}});
  e.target.textContent = physOn ? "freeze layout" : "unfreeze";
};
document.getElementById("fit").onclick=()=>net.fit({animation:true});
let hier=false;
document.getElementById("hier").onclick=e=>{
  hier=!hier;
  net.setOptions(hier
    ? {layout:{hierarchical:{enabled:true,direction:"LR",sortMethod:"directed",
               nodeSpacing:130,levelSeparation:190}},physics:{enabled:false}}
    : {layout:{hierarchical:{enabled:false}},physics:{enabled:physOn}});
  e.target.textContent = hier ? "scatter" : "line it up";
};
net.once("stabilizationIterationsDone",()=>net.fit());
show(0);
</script></body></html>"""
    return (tpl.replace("__DATA__", data)
               .replace("__NODES__", json.dumps(nodes))
               .replace("__EDGES__", json.dumps(edges))
               .replace("__QUESTION__", html.escape(question))
               .replace("__SUMMARY__", summary)
               .replace("__LEGEND__", legend))


async def main():
    print(f"Q: {QUESTION}\n")
    rec("task", "THE TASK", QUESTION,
        meta={"note": "Nothing has happened yet. This string is the only input."})

    ctx = Context(workflow)
    handler = workflow.run(user_msg=QUESTION, ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            continue
        if isinstance(ev, AgentInput):
            name = getattr(ev, "current_agent_name", None) or CUR["agent"]
            if name != CUR["agent"]:
                CUR["agent"] = name
                rec("agent", f"CONTROL PASSES TO {name}", "",
                    meta={"note": "A different agent now receives the prompt. "
                                  "It has its own system prompt and its own tools."})
        elif isinstance(ev, ToolCall):
            kind = "handoff" if ev.tool_name == "handoff" else "tool"
            rec(kind, f"THE FRAMEWORK EXECUTES `{ev.tool_name}`",
                json.dumps(ev.tool_kwargs, indent=2),
                meta={"note": "Text became action here. The framework matched the "
                              "name against its tool dict and called real Python."})
        elif isinstance(ev, ToolCallResult):
            rec("result", "TOOL OUTPUT → becomes the next Observation",
                str(ev.tool_output),
                meta={"note": "This is appended to the transcript as a message and "
                              "re-sent with everything else on the next pass."})

    response = await handler
    state = await ctx.store.get("state")
    rec("final", "FINAL ANSWER", str(response))

    prompts = [r for r in REC if r["kind"] == "prompt"]
    tools   = [r for r in REC if r["kind"] == "tool"]
    summary = (f"<span>steps <b>{len(REC)}</b></span>"
               f"<span>LLM round-trips <b>{len(prompts)}</b></span>"
               f"<span>tools actually executed <b>{state['num_fn_calls']}</b></span>"
               f"<span>largest prompt <b>{max([p['meta']['chars'] for p in prompts] or [0]):,}</b> chars</span>"
               f"<span>wall clock <b>{round(time.time()-T0,1)}s</b></span>"
               f"<span>model <b>{MODEL}</b></span>")

    out = "run_trace.html"
    open(out, "w", encoding="utf-8").write(build_html(REC, QUESTION, summary))
    print(f"\nFINAL: {response}")
    print(f"tools actually executed: {state['num_fn_calls']}")
    print(f"\nwrote {out}  —  open run_trace.html")


asyncio.run(main())
