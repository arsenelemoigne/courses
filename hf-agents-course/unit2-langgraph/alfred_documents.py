# ============================================================================
#  ALFRED'S DOCUMENT ANALYST (HF Agents Course, Unit 2.3, "Document Analysis Graph")
# ============================================================================
#
#  What it does: you ask Alfred something. He can use 2 tools:
#    - divide(a, b)          a calculator
#    - extract_text(img)     reads the text in an image, using a model that can see
#  He decides by himself which tool to use, how many times, and when to answer.
#
#  The graph:
#
#     START
#       |
#   [assistant]  <------------.     asks the model what to do next
#       |                     |
#   tools_condition()         |     a decision: did the model ask for a tool?
#     |        |              |
#    yes       no             |
#     |        |              |
#  [tools]    END             |     runs the tool the model asked for
#     |                       |
#     '-----------------------'     ...and goes back to the assistant (THE LOOP)
#
#  THIS IS AN AGENT, unlike alfred_email.py: the arrow tools -> assistant lets
#  the model go round as many times as it wants. It's the same graph as
#  Lupin/pythonLG.py, but using LangGraph's ready-made pieces:
#     ToolNode(tools)   = pythonLG.py's run_tools
#     tools_condition   = pythonLG.py's did_it_answer
#
#    PART 1  Imports
#    PART 2  The state      the "form": the file + the messages
#    PART 3  The tools      extract_text, divide
#    PART 4  The model      which LLM, with the tools attached
#    PART 5  The node       assistant
#    PART 6  The graph      boxes and arrows, with the loop
#    PART 7  Run it         Example 1 (divide), Example 2 (read an image)
#
#  CHANGED from the course (marked "CHANGED" below):
#    - gpt-4o (OpenAI, paid) -> your local models in Ollama
#    - the image-reading model is a separate setting, VISION_MODEL, because
#      your Qwen can't see images (see Part 3)
#    - no IPython / display(): that only works in notebooks
#    - Example 2 only runs if the image file is in this folder
#
#  HOW TO READ THIS FILE
#  Like alfred_email.py: Parts 1-6 only PREPARE. The action starts in Part 7.
#  Follow the "▶ STEP" markers to read the code in the order it really runs:
#    ▶ STEP 1-5   Parts 1-6: imports, form, tools, model, node, graph (preparation)
#    ▶ STEP 6     Part 7: your question goes in, invoke() starts the graph
#    ▶ STEP 7     assistant runs: the model reads everything, answers or asks for a tool
#    ▶ STEP 8     tools_condition decides: tool asked? -> STEP 9.  No? -> END
#    ▶ STEP 9     the tools node runs the tool, then back to STEP 7
#    ▶ STEP 10    END: we print the whole conversation
# ============================================================================


# ============================================================================
#  PART 1: IMPORTS                                              ▶ STEP 1
# ============================================================================
import base64                                    # turns an image file into text, so it can be sent to a model
import os                                        # checks whether a file exists (Part 7)
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages  # the "add, don't replace" rule for messages (Part 2)
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition  # LangGraph's ready-made tool box and decision

from run_log import RunLog   # NOT from the course: saves every step of the run in runs/ (see run_log.py)


# ============================================================================
#  PART 2: THE STATE (the "form")                               ▶ STEP 2
# ============================================================================
# Two fields:
#   input_file : the path of an image to analyse, or None
#   messages   : the conversation (same as Lupin's `messages` list)
#
# NEW IDEA: Annotated[..., add_messages].
# Normally, when a node returns {"messages": [x]}, LangGraph REPLACES the field
# with [x]. add_messages changes the rule to "ADD x at the end of the list".
# That's what lets the conversation grow, like messages.append() in lupin.py.
# (pythonLG.py's MessagesState is a ready-made state with exactly this field.)
class AgentState(TypedDict):
    input_file: Optional[str]                               # e.g. "Batman_training_and_meals.png", or None
    messages: Annotated[list[AnyMessage], add_messages]     # the conversation; new messages get ADDED


# ============================================================================
#  PART 3: THE TOOLS                                            ▶ STEP 3
# ============================================================================
# Here a tool is just a normal Python function. LangChain reads three things
# from it to build the description the model sees (like @tool in pythonLG.py):
#   the function's NAME, its ARGUMENTS with their types (a: int), and its DOCSTRING.
# So the docstring isn't just a comment here: the model reads it.

# CHANGED: a model that can SEE images. gpt-4o in the course.
# qwen3 and qwen2.5 are text-only, so we downloaded a small Qwen that can see
# images (`ollama pull qwen2.5vl:3b`, 3.2 GB). "vl" = vision-language.
VISION_MODEL = "qwen2.5vl:3b"
vision_llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model=VISION_MODEL)


# --- Tool 1: extract_text ----------------------------------------------------
# Reads an image file and asks the vision model to write down all its text.
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # 1. Read the image file as raw bytes ("rb" = read binary).
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        # 2. Models receive text, not files. base64 turns the bytes into a
        #    (very long) string of letters and digits that can travel as text.
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 3. One message with TWO parts: an instruction, and the image itself.
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # 4. Ask the vision model. Note: a tool can call ANOTHER model inside it.
        response = vision_llm.invoke(message)

        all_text += response.content + "\n\n"
        return all_text.strip()   # .strip() removes the blank lines at the end
    except Exception as e:
        # If anything goes wrong (no file, no vision model...), don't crash:
        # print the problem and return an empty text.
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""


# --- Tool 2: divide ----------------------------------------------------------
# A calculator. Small models are bad at arithmetic, so a tool is safer.
def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


# The tool list (lupin.py: TOOLS).
tools = [
    divide,
    extract_text
]


# ============================================================================
#  PART 4: THE MODEL (with the tools attached)                  ▶ STEP 4
# ============================================================================
# CHANGED: your local Qwen instead of gpt-4o.
llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen3:1.7b")

# bind_tools: attach the tool list once, so every call sends it (pythonLG.py did the same).
# parallel_tool_calls=False: "one tool call at a time, please".
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# ============================================================================
#  PART 5: THE NODE "assistant"                                 ▶ STEP 5 (stored)
# ============================================================================
# The only node we write ourselves. It runs at STEP 7, every time round the loop.
# (pythonLG.py: ask_model)
def assistant(state: AgentState):
    # A plain-text list of the tools, pasted into the system prompt.
    # (The model also receives the real tool descriptions via bind_tools;
    # the course adds this text on top, to make it extra clear.)
    textual_description_of_tool="""
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    # Which image is loaded (from the form). None if there isn't one.
    image=state["input_file"]

    # The system prompt: who Alfred is + the tools + the current image.
    # It's rebuilt on every call, and NOT stored in the messages.
    sys_msg = SystemMessage(content=f"You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    # Ask the model: the system prompt + the whole conversation so far.
    # Its reply is either an answer or a tool request. Returning it ADDS it to
    # the messages (thanks to add_messages in Part 2).
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }


# ============================================================================
#  PART 6: THE GRAPH (with the loop)                            ▶ STEP 5
# ============================================================================
builder = StateGraph(AgentState)

# The boxes.
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))   # ready-made box: runs whatever tool the model asked for

# The arrows.
builder.add_edge(START, "assistant")         # always start by asking the model

# The fork, after the assistant: tools_condition looks at the model's last
# message. Tool call -> go to "tools". No tool call -> go to END.
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

# THE LOOP: after running a tool, go back to the assistant, so the model sees
# the result and decides what's next. Remove this line and it's no longer an agent.
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

# CHANGED: the course shows a picture of the graph with display(), which only
# works in notebooks. To save it as an image instead, remove the # below.
# (It needs internet: the picture is drawn by an online service, mermaid.ink.)
# with open("alfred_documents_graph.png", "wb") as f:
#     f.write(react_graph.get_graph(xray=True).draw_mermaid_png())


# ============================================================================
#  PART 7: RUN IT                                               ▶ STEP 6
# ============================================================================

# --- Example 1: a simple calculation ----------------------------------------
# The question goes in as the first message. No image: input_file is None.
print("\n=== Example 1: Divide 6790 by 5 ===")
messages = [HumanMessage(content="Divide 6790 by 5")]

# invoke() STARTS the graph (STEPS 7 to 9, looping), and returns the final
# form when it reaches END.
#
# NOT from the course: config={"callbacks": [log]} plugs in our run log, so
# every model call and tool use is saved in runs/<date_time>.txt.
log = RunLog()
messages = react_graph.invoke({"messages": messages, "input_file": None}, config={"callbacks": [log]})

# ▶ STEP 10: print every message of the conversation, nicely formatted.
# You should see: your question -> a divide tool call -> its result -> Alfred's answer.
for m in messages['messages']:
    m.pretty_print()


# --- Example 2: reading Mr. Wayne's note (an image) -------------------------
# CHANGED: only runs if the image is in this folder AND you set up a vision
# model in Part 3. The image comes from the course's notebook.
IMAGE = "Batman_training_and_meals.png"
if os.path.exists(IMAGE):
    print("\n=== Example 2: Mr. Wayne's note ===")
    messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?")]
    messages = react_graph.invoke({"messages": messages, "input_file": IMAGE}, config={"callbacks": [log]})
    for m in messages['messages']:
        m.pretty_print()
else:
    print(f"\n(Example 2 skipped: put {IMAGE} in this folder to run it.)")
