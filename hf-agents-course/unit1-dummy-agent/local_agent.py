"""
Unit 1 section 12 - Dummy Agent Library, running on LOCAL Ollama.
No HF credits needed.  Run:  python local_agent.py

Uses qwen2.5:3b, which has NO reasoning channel - so the course's original
max_tokens values (200 / 150) work exactly as written.
"""
import sys
from openai import OpenAI

MODEL = "qwen2.5:3b"

# Ollama speaks the OpenAI protocol at /v1, so the client code is identical
# to the HF version - only the base_url changes.
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")

try:
    client.models.list()
except Exception as e:
    sys.exit(f"Ollama unreachable: {e}\nStart it with:  open -a Ollama")

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {{"location": "New York"}}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]


def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------- ACT 1
banner("ACT 1  -  course code, no stop  ->  it will INVENT an Observation")

act1 = client.chat.completions.create(
    model=MODEL, messages=messages, stream=False, max_tokens=200,
)
print(act1.choices[0].message.content)
print(f"\n[finish_reason: {act1.choices[0].finish_reason}]")

# ---------------------------------------------------------------- ACT 2
banner("ACT 2  -  stop=['Observation:']  ->  halts before inventing")

act2 = client.chat.completions.create(
    model=MODEL, messages=messages, max_tokens=150, stop=["Observation:"],
)
stopped = act2.choices[0].message.content
print(stopped)
print(f"\n[finish_reason: {act2.choices[0].finish_reason}  |  chars: {len(stopped)}]")

if not stopped.strip():
    sys.exit("\nstop returned empty here too.")

# ---------------------------------------------------------------- TOOL
banner("THE TOOL  -  ordinary Python the model cannot touch")


def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"


real = get_weather("London")
print(f"get_weather('London')  ->  {real!r}")

# ---------------------------------------------------------------- ACT 3
banner("ACT 3  -  real observation spliced in, generation resumed")

messages2 = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    {"role": "assistant", "content": stopped + "Observation:\n" + real},
]

act3 = client.chat.completions.create(
    model=MODEL, messages=messages2, stream=False, max_tokens=200,
)
print(act3.choices[0].message.content)

banner("Compare ACT 1's invented weather with ACT 3's real one.")
