# ============================================================================
#  RUN LOG: records every step of a LangGraph run in a text file
# ============================================================================
# Same idea as Lupin's view.py, for the HF course files. NOT part of the agent.
#
# How it works: LangChain lets you plug in a "callback handler", an object
# whose methods LangChain calls automatically at each step:
#   on_chat_model_start  -> right before a model is called (we save what it receives)
#   on_llm_end           -> right after it answers       (we save its reply)
#   on_tool_start        -> right before a tool runs     (we save its input)
#   on_tool_end          -> right after                  (we save its result)
# It even catches the model calls hidden INSIDE a tool (like the vision model
# in extract_text).
# This is exactly how Langfuse works (Step 6 of the previous lesson), except
# that we write to a local file instead of a website.
#
# Use:   log = RunLog()                                       -> creates runs/<date_time>.txt
#        graph.invoke(..., config={"callbacks": [log]})       -> records that run
# ============================================================================

import json
import os
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler


class RunLog(BaseCallbackHandler):

    def __init__(self):
        # One file per run of the script, in runs/, named after the date and time.
        os.makedirs("runs", exist_ok=True)
        self.path = datetime.now().strftime("runs/%Y-%m-%d_%H-%M-%S.txt")
        self.model_calls = 0
        print(f"(Every step of this run is saved in {self.path})")

    # ---- Called by LangChain at each step ----------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # `messages` = everything the model is about to receive.
        self.model_calls += 1
        params = kwargs.get("invocation_params", {})
        model = params.get("model") or params.get("model_name") or "?"
        tools = [t["function"]["name"] for t in params.get("tools", [])]
        text = ""
        if tools:
            text += f"(tools offered: {', '.join(tools)})\n\n"
        for m in messages[0]:
            text += self._format(m) + "\n\n"
        self._write(f"MODEL CALL {self.model_calls} ({model}): EVERYTHING IT RECEIVES", text)

    def on_llm_end(self, response, **kwargs):
        reply = response.generations[0][0].message
        self._write(f"MODEL CALL {self.model_calls}: ITS REPLY", self._format(reply))

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        self._write(f"TOOL {name}: STARTS WITH", input_str)

    def on_tool_end(self, output, **kwargs):
        # `output` is a ToolMessage; its .content is the tool's result.
        self._write("TOOL: RESULT", str(getattr(output, "content", output)))

    # ---- Helpers ------------------------------------------------------------

    def _format(self, m):
        # One message as readable text: [role] then its content.
        text = f"[{m.type}]\n"
        if isinstance(m.content, str):
            text += m.content
        else:
            # A message with several parts (e.g. text + an image).
            for part in m.content:
                if part.get("type") == "text":
                    text += part["text"] + "\n"
                elif part.get("type") == "image_url":
                    # The image travels as a huge base64 string: show only its start.
                    url = part["image_url"]["url"]
                    text += f"<image: {url[:60]}... ({len(url)} characters)>\n"
        for call in getattr(m, "tool_calls", None) or []:
            text += f"\nTOOL CALL -> {call['name']}({json.dumps(call['args'])})"
        return text.strip()

    def _write(self, title, text):
        with open(self.path, "a") as f:
            f.write(f"\n{'#' * 70}\n#  {title}\n{'#' * 70}\n\n{text}\n")
