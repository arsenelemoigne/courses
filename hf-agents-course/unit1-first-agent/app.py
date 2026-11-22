from smolagents import CodeAgent,DuckDuckGoSearchTool, LiteLLMModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

#CodeAgent is the loop
#LiteLLMModel runs the model locally (Ollama) instead of the HF API

from reasoning_model import ReasoningLiteLLMModel   # LiteLLMModel that prints hidden reasoning

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !

# v1 returned a JSON string. The model then wrote  ratio = count_tokens(a) / count_tokens(b)
# -> TypeError: str / str, and it never recovered. A tool's return type must be the thing
# the model needs to compute with. Hence -> int. One tool, one kind of return value.
@tool
def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:         #it's import to specify the return type
                                                                                #Keep this format for the description / args / args description but feel free to modify the tool
    """Counts how many tokens a piece of text becomes. Returns a plain integer you can do arithmetic with.

    Args:
        text: The text to tokenize.
        encoding_name: Tokenizer encoding, e.g. 'cl100k_base' (GPT-4) or 'o200k_base' (GPT-4o).
    """
    import tiktoken
    return len(tiktoken.get_encoding(encoding_name).encode(text))


# the human-readable half, split out into its own tool
@tool
def show_token_split(text: str, encoding_name: str = "cl100k_base") -> str:
    """Shows how a piece of text is split into tokens, as a readable list of pieces.

    Args:
        text: The text to tokenize.
        encoding_name: Tokenizer encoding, e.g. 'cl100k_base' (GPT-4) or 'o200k_base' (GPT-4o).
    """
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return " | ".join(enc.decode([i]) for i in enc.encode(text)[:40])

# a comparison tool: does the two-measurements-and-compare step in Python,
# because a 3B model reliably loses track of variables across statements
@tool
def compare_token_counts(text_a: str, text_b: str, encoding_name: str = "cl100k_base") -> str:
    """Compares how many tokens two pieces of text use and says which is cheaper.

    Args:
        text_a: The first text to compare.
        text_b: The second text to compare.
        encoding_name: Tokenizer encoding, e.g. 'cl100k_base' (GPT-4) or 'o200k_base' (GPT-4o).
    """
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    a, b = len(enc.encode(text_a)), len(enc.encode(text_b))
    if a == b:
        return f"'{text_a}' and '{text_b}' both use {a} tokens - neither is cheaper."
    cheaper, n_lo, n_hi = (text_a, a, b) if a < b else (text_b, b, a)
    return f"'{cheaper}' is cheaper: {n_lo} tokens vs {n_hi} ({n_hi - n_lo} fewer)."

# this tool is actually working 
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"



final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

# LOCAL CHANGE: was HfApiModel -> Qwen2.5-Coder-32B on HF's servers (costs inference credits).
# Now LiteLLMModel -> Ollama on this laptop. Everything else is the course's.

model = ReasoningLiteLLMModel(
max_tokens=2096,                            #untouched
temperature=0.5,                            #untouched
model_id='ollama_chat/qwen3:1.7b',          #reasoning model - slower, but shows its thinking
api_base='http://127.0.0.1:11434',          #api_base point at my computer
num_ctx=8192,
custom_role_conversions=None,
)


# Import tool from Hub
# LOCAL CHANGE: disabled - this tool calls HF inference and returns 402 without credits.
# The template never puts it in `tools` anyway, so nothing else changes.
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:               #prompts.yaml is the system prompt = Thought: / Code: / Observation and working examples
    prompt_templates = yaml.safe_load(stream)           #this agent writes Python and not JSON
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, count_tokens, show_token_split, compare_token_counts, get_current_time_in_timezone],  # add your tools here (don't remove final answer)
    max_steps=6,                                            # max_steps=6 caps the Thought→Code→Observation loop
    verbosity_level=1,                                      # verbosity_level=1 prints each step
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
    additional_authorized_imports=["json"],   # the sandbox blocks imports unless listed
)


GradioUI(agent).launch()