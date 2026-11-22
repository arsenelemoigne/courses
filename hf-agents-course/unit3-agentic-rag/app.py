# ============================================================================
#  app.py: Alfred, the gala agent (course page "Creating Your Gala Agent")
# ============================================================================
#   STEP 6  Choose the model          (same one-line switch as Lupin)
#   STEP 7  Give Alfred all 4 tools   (from retriever.py and tools.py)
#   STEP 8  Ask one of the course's example questions
#
# Course code, except STEP 6: the course uses InferenceClientModel (Hugging
# Face's paid servers). We use OpenAIServerModel instead, which talks to any
# OpenAI-compatible server, exactly like Lupin: your Qwen in Ollama, or
# DeepSeek on OpenRouter (key read from Lupin's .env file).
#
# Run:  python3 app.py
# ============================================================================

# Import necessary libraries
import os

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset


# ----------------------------------------------------------------------------
#  STEP 6: the model (change this one line to switch)
# ----------------------------------------------------------------------------
MODEL = "deepseek/deepseek-v4-flash-0731"

load_dotenv()   # loads OPENROUTER_API_KEY
LOCAL = ("http://localhost:11434/v1", "ollama")
ONLINE = ("https://openrouter.ai/api/v1", os.environ.get("OPENROUTER_API_KEY"))
WHERE = {
    "qwen3:1.7b": LOCAL,
    "deepseek/deepseek-v4-flash-0731": ONLINE,
}
BASE_URL, API_KEY = WHERE[MODEL]

model = OpenAIServerModel(model_id=MODEL, api_base=BASE_URL, api_key=API_KEY)


# ----------------------------------------------------------------------------
#  STEP 7: build Alfred with all the tools
# ----------------------------------------------------------------------------
# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools.
# A CodeAgent is smolagents' version of Lupin's loop, with one big difference:
# instead of asking for a tool in JSON, the model writes PYTHON CODE that
# calls the tools, e.g.   info = guest_info_retriever(query="Ada Lovelace")
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)


# ----------------------------------------------------------------------------
#  STEP 8: ask a question (change EXAMPLE to try another one)
# ----------------------------------------------------------------------------
EXAMPLES = {
    1: "Tell me about 'Lady Ada Lovelace'",                                                    # guest RAG
    2: "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",  # weather
    3: "One of our guests is from Qwen. What can you tell me about their most popular model?",     # hub stats
    4: "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?",  # several tools
}
EXAMPLE = 1

response = alfred.run(EXAMPLES[EXAMPLE])

print("🎩 Alfred's Response:")
print(response)
