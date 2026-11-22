# ============================================================================
#  tools.py: Alfred's other tools (course page "Building and Integrating Tools")
# ============================================================================
#   STEP 3  Web search      (DuckDuckGo, ready-made by smolagents)
#   STEP 4  Weather         (a FAKE weather API: picks a random weather)
#   STEP 5  Hub stats       (a REAL API: asks Hugging Face for download counts)
#
# In the categories we used for Lupin's tools/ folder:
#   web search + hub stats = API tools (the work happens on a web service)
#   weather                = native tool (pure Python, nothing leaves your Mac)
#
# Course code, unchanged.
# ============================================================================

import random

from huggingface_hub import list_models
from smolagents import DuckDuckGoSearchTool, Tool   # app.py imports DuckDuckGoSearchTool from here


# ----------------------------------------------------------------------------
#  STEP 4: weather (dummy)
# ----------------------------------------------------------------------------
class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# ----------------------------------------------------------------------------
#  STEP 5: Hugging Face Hub stats (real)
# ----------------------------------------------------------------------------
class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


# Run this file alone to test the 3 tools WITHOUT any model:
#   python3 tools.py
if __name__ == "__main__":
    print(DuckDuckGoSearchTool()("Who's the current President of France?")[:500])
    print(WeatherInfoTool()("Paris"))
    print(HubStatsTool()("facebook"))
