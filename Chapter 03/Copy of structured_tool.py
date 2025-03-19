from langchain_core.tools import StructuredTool

# Function to get current weather
def get_weather(location: str) -> str:
    """Fetch the current weather for a location."""
    # In a real-world scenario, this could call a weather API
    return f"The weather in {location} is 22°C and sunny."

# Asynchronous version to get current weather
async def async_get_weather(location: str) -> str:
    """Fetch the current weather for a location asynchronously."""
    # Simulates an async call to a weather API
    return f"The weather in {location} is 22°C and sunny (async)."

# Create a structured tool from the sync and async weather functions
weather_tool = StructuredTool.from_function(func=get_weather, coroutine=async_get_weather)

# Synchronous invocation
print(weather_tool.invoke({"location": "San Francisco"}))  
# Output: The weather in San Francisco is 22°C and sunny
