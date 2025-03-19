from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

class WeatherToolException(Exception):
    """Custom exception for handling weather tool errors."""

    def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception


def weather_tool_exception_handler(msg: AIMessage, config: RunnableConfig) -> Runnable:
    try:
        return weather_forecast_tool.invoke(msg.tool_calls[0]["args"], config=config)
    except Exception as e:
        raise WeatherToolException(msg.tool_calls[0], e)


def handle_exception_and_retry(inputs: dict) -> dict:
    exception = inputs.pop("exception")

    # Add historical messages to the original input, so the model knows what went wrong with the last tool call.
    messages = [
        AIMessage(content="", tool_calls=[exception.tool_call]),
        ToolMessage(
            tool_call_id=exception.tool_call["id"], content=str(exception.exception)
        ),
        HumanMessage(
            content="The last weather forecast tool call failed. Please retry with corrected arguments and avoid previous errors."
        ),
    ]
    inputs["last_output"] = messages
    return inputs

@tool
def weather_forecast_tool(city: str, date: str) -> str:
    """Get the weather forecast for a given city on a specific date."""
    # This is just a placeholder implementation. Replace it with actual logic or API calls.
    return f"The weather in {city} on {date} is expected to be sunny with mild temperatures."

# Create an instance of the LLM (OpenAI or any other supported model)
llm = ChatOpenAI(model="gpt-4-mini")

# Bind the weather tool to the LLM
llm_with_tools = llm.bind_tools(
    [weather_forecast_tool],
)

# Set up a prompt that allows for error handling and retries
prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("placeholder", "{last_output}")]
)
weather_chain = prompt | llm_with_tools | weather_tool_exception_handler

# If the initial weather tool call fails, retry with the exception passed as a message
self_correcting_weather_chain = weather_chain.with_fallbacks(
    [handle_exception_and_retry | weather_chain], exception_key="exception"
)
