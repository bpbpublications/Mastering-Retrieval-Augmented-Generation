from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class WeatherInput(BaseModel):
    location: str = Field(description="Location to get the weather forecast for")
    date: Optional[str] = Field(default=None, description="Date for the forecast, optional")


class CustomWeatherTool(BaseTool):
    name = "WeatherForecast"
    description = "Provides weather forecast for a given location"
    args_schema: Type[BaseModel] = WeatherInput
    return_direct: bool = True

    def _run(
        self, location: str, date: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Fetch the weather for the given location and date."""
        forecast = f"Sunny in {location}"  # Placeholder for actual weather API call
        return forecast if date is None else f"Sunny in {location} on {date}"

    async def _arun(
        self,
        location: str,
        date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch the weather asynchronously."""
        return self._run(location, date, run_manager=run_manager.get_sync())

# Example usage
weather_tool = CustomWeatherTool()
print(weather_tool.name)
print(weather_tool.description)
print(weather_tool.args_schema)
print(weather_tool.return_direct)

print(weather_tool.invoke({"location": "New York"}))
