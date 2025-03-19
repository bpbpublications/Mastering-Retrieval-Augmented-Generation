from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)


# Define your desired data structure.
class WeatherReport(BaseModel):
    city: str = Field(description="Name of the city")
    temperature: str = Field(description="Temperature in the city")
    
    # Custom validation to ensure city name is not empty
    @validator("city")
    def city_name_not_empty(cls, field):
        if not field.strip():
            raise ValueError("City name cannot be empty!")
        return field

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=WeatherReport)

prompt = PromptTemplate(
    template="Provide a weather report.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Example query asking for the weather in a specific city.
prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "What's the weather like in New York?"})
print(parser.invoke(output))
