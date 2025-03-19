from langchain_openai import OpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

# Streamed output with partial JSON:
print(list(json_chain.stream({"question": "Who discovered penicillin?"})))

# Expected Output:
# [{}, {'answer': ''}, {'answer': 'Al'}, {'answer': 'Alexa'}, {'answer': 'Alexander'}, {'answer': 'Alexander F'}, {'answer': 'Alexander Fleming'}]
