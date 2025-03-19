from langchain_core.prompts import PromptTemplate

# Define a prompt template with placeholders for dynamic input
prompt_template = PromptTemplate(
    input_variables=["topic"], 
    template="Explain the key features of {topic}."
)
# invoke the prompt_template
prompt_template.invoke({"topic": "python"})