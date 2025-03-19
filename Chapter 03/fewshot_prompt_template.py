from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

# Define a template for each example
example_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}\n"
)

# Define some examples for the few-shot prompt
examples = [
    {"question": "Who was the first President of the United States?", "answer": "George Washington."},
    {"question": "What year did World War II begin?", "answer": "1939."}
]

# Define the prompt with a few examples and placeholders for dynamic input
few_shot_template = FewShotPromptTemplate(
    examples=examples,  # Few-shot examples
    example_prompt=example_template,  # Format for the examples
    prefix="Here are some historical facts:\n\n",  # Prefix text before examples
    suffix="Q: {user_question}\nA:",  # Suffix text where the model generates the answer
    input_variables=["user_question"]  # The dynamic part (user's question)
)

template = few_shot_template.invoke({"user_question": "Who wrote the Declaration of Independence?"})
print(template)