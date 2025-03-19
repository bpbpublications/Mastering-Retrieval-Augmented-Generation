from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Function to retrieve the session history from the SQLite database
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

# Initialize the Chat Model and the runnable with history
model = ChatOpenAI(model="gpt-4o-mini")
runnable_with_history = RunnableWithMessageHistory(
    model,
    get_session_history,
)

# Start the conversation with the first user input
runnable_with_history.invoke(
    [HumanMessage(content="hi - im bob!")],
    config={"configurable": {"session_id": "1"}},
)

# Continue the conversation with the same session_id
runnable_with_history.invoke(
    [HumanMessage(content="Can you recommend a good restaurant?")],
    config={"configurable": {"session_id": "1"}},
)

# Further interactions in the same session
response = runnable_with_history.invoke(
    [HumanMessage(content="What’s the weather like today and do you remember my name?")],
    config={"configurable": {"session_id": "1"}},
)

print(response.content)
