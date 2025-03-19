from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from collections import Counter
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

class IssueExtractionChatModel(BaseChatModel):
    """
    A custom chat model that extracts key issues from customer support conversations.

    This model is designed for use in customer support, where it processes user
    messages and identifies critical issues based on common problem keywords such as
    'error', 'not working', 'failed', etc. The model highlights these key issues for 
    support agents to prioritize.

    Example:

        .. code-block:: python

            model = IssueExtractionChatModel(keywords=["error", "not working", "failed"])
            result = model.invoke([HumanMessage(content="My application is not working.")])
            result = model.batch([[HumanMessage(content="error: 404 not found")],
                                  [HumanMessage(content="payment failed.")]])
    """

    model_name: str
    """ The name of the model."""
    keywords: List[str]
    """ A list of keywords that represent common issues in customer support conversations."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response by identifying critical issues from the conversation."""
        conversation = " ".join([msg.content for msg in messages])
        identified_issues = self._extract_issues(conversation)
        
        if identified_issues:
            response_content = "Identified Issues: " + ", ".join(identified_issues)
        else:
            response_content = "No critical issues identified."

        message = AIMessage(
            content=response_content,
            response_metadata={"processed_in": "1 second"},
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the identified issues in real-time for faster feedback."""
        conversation = " ".join([msg.content for msg in messages])
        identified_issues = self._extract_issues(conversation)

        if identified_issues:
            summary = "Identified Issues: " + ", ".join(identified_issues)
        else:
            summary = "No critical issues identified."

        for word in summary:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk

        # Send metadata at the end of the stream
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"processed_in": "1 second"})
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    def _extract_issues(self, conversation: str) -> List[str]:
        """Extract critical issues from the conversation based on keywords."""
        found_issues = set()
        conversation_lower = conversation.lower()

        for keyword in self.keywords:
            if keyword.lower() in conversation_lower:
                found_issues.add(keyword)

        return list(found_issues)

    @property
    def _llm_type(self) -> str:
        """ Return the type of language model."""
        return "issue-extraction-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """ Return identifying parameters for tracking and monitoring."""
        return {
            "model_name": self.model_name,
            "keywords": self.keywords,
        }

# Example usage
if __name__ == "__main__":
    # Create sample messages simulating a customer support chat
    messages = [
        SystemMessage(content="Please focus on identifying customer issues related to functionality."),
        HumanMessage(content="I'm facing a payment error."),
        HumanMessage(content="My app is not working after the latest update."),
        AIMessage(content="Thank you for providing the details. I will assist you with the payment error."),
    ]

    # Initialize the model with keywords that commonly represent issues
    model = IssueExtractionChatModel(model_name="SupportIssueModel", keywords=["error", "not working", "failed"])

    # Test synchronous method
    result = model.invoke(messages)
    print(result)  # Output: "Identified Issues: error, not working"

    # Test streaming method
    for chunk in model._stream(messages):
        print(chunk.message.content, end="")  # Output: "Identified Issues: e, r, r, o, r, , n, o, t, , w, o, r, k, i, n, g"
