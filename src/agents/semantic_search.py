"""An agent designed to hold a conversation in addition to choosing tools using semantic search."""

from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate
from prompts.dynamic import DynamicPromptTemplate
from typing import Dict, List, Optional, Sequence, Union, Callable
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX


class DynamicConversationalAgent(ConversationalAgent):
    """An agent that holds a conversation in addition to dynamically choosing examples."""

    @classmethod
    def create_prompt(
            cls,
            tools: Sequence[BaseTool],
            prefix: str = PREFIX,
            suffix: str = SUFFIX,
            format_instructions: str = FORMAT_INSTRUCTIONS,
            ai_prefix: str = "AI",
            human_prefix: str = "Human",
            input_variables: Optional[List[str]] = None,
            partial_variables: Optional[Dict[str, Union[str, Callable]]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero-shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            format_instructions: String describing how the AI should describe tool usage.
            ai_prefix: String to use before AI output.
            human_prefix: String to use before human output.
            input_variables: List of input variables the final prompt will expect.
            partial_variables: Dictionary with partial variables for the prompt.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names,
                                                         ai_prefix=ai_prefix,
                                                         human_prefix=human_prefix)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return DynamicPromptTemplate(template=template,
                                     input_variables=input_variables,
                                     partial_variables=partial_variables)


if __name__ == '__main__':
    from langchain.agents import load_tools


    def superguay(**kwargs):
        print('asdasd')


    tools = load_tools([])
    p = DynamicConversationalAgent.create_prompt(tools=tools,
                                                 prefix='THIS IS THE PREFIX\n\n{examples}',
                                                 suffix='THIS IS THE SUFFIX\n{input}',
                                                 input_variables=['input'],
                                                 partial_variables={'examples': superguay})

    p.format(input='What is my name?')
