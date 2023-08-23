"""SQL agent extended to work with Webbeds data."""
from typing import Any, Dict, Optional, Sequence, List, Union, Callable
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.formatting import formatter
from langchain.agents import ConversationalAgent
from langchain.agents.agent import AgentExecutor
from prompts.dynamic import DynamicPromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, SUFFIX, PREFIX


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
    ) -> DynamicPromptTemplate:
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

    @classmethod
    def from_llm_and_tools(cls,
                           llm: BaseLanguageModel,
                           tools: Sequence[BaseTool],
                           callback_manager: Optional[BaseCallbackManager] = None,
                           output_parser: Optional[AgentOutputParser] = None,
                           prefix: str = PREFIX,
                           suffix: str = SUFFIX,
                           format_instructions: str = FORMAT_INSTRUCTIONS,
                           ai_prefix: str = "AI",
                           human_prefix: str = "Human",
                           input_variables: Optional[List[str]] = None,
                           partial_variables: Optional[Dict[str, Union[str, Callable]]] = None,
                           **kwargs: Any) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(tools,
                                   ai_prefix=ai_prefix,
                                   human_prefix=human_prefix,
                                   prefix=prefix,
                                   suffix=suffix,
                                   format_instructions=format_instructions,
                                   input_variables=input_variables,
                                   partial_variables=partial_variables)
        llm_chain = LLMChain(llm=llm,
                             prompt=prompt,
                             callback_manager=callback_manager)
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(ai_prefix=ai_prefix)
        return cls(llm_chain=llm_chain,
                   allowed_tools=tool_names,
                   ai_prefix=ai_prefix,
                   output_parser=_output_parser,
                   **kwargs)


def create_sql_agent(llm: BaseLanguageModel,
                     toolkit: SQLDatabaseToolkit,
                     callback_manager: Optional[BaseCallbackManager] = None,
                     prefix: str = SQL_PREFIX,
                     suffix: Optional[str] = SUFFIX,
                     format_instructions: str = FORMAT_INSTRUCTIONS,
                     top_k: int = 10,
                     max_iterations: Optional[int] = 15,
                     max_execution_time: Optional[float] = None,
                     early_stopping_method: str = "force",
                     verbose: bool = False,
                     agent_executor_kwargs: Optional[Dict[str, Any]] = None,
                     **kwargs: Dict[str, Any]) -> AgentExecutor:
    """Construct an SQL agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)

    agent = DynamicConversationalAgent.from_llm_and_tools(llm,
                                                          tools,
                                                          prefix=prefix,
                                                          format_instructions=format_instructions,
                                                          suffix=suffix,
                                                          **kwargs)

    return AgentExecutor.from_agent_and_tools(agent=agent,
                                              tools=tools,
                                              callback_manager=callback_manager,
                                              verbose=verbose,
                                              max_iterations=max_iterations,
                                              max_execution_time=max_execution_time,
                                              early_stopping_method=early_stopping_method,
                                              **(agent_executor_kwargs or {}))
