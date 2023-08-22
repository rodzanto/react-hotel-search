"""SQL agent extended to work with Webbeds data."""
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, SUFFIX
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema.language_model import BaseLanguageModel
from langchain.agents.conversational.base import ConversationalAgent


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
                     **kwargs: Dict[str, Any],
                     ) -> AgentExecutor:
    """Construct an SQL agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)

    agent = ConversationalAgent.from_llm_and_tools(llm,
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
