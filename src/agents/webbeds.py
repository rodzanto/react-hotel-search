"""SQL agent extended to work with Webbeds data."""
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.conversational.prompt import (PREFIX, FORMAT_INSTRUCTIONS, SUFFIX)
from langchain.agents.agent_toolkits.sql.prompt import (SQL_PREFIX,
                                                        SQL_SUFFIX)
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


if __name__ == '__main__':
    from langchain.sql_database import SQLDatabase
    from langchain.llms.bedrock import Bedrock
    from langchain.memory import ConversationBufferMemory

    llm = Bedrock(model_id='anthropic.claude-v2',
                  model_kwargs={'max_tokens_to_sample': 4096,
                                'temperature': 0.5,
                                'top_k': 250,
                                'stop_sequences': ['\n\nQuestion'],
                                'top_p': 1})
    db = SQLDatabase.from_uri('postgresql+psycopg2://postgres:wdbQDPDKGYHwpZdVJ4Jr@localhost:5432/wb_hotels')
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_executor = create_sql_agent(llm=llm,
                                      toolkit=toolkit,
                                      agent_executor_kwargs={'memory': memory},
                                      verbose=True,
                                      format_instructions='''To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    {ai_prefix}: [your response here]
    ```''',
                                      prefix='''Assistant is a large language model trained by Amazon.

                                      Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

                                      Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

                                      Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

                                      Assistant has access to a {dialect} database whose main table name is wb_hotels that contains information about hotels in different cities, Assistant can query it to get details about hotels.

                                      TOOLS:
                                      ------
                                      Assistant has access to the following tools:
                                      ''')

    agent_executor.run(input='How many hotels are there?')
    agent_executor.run(input="How many of those are in CADIZ?")
