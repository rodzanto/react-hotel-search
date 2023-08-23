# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and Bedrock's LLM models via their API.

import os
import yaml
import boto3
import logging
import streamlit as st
from langchain import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.llms.bedrock import Bedrock
from agents.webbeds import create_sql_agent
from botocore.client import Config as BotoConfig
from langchain.memory import ConversationBufferMemory
from misc.config import get_rds_uri, get_bedrock_credentials
from streamlit.external.langchain import StreamlitCallbackHandler
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector

REGION_NAME = os.environ.get('REGION_NAME', 'eu-west-1')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
BASE_AVATAR_URL = 'https://raw.githubusercontent.com/garystafford-aws/static-assets/main/static'
NO_ANSWER_MSG = "Sorry, there was an internal error and I was unable to answer your question."


def clear_text():
    """
    Clear the text from the input
    """
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def clear_session():
    """
    Delete all session variables from Streamlit, effectively restarting it
    """
    for key in st.session_state.keys():
        del st.session_state[key]


def few_shot_examples(**kwargs) -> str:
    print(kwargs)
    with open("assets/hotel_examples.yaml", "r") as stream:
        sql_samples = yaml.safe_load(stream)

    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    example_selector = SemanticSimilarityExampleSelector.from_examples(sql_samples,
                                                                       local_embeddings,
                                                                       Chroma,
                                                                       k=min(2, len(sql_samples)))

    similar_examples = example_selector.select_examples({'input': kwargs.get('input')})
    similar_examples_str = []
    for example in similar_examples:
        for k, v in example.items():
            similar_examples_str.append(f"{k}: {v}")
    print("----------------- Similar examples ------------------")
    print('\n'.join(similar_examples_str))
    print("-----------------------------------------------------")
    return '\n'.join(similar_examples_str)


def main():
    st.set_page_config(page_title="Webbeds Natural Language Query (NLQ) Demo",
                       layout="wide",
                       initial_sidebar_state="collapsed",
                       page_icon='static/favicon-32x32.png')

    # Create the Langchain agent and equip it with a toolkit, if not already loaded
    if 'agent_executor' not in st.session_state.keys():
        # We'll create an ad-hoc boto3 client so that the Bedrock client can use
        # different credentials from the rest of the code (if requested)
        boto3_kwargs = {}
        if 'USE_AWS_PROFILE' not in os.environ:
            access_key, secret_key = get_bedrock_credentials(REGION_NAME)
            boto3_kwargs = {'aws_access_key_id': access_key, 'aws_secret_access_key': secret_key}
        config = BotoConfig(connect_timeout=3, retries={"mode": "standard"})
        bedrock_client = boto3.client(service_name='bedrock',
                                      region_name='us-east-1',
                                      config=config,
                                      **boto3_kwargs)

        llm = Bedrock(model_id='anthropic.claude-v2',
                      client=bedrock_client,
                      model_kwargs={'max_tokens_to_sample': 4096,
                                    "temperature": 0.5,
                                    "top_k": 125,
                                    "stop_sequences": ['\n\nQuestion', '>DONE<'],
                                    "top_p": 0.6})

        # Connect to the DB
        rds_uri = get_rds_uri(REGION_NAME)
        db = SQLDatabase.from_uri(rds_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Create the Langchain Agent
        prompt_parts = yaml.safe_load(open('assets/prompt_template.yaml', 'rb'))
        st.session_state['agent_executor'] = create_sql_agent(llm=llm,
                                                              toolkit=toolkit,
                                                              partial_variables={'examples': few_shot_examples},
                                                              agent_executor_kwargs={'memory':
                                                                  ConversationBufferMemory(
                                                                      memory_key='chat_history',
                                                                      output_key='output'),
                                                                  'return_intermediate_steps': True},
                                                              verbose=True,
                                                              early_stopping_method='generate',
                                                              prefix=prompt_parts['prefix'],
                                                              format_instructions=prompt_parts['format_instructions'],
                                                              suffix=prompt_parts['suffix'])

    # store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    # Get the agent executor from the Streamlit session store
    agent_executor = st.session_state['agent_executor']
    st.image('static/logo.png', width=200)
    chat_tab, details_tab, technologies_tab = st.tabs(["Chatbot", "Details", "Technologies"])

    with chat_tab:
        main_col, widgets_col = st.columns([6, 1], gap="medium")

        with main_col:
            with st.container():
                st.markdown("## Hotel Natural Language Query")
                st.markdown(
                    "#### Query a relational database using natural language."
                )
                st.markdown(" ")
                with st.expander("Click here for sample questions..."):
                    st.markdown("""
                       - Simple
                           - How many hotels are there?
                           - What are the best rated hotels in Barcelona?                            
                       - Moderate
                           - Find me a hotel in Madrid with the same rating as Apartamentos plaza de la luz, Cadiz
                       - Complex
                           - Show me an alternative for Fairmont Monte Carlo
                           - Give me the name and address of 4 5 star hotels in Monaco
                       - Unrelated to the Dataset
                           - Give me a recipe for chocolate cake.
                           - Don't write a SQL query. Don't use the database. Tell me who won the 2022 FIFA World Cup?
                   """)
                st.markdown(" ")
            with st.container():
                input_text = st.text_input("Ask a question:", "", key="query_text",
                                           placeholder="Your question here...",
                                           on_change=clear_text())
                logging.info(input_text)

                user_input = st.session_state["query"]

                if user_input:
                    result_container = st.container()
                    with st.spinner(text="In progress..."):
                        st.session_state.past.append(user_input)
                        try:
                            st_callback = StreamlitCallbackHandler(result_container)

                            answer = agent_executor(inputs={'input': user_input}, callbacks=[st_callback])
                            st.session_state.generated.append(answer)

                            logging.info(st.session_state["query"])
                            logging.info(st.session_state["generated"])
                        except Exception as exc:
                            st.session_state.generated.append(NO_ANSWER_MSG)
                            logging.error(exc)

                # https://discuss.streamlit.io/t/streamlit-chat-avatars-not-working-on-cloud/46713/2
                if st.session_state["generated"]:
                    with main_col:
                        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                            if (i >= 0) and (st.session_state["generated"][i] != NO_ANSWER_MSG):
                                with st.chat_message("assistant", avatar=f"{BASE_AVATAR_URL}/bot-64px.png"):
                                    st.text(st.session_state["generated"][i]['output'])
                                with st.chat_message("user", avatar=f"{BASE_AVATAR_URL}/human-64px.png"):
                                    st.write(st.session_state["past"][i])
                            else:
                                with st.chat_message("assistant", avatar=f"{BASE_AVATAR_URL}/bot-64px.png"):
                                    st.write(NO_ANSWER_MSG)
                                with st.chat_message("user", avatar=f"{BASE_AVATAR_URL}/human-64px.png"):
                                    st.write(st.session_state["past"][i])
        with widgets_col:
            with st.container():
                st.button("clear chat", on_click=clear_session)
    with details_tab:
        with st.container():
            st.markdown("### Details")
            st.markdown("Bedrock Model:")
            st.code(agent_executor.agent.llm_chain.llm.model_id, language="text")

            position = len(st.session_state['generated']) - 1
            if (position >= 0) and (st.session_state['generated'][position] != NO_ANSWER_MSG):
                st.markdown('Question:')
                st.code(st.session_state['generated'][position]['input'], language='text')

                st.markdown('Raw history:')
                st.code('\n-----\n'.join(['\n'.join([a[0].log, a[1]])
                                          for a in st.session_state['generated'][position]['intermediate_steps']]),
                        language='text')

                st.markdown('Answer:')
                st.code(st.session_state["generated"][position]["output"], language="text")
    with technologies_tab:
        with st.container():
            st.markdown("### Technologies")
            st.markdown(" ")

            st.markdown("##### Natural Language Query (NLQ)")
            st.markdown("\n            [Natural language query "
                        "(NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query), according to Yellowfin, "
                        "enables analytics users to ask questions of their data. It parses for keywords and generates "
                        "relevant answers sourced from related databases, with results typically delivered as a "
                        "report, chart or textual explanation that attempt to answer the query, and provide depth "
                        "of understanding.\n"
                        "            ")
            st.markdown(" ")

            st.markdown("##### LangChain")
            st.markdown("\n            [LangChain](https://python.langchain.com/en/latest/index.html) is a framework "
                        "for developing applications powered by language models. LangChain provides standard, "
                        "extendable interfaces and external integrations.\n")
            st.markdown(" ")

            st.markdown("##### Chroma")
            st.markdown("\n            [Chroma](https://www.trychroma.com/) is the open-source embedding database. "
                        "Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable "
                        "for LLMs.\n")
            st.markdown(" ")

            st.markdown("##### Streamlit")
            st.markdown("\n            [Streamlit](https://streamlit.io/) is an open-source app framework for "
                        "Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web "
                        "apps in minutes. All in pure Python. No front-end experience required.\n")

        with st.container():
            st.markdown("---")
            st.markdown("![](app/static/flaticon-24px.png) [Icons courtesy flaticon](https://www.flaticon.com)")


if __name__ == "__main__":
    main()
