# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and OpenAI's LLM models via their API.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2023-07-17
# Application expects the following environment variables (adjust for your environment):
# export REGION_NAME="us-east-1"
# Usage: streamlit run app_openai.py --server.runOnSave true

import ast
import json
import logging
import os

import boto3
import pandas as pd
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain import (FewShotPromptTemplate, PromptTemplate, SQLDatabase)
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.prompt import (PROMPT_SUFFIX,
                                                  _postgres_prompt)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from botocore.client import Config as BotoConfig
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

REGION_NAME = os.environ.get("REGION_NAME", "eu-west-1")
MODEL_NAME = os.environ.get("MODEL_NAME", "anthropic.claude-v2")
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
BASE_AVATAR_URL = (
    "https://raw.githubusercontent.com/garystafford-aws/static-assets/main/static"
)


def main():
    st.set_page_config(
        page_title="Natural Language Query (NLQ) Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # # hide the hamburger bar menu
    # hide_streamlit_style = """
    #     <style>
    #     #MainMenu {visibility: hidden;}
    #     footer {visibility: hidden;}
    #     </style>

    # """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    NO_ANSWER_MSG = "Sorry, I was unable to answer your question."

    if 'USE_AWS_PROFILE' in os.environ:
        boto3_kwargs = {}
    else:
        access_key, secret_key = get_bedrock_credentials(REGION_NAME)
        boto3_kwargs = {'aws_access_key_id': access_key, 'aws_secret_access_key': secret_key}

    config = BotoConfig(connect_timeout=3, retries={"mode": "standard"})
    bedrock_client = boto3.client(service_name='bedrock', region_name='us-east-1', **boto3_kwargs, config=config)
    inference_modifier = {'max_tokens_to_sample': 4096,
                          "temperature": 0.5,
                          "top_k": 250,
                          "stop_sequences": ["\n\nQuestion"],
                          "top_p": 1
                          }

    # titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock)
    anthropic_llm = Bedrock(model_id=MODEL_NAME, client=bedrock_client, model_kwargs=inference_modifier)
    llm = anthropic_llm

    # define datasource uri
    rds_uri = get_rds_uri(REGION_NAME)
    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    sql_db_chain = load_few_shot_chain(llm, db, examples)
    sql_tool = Tool(
        name='Hotels DB',
        func=sql_db_chain.run,
        description="Useful for when you need to answer questions about hotels and their ratings."
    )
    tools = load_tools(
        ["llm-math"],
        llm=llm
    )
    tools.append(sql_tool)

    conversational_agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        # return_intermediate_steps=True,
        verbose=True,
        # max_iterations=1,
        memory=ConversationBufferMemory(memory_key="chat_history", input_key='input', output_key="output",
                                        return_messages=True),
    )

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

    tab1, tab2, tab3 = st.tabs(["Chatbot", "Details", "Technologies"])

    with tab1:
        col1, col2 = st.columns([6, 1], gap="medium")

        with col1:
            with st.container():
                st.markdown("## Hotel Natural Language Query")
                st.markdown(
                    "#### Query a relational database using natural language."
                )
                st.markdown(" ")
                with st.expander("Click here for sample questions..."):
                    st.markdown(
                        """
                       - Simple
                           - How many hotels are there?
                           - What are the best rated hotels in Barcelona?                            
                       - Moderate
                           - Show me 5 hotels with 4 star rating
                       - Complex
                           - TBD
                       - Unrelated to the Dataset
                           - Give me a recipe for chocolate cake.
                           - Don't write a SQL query. Don't use the database. Tell me who won the 2022 FIFA World Cup final?
                   """
                    )
                st.markdown(" ")
            with st.container():
                input_text = st.text_input(
                    "Ask a question:",
                    "",
                    key="query_text",
                    placeholder="Your question here...",
                    on_change=clear_text(),
                )
                logging.info(input_text)

                user_input = st.session_state["query"]

                if user_input:
                    with st.spinner(text="In progress..."):
                        st.session_state.past.append(user_input)
                        try:
                            output = conversational_agent({"input": user_input})
                            st.session_state.generated.append(output)
                            print("conversational_agent out:")
                            print(output)
                            output2 = sql_db_chain(user_input)
                            print("conversational_agent out:")
                            print(output2)

                            logging.info(st.session_state["query"])
                            logging.info(st.session_state["generated"])
                        except Exception as exc:
                            st.session_state.generated.append(NO_ANSWER_MSG)
                            logging.error(exc)

                # https://discuss.streamlit.io/t/streamlit-chat-avatars-not-working-on-cloud/46713/2
                if st.session_state["generated"]:
                    with col1:
                        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                            if (i >= 0) and (
                                    st.session_state["generated"][i] != NO_ANSWER_MSG
                            ):
                                with st.chat_message(
                                        "assistant",
                                        avatar=f"{BASE_AVATAR_URL}/bot-64px.png",
                                ):
                                    st.text(st.session_state["generated"][i]["result"])
                                    # st.code(st.session_state["generated"][i]["result"], language="text")
                                with st.chat_message(
                                        "user",
                                        avatar=f"{BASE_AVATAR_URL}/human-64px.png",
                                ):
                                    st.write(st.session_state["past"][i])
                            else:
                                with st.chat_message(
                                        "assistant",
                                        avatar=f"{BASE_AVATAR_URL}/bot-64px.png",
                                ):
                                    st.write(NO_ANSWER_MSG)
                                with st.chat_message(
                                        "user",
                                        avatar=f"{BASE_AVATAR_URL}/human-64px.png",
                                ):
                                    st.write(st.session_state["past"][i])
        with col2:
            with st.container():
                st.button("clear chat", on_click=clear_session)
    with tab2:
        with st.container():
            st.markdown("### Details")
            st.markdown("Bedrock Model:")
            st.code(MODEL_NAME, language="text")

            position = len(st.session_state["generated"]) - 1
            if (position >= 0) and (
                    st.session_state["generated"][position] != NO_ANSWER_MSG
            ):
                st.markdown("Question:")
                st.code(
                    st.session_state["generated"][position]["query"], language="text"
                )

                st.markdown("SQL Query:")
                st.code(
                    st.session_state["generated"][position]["intermediate_steps"][1],
                    language="sql",
                )

                st.markdown("Results:")
                st.code(
                    st.session_state["generated"][position]["intermediate_steps"][3],
                    language="python",
                )

                st.markdown("Answer:")
                st.code(
                    st.session_state["generated"][position]["result"], language="text"
                )

                data = ast.literal_eval(
                    st.session_state["generated"][position]["intermediate_steps"][3]
                )
                if len(data) > 0 and len(data[0]) > 1:
                    df = None
                    st.markdown("Pandas DataFrame:")
                    df = pd.DataFrame(data)
                    df
    with tab3:
        with st.container():
            st.markdown("### Technologies")
            st.markdown(" ")

            st.markdown("##### Natural Language Query (NLQ)")
            st.markdown(
                """
            [Natural language query (NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query), according to Yellowfin, enables analytics users to ask questions of their data. It parses for keywords and generates relevant answers sourced from related databases, with results typically delivered as a report, chart or textual explanation that attempt to answer the query, and provide depth of understanding.
            """
            )
            st.markdown(" ")

            st.markdown("##### The MoMa Collection Datasets")
            st.markdown(
                """
            [The Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection) contains over 120,000 pieces of artwork and 15,000 artists. The datasets are available on GitHub in CSV format, encoded in UTF-8. The datasets are also available in JSON. The datasets are provided to the public domain using a [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).
            """
            )
            st.markdown(" ")

            st.markdown("##### Amazon SageMaker JumpStart Foundation Models")
            st.markdown(
                """
            [Amazon SageMaker JumpStart Foundation Models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html) offers state-of-the-art foundation models for use cases such as content writing, image and code generation, question answering, copywriting, summarization, classification, information retrieval, and more.
            """
            )
            st.markdown(" ")

            st.markdown("##### LangChain")
            st.markdown(
                """
            [LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models. LangChain provides standard, extendable interfaces and external integrations.
            """
            )
            st.markdown(" ")

            st.markdown("##### Chroma")
            st.markdown(
                """
            [Chroma](https://www.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
            """
            )
            st.markdown(" ")

            st.markdown("##### Streamlit")
            st.markdown(
                """
            [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
            """
            )

        with st.container():
            st.markdown("""---""")
            st.markdown(
                "![](app/static/github-24px-blk.png) [Feature request or bug report?](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws/issues)"
            )
            st.markdown(
                "![](app/static/github-24px-blk.png) [The MoMA Collection datasets on GitHub](https://github.com/MuseumofModernArt/collection)"
            )
            st.markdown(
                "![](app/static/flaticon-24px.png) [Icons courtesy flaticon](https://www.flaticon.com)"
            )


def get_bedrock_credentials(region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        secret = client.get_secret_value(SecretId="/nlq/bedrock_credentials")
        secret = json.loads(secret["SecretString"])
        access_key = secret["access_key"]
        secret_key = secret["secret_key"]

    except ClientError as e:
        logging.error(e)
        raise e

    return access_key, secret_key


def get_rds_uri(region_name):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    if 'DB_URI' in os.environ:
        return os.getenv('DB_URI')

    rds_username = None
    rds_password = None
    rds_endpoint = None
    rds_port = None
    rds_db_name = None

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        secret = client.get_secret_value(SecretId="/nlq/RDS_URI")
        secret = json.loads(secret["SecretString"])
        rds_endpoint = secret["RDSDBInstanceEndpointAddress"]
        rds_port = secret["RDSDBInstanceEndpointPort"]
        rds_db_name = secret["NLQAppDatabaseName"]

        secret = client.get_secret_value(SecretId="/nlq/NLQAppUsername")
        rds_username = secret["SecretString"]

        secret = client.get_secret_value(SecretId="/nlq/NLQAppUserPassword")
        rds_password = secret["SecretString"]
    except ClientError as e:
        logging.error(e)
        raise e

    return f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"


def load_samples():
    # Load the sql examples for few-shot prompting examples
    sql_samples = None

    with open("assets/hotel_examples.yaml", "r") as stream:
        sql_samples = yaml.safe_load(stream)

    return sql_samples


def load_few_shot_chain(llm, db, examples):
    example_prompt = PromptTemplate(
        input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
        template=(
            "{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult:"
            " {sql_result}\nAnswer: {answer}"
        ),
    )

    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        local_embeddings,
        Chroma,
        k=min(3, len(examples)),
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_postgres_prompt + "Here are some examples:",
        suffix=PROMPT_SUFFIX,
        input_variables=["table_info", "input", "top_k"],
    )

    return SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=few_shot_prompt,
        use_query_checker=False,
        verbose=True,
        return_intermediate_steps=True,
    )


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
