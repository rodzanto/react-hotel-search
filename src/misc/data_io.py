import yaml
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.sql import SQLDatabaseChain
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import _postgres_prompt, PROMPT_SUFFIX


def load_samples():
    # Load the sql examples for few-shot prompting examples
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

    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    example_selector = SemanticSimilarityExampleSelector.from_examples(examples, local_embeddings,
                                                                       Chroma, k=min(3, len(examples)))

    few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector,
                                            example_prompt=example_prompt,
                                            prefix=_postgres_prompt + "Here are some examples:",
                                            suffix=PROMPT_SUFFIX,
                                            input_variables=["table_info", "input", "top_k"])

    return SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt, use_query_checker=False,
                                     verbose=True, return_intermediate_steps=True)
