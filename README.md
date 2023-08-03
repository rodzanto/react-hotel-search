# Hotel Search using ReAct Prompting with Amazon Bedrock

### Overview
This example implements a Reasoning & Acting ([ReAct](https://www.promptingguide.ai/techniques/react)) demonstration.

We use an [Amazon Bedrock](https://aws.amazon.com/bedrock/) Large Language Model (LLM) with a [Langchain tool](https://python.langchain.com/docs/modules/agents/tools/) for integrating to a Postgres SQL databse in [Amazon RDS](https://aws.amazon.com/rds/).

![example](./example.png)

### Architecture and flow
![arq](./pics/nlq-arq.png)


### Instructions
- Download dependencies (boto3 and botocore SDK packages) into `docker/dependencies`

Repo forked from [AWS Solutions Library repo](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws)

[AWS Solution NLQ website](https://aws.amazon.com/solutions/guidance/natural-language-queries-of-relational-databases-on-aws/)
