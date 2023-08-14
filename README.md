# Hotel Search using ReAct Prompting with Amazon Bedrock

## Overview
This example implements a Reasoning & Acting ([ReAct](https://www.promptingguide.ai/techniques/react)) demonstration.

We use an [Amazon Bedrock](https://aws.amazon.com/bedrock/) Large Language Model (LLM) with a [Langchain tool](https://python.langchain.com/docs/modules/agents/tools/) for integrating to a Postgres SQL databse in [Amazon RDS](https://aws.amazon.com/rds/).

![example](./example.png)

## Architecture and flow
![arq](./pics/nlq-arq.png)


## Instructions
- Download dependencies (boto3 and botocore SDK packages) into `src/dependencies`

Repo forked from [AWS Solutions Library repo](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws)

[AWS Solution NLQ website](https://aws.amazon.com/solutions/guidance/natural-language-queries-of-relational-databases-on-aws/)

## Running locally

You can run a development environment locally using Docker. First, make sure
to copy the botocore & boto3 whl files from the BedRock SDK into
[`dependencies`](src/dependencies), then start the environment by running:

```bash
docker compose up
```

You can further modify the behaviour of the application by setting environment
variables in [`docker-compose.yml`](docker-compose.yml). Of particular
interest is:

* `USE_AWS_PROFILE`: If present, the code will honour the normal `AWS_ACCESS_KEY_ID` &
  `AWS_SECRET_ACCESS_KEY` and related variables for the BedRock client. It will otherwise
  look for the credentials in AWS Secrets manager.
