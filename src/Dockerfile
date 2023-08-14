# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and OpenAI's LLM models via their API.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2023-07-17

FROM python:3.11.4-slim

ENV PIP_DEFAULT_TIMEOUT=100 \
    # allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

RUN set -ex
# create a non-root user
RUN groupadd --system --gid 1001 appgroup
RUN useradd --system --uid 1001 --gid 1001 --create-home appuser
# Install the dependencies for the environment
COPY requirements.txt .
COPY dependencies/ ./dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install gcc g++ git make -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r requirements.txt -U && \
    pip uninstall boto3 -y && \
    pip install -IU ./dependencies/boto3-*.whl && \
    pip uninstall botocore -y && \
    pip install -IU ./dependencies/botocore-*.whl && \
    rm -rf ./dependencies && \
    pip install watchdog

WORKDIR /home/appuser/app/

# copy required files to image
COPY --chown=appuser:appgroup static static
COPY --chown=appuser:appgroup assets assets
COPY --chown=appuser:appgroup .streamlit .streamlit
COPY --chown=appuser:appgroup app_bedrock.py app.py

EXPOSE 8501

# Set the user to run the application
USER appuser
ENTRYPOINT [ "streamlit", "run", "app.py" ]
