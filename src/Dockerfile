# Install the dependencies in a separate container, so that the final
# image does not have extra packages. We need to do it this way since
# some of the dependencies do not have pre-built wheels

FROM python:3.11-slim AS BUILDENV
ENV PIP_DEFAULT_TIMEOUT=100 \
    # allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

RUN set -ex
# Install the dependencies for the environment
COPY requirements.txt .
COPY dependencies/ ./dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install gcc g++ git make rustc -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r requirements.txt -U && \
    pip uninstall boto3 -y && \
    pip install -IU ./dependencies/boto3-*.whl && \
    pip uninstall botocore -y && \
    pip install -IU ./dependencies/botocore-*.whl && \
    rm -rf ./requirements.txt ./dependencies

# Runtime image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN set -ex
# Create a non-root user for running the app
RUN groupadd --system --gid 1001 appgroup
RUN useradd --system --uid 1001 --gid 1001 --create-home appuser
# Install the dependencies for the environment
COPY --from=BUILDENV /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code to the image
WORKDIR /home/appuser/app/
COPY --chown=appuser:appgroup .streamlit .streamlit
COPY --chown=appuser:appgroup agents agents
COPY --chown=appuser:appgroup assets assets
COPY --chown=appuser:appgroup misc misc
COPY --chown=appuser:appgroup prompts prompts
COPY --chown=appuser:appgroup static static
COPY --chown=appuser:appgroup app.py app.py

EXPOSE 8501

# Set the user to run the application
USER appuser
ENTRYPOINT [ "python3.11", "-m", "streamlit", "run", "app.py" ]
