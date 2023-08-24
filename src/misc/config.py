import os
import json
import boto3
import logging
from botocore.exceptions import ClientError


def get_rds_uri(region_name):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    if 'DB_URI' in os.environ:
        return os.getenv('DB_URI')

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        secret = client.get_secret_value(SecretId="/nlq/RDS_URI")
        secret = json.loads(secret["SecretString"])
        rds_endpoint = secret["RDSDBInstanceEndpointAddress"]
        rds_port = secret["RDSDBInstanceEndpointPort"]
        rds_db_name = secret["NLQAppDatabaseName"]

        secret = client.get_secret_value(SecretId="/nlq/MasterUsername")
        rds_username = secret["SecretString"]

        secret = client.get_secret_value(SecretId="/nlq/MasterUserPassword")
        rds_password = secret["SecretString"]
    except ClientError as e:
        logging.error(e)
        raise e

    return f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"


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
