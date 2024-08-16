import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from flask import request, session
from msrest.authentication import CognitiveServicesCredentials
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
# Load environment variables from .env file for local development
load_dotenv()

# Check if running in production or development environment
IS_DEVELOPMENT = os.getenv('IS_DEVELOPMENT', 'false').lower() == 'true'
print(f"IS_DEVELOPMENT: {IS_DEVELOPMENT}")

if not IS_DEVELOPMENT:
    print("Running in production mode.")
    # For default Azure account use only
    vectorsecret = "vectordatabsekey"
    computer_vision = "computer-vision-key-v1"
    openapi_key = "OPENAI-API-KEY"
    MySQL_db_pwd = "mysql-db-pwd"
    DB_USERNAME = "extremumadmin"
    DB_HOST = "extremum-mysql-db.mysql.database.azure.com"
    DB_NAME = "cognilink-master-dev"
    LOG_DB_NAME = "cognilink-logging-dev"
    KVUri = f"https://eavault.vault.azure.net/"

    # Neo4j connection parameters
    NEO4J_URI = "neo4j+ssc://04287b0a.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "kiR816EJfKr1bDlVRuahILH0j8CjtqQPCzOFX0cQfAw"
    OPENAI_API_VERSION = "2023-05-15"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)

    main_key = "2355a247f79f4b8ea2adaa0929cd32c2"
    vector_store = client.get_secret(vectorsecret).value
    computer_vision_key = client.get_secret(computer_vision).value
    DB_PASSWORD = client.get_secret(MySQL_db_pwd).value
else:
    print("Running in development mode.")
    # For local use only
    main_key = os.getenv("MAIN_KEY")
    vector_store = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    computer_vision_key = os.getenv("COMPUTER_VISION_SUBSCRIPTION_KEY")

    # MySQL Database Connection
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    LOG_DB_NAME = os.getenv("LOG_DB_NAME")
    OPENAI_API_VERSION = "2023-05-15"

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    print(f"main_key: {main_key}")
    print(f"vector_store: {vector_store}")
    print(f"computer_vision_key: {computer_vision_key}")

# Set OpenAI environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = main_key
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ea-openai.openai.azure.com/"

# Set MySQL DB Environment Variables
os.environ["DB_USERNAME"] = DB_USERNAME
os.environ["DB_PASSWORD"] = DB_PASSWORD
os.environ["DB_HOST"] = DB_HOST
os.environ["DB_NAME"] = DB_NAME
os.environ["LOG_DB_NAME"] = LOG_DB_NAME

os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME

# Computer Vision setup
computer_vision_api_endpoint = "https://test-computer-vision-extremum.cognitiveservices.azure.com/"
computervision_credentials = CognitiveServicesCredentials(computer_vision_key)
computervision_client = ComputerVisionClient(computer_vision_api_endpoint, computervision_credentials)

# Azure Cognitive Search setup
vector_store_address = "https://cognilink-vectordb.search.windows.net"
vector_store_password = vector_store

# Blob Storage setup
if not IS_DEVELOPMENT:
    account_name = "testcongnilink"
    container_name = "congnilink-container"
    account_url = "https://testcongnilink.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
else:
    account_name = os.getenv('ACCOUNT_NAME')
    account_key = os.getenv('ACCOUNT_KEY')
    container_name = os.getenv('CONTAINER_NAME')
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    print(f"account_name: {account_name}")
    print(f"account_key: {account_key}")
    print(f"container_name: {container_name}")

container_client = blob_service_client.get_container_client(container_name)

# Embeddings setup
embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding')

# Print a message indicating the environment
print("Running in production" if not IS_DEVELOPMENT else "Running in development")
