import os
import openai
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    ServiceContext,
    StorageContext
)

os.environ["OPENAI_API_BASE"] = "https://tanzubot.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = ""
# Note the model deployment names are the names of your azure deployments, not the names of the models. The model_deployment_name values on the following 2 lines what I named my azure model deployments.
os.environ["AZURE_QUERY_MODEL_DEPLOYMENT_NAME"] = "gpt-35-turbo"
os.environ["AZURE_EMBEDDINGS_MODEL_DEPLOYMENT_NAME"] = "text-embedding-ada-002"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_query_deployment = os.getenv("AZURE_QUERY_MODEL_DEPLOYMENT_NAME")
openai.azure_embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_MODEL_DEPLOYMENT_NAME")


llm = AzureChatOpenAI(
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    deployment_name=openai.azure_query_deployment,
    openai_api_key=openai.api_key,
    openai_api_type=openai.api_type,
)

# llm = AzureOpenAI(deployment_name=openai.azure_query_deployment, model_kwargs={
#     "api_key": openai.api_key,
#     "api_base": openai.api_base,
#     "api_type": openai.api_type,
#     "api_version": openai.api_version,
# })
llm_predictor = LLMPredictor(llm=llm)

# embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
#     document_model_name=openai.azure_embeddings_deployment,
#     query_model_name=openai.azure_query_deployment
# ))

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment=openai.azure_embeddings_deployment,
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)


documents = SimpleDirectoryReader('html_downloads').load_data()

parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm
)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.set_index_id("combined_index")
index.storage_context.persist()
