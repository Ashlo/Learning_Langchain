
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")


docs = loader.load()

from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()

print(embeddings)
