import os
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():

	web_urls = ['www.caesefire.in']
	pdf_paths = ['demo1.pdf', demo2.pdf]

	data = []

	for url in web_urls:
		loader = WebBaseLoader(url)
		data.extend(loader.load())

	for pdf in pdf_paths:
		loader = PyPDFLoader(pdf)
		data.extend(loader.load())