import os
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():

	web_urls = ['https://www.ceasefire.in/']
	pdf_paths = ['/Users/ashutoshbele/Desktop/claude/fire_ratings.pdf','/Users/ashutoshbele/Desktop/claude/WHY.pdf']

	data = []

	for url in web_urls:
		loader = WebBaseLoader(url)
		data.extend(loader.load())

	for pdf in pdf_paths:
		loader = PyPDFLoader(pdf)
		data.extend(loader.load())

	# Split the load data

	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
	docs = text_splitter.split_documents(data)

	# Create Ollama Embeddings
	ollama_embeddings = OllamaEmbeddings(model='llama2')

	# Create a Chroma vector database from the docs

	ABS_PATH = os.path.dirname(os.path.abspath(__file__))
	DB_DIR = os.path.join(ABS_PATH, "db")

	vectordb =Chroma.from_documents(documents=docs, embedding=ollama_embeddings, persist_directory=DB_DIR)
	vectordb.persist()


	# Create a retriever from the Chroma vector db

	retriever = vectordb.as_retriever(search_kwargs={"k":3})

	# Use LLm

	llm = Ollama(model="llama2")

	# create a Retriveal QA

	qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
	while(True):
		print("Enter prompt")
		prompt = input()
		if "exit" == prompt:
			break
		response = qa(prompt)
		print(response)

if __name__ == "__main__":
	main()