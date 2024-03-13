from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder

llm = Ollama(model='llama2')

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

llm = Ollama(model='llama2')

docs = loader.load()

from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
document_chain.invoke({
    "input": "how cam langsmith help with building SaaS",
    "context": [Document(page_content="Langsmith can help you create SaaS from Scratch")]
})

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsimth help with building SaaS"})
if response:
    print(response['answer'])

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# test

from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my SaaS applications?"), AIMessage(content="Yes!")]
response = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response)
