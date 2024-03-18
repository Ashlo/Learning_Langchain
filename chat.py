from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

print("Import Success")

def initialize():
    prompt = ChatPromptTemplate(
    messages = [SystemMessagePromptTemplate.from_template(
        "Output as if you are a nice chatbot having a conversation with a human do not show the output of the whole conversation just the chatbot part. I'm looking to buy books, i am looking for life philosophy I'm deciding between two options:\n" \
          "1. self help books /s.\n" \
          "2. or books like kafka and doestoevsky/.\n" \
          "Can you help me decide which one is the better choice for me? "
    ),
    MessagesPlaceholder(variable_name='chat_history'),
    HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    callbacks = [StreamingStdOutCallbackHandler()]

    llm = Ollama(model='llama2')
    memory = ConversationBufferMemory(memory_key = "chat_history",return_messages=True)
    llm_chain = LLMChain(prompt=prompt,llm=llm,memory=memory,verbose=False)

    return memory, llm_chain


def chat(question, memory,llm_chain):
    answer = llm_chain({"question":question,"chat_history":memory})
    return answer

if __name__=="__main__":
    print("Chatbot: Hello ! Type exit to end")
    memory,llm_chain =initialize()

    while True:
        user_input=input("User: ")
        if user_input.lower() == "exit":
            break
        response = chat(user_input, memory, llm_chain)
        print(f"Chatbot: {response["text"]}")
