from langchain_community.llms import Ollama

llm = Ollama(model='llama2')
#response=llm.invoke('How langchain works and how can it help with buildling SaaS project')
#print(response)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

prompt=ChatPromptTemplate.from_messages([('system',"You are world class technical documentation writer"),
                                        ("user","{input}")])


chain = prompt | llm | output_parser

response = chain.invoke({"input":"How can langsmith help with testing"})
print(response)
