# chatgpt.py
import os
import sys
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

# 1: add API key as environment variable
API=input('API KEY: ')
os.environ["OPENAI_API_KEY"] = API

# 2: use the OpenAI LLM (with the `model_name` optionally)
llm = OpenAI(model_name="text-davinci-003")

# 3: load your data document
data=input('Directory to the Custom Data: ')
loader = TextLoader(data)

# 4: create an index
index = VectorstoreIndexCreator().from_loaders([loader])


def query():
    # 5: get prompt specified via command line
    while True:
        query=input('Enter Query: ')
        # 6: query the index
        # 7: output the response
        print(index.query(query, llm=llm))
        if query=='quit':
            break

query()
