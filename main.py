from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

os.environ['OPENAI_API_KEY'] = 'paste your api... '

## Load Data
loader = DirectoryLoader ('PakStudy/', glob='**/*.txt')
documents = loader.load ()

## Split data to feed it to the NLTK for deep learning....... 
text_splitter = CharacterTextSplitter (chunk_size = 1000, chunk_overlap = 0)
texts = text_splitter.split_documents (documents)

## Embedding
embeddings = OpenAIEmbeddings (openai_api_key=os.environ['OPENAI_API_KEY'])

# Feed data
docsearch = Chroma.from_documents(texts, embeddings)

# Boom Actual Operation
qa = VectorDBQA.from_chain_type (llm = OpenAI(), chain_type = 'stuff' , vectorstore = docsearch)

# It needs some improvement, and testing according to your need........
query = " What is total area of Pakistan ?"
qa.run (query)

#
# Coded by Muhammad Aadil
# muhammadaadil150@gmail.com
