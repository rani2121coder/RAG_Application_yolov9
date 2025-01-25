import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


PINECONE_INDEX_NAME = "first-rag-project"

os.environ['PINECONE_API_KEY'] = "pcsk_C6GD6_45BULNVj9MCbrqmtzZMz71vR3aRAiszubvhfNeQMZrmN9DFgAyWqAia7omEwfCz"

st.title("YOLOV9 Q&A Application")

loader = PyPDFLoader(r"C:\Users\Sandhya\Downloads\yolov9_paper.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
# Set the GOOGLE_API_KEY environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyC8DiKrgxZYF48fwfaI7GeJ6_8xBFhdIcQ"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME,embeddings)
retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

query = st.chat_input("Say something: ")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you din't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
  )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])