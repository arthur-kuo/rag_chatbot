# _*_ coding: utf-8 _*_
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chardet
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def embedding():
    directory_path = './resumes'
    file_paths = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-3-large"  # recommend
    # )
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv('INFERENCE_API_KEY'),
        model_name="intfloat/multilingual-e5-large"
    )
    embedding = embedding_model

    all_docs = []

    for file_path in file_paths:
        encoding = detect_encoding(file_path)
        loader = TextLoader(file_path=file_path, encoding=encoding)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)
        print(f"{file_path} embedded.")

    # db = FAISS.from_documents(all_docs, embedding)
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    return(db)


embedding()
