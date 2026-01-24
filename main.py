import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.globals import set_debug

from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.3,
    max_tokens=500
)
st.title('News Research Tool')

st.sidebar.title('News Article URLs')

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")

file_path='faiss_store_openai.pkl'

main_placeholder = st.empty()

if process_url_clicked:

    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started...")
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=50
    )

    main_placeholder.text("Text Splitter Started...")
    docs=text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    # 1) Save FAISS index to a folder
    index_folder = "faiss_store_openai"     # folder, not .pkl
    vectordb.save_local(index_folder)

    # 2) Optionally save light metadata in a pickle
    meta = {"index_path": index_folder, "n_docs": len(docs)}
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(meta, f)

    main_placeholder.text("Embedding Vector Store Built & Saved.")

query=main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("faiss_store_openai.pkl"):
        with open("faiss_store_openai.pkl", "rb") as f:
            meta = pickle.load(f)

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.load_local(
            meta["index_path"],  # "faiss_store_openai" folder
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})


        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)


        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the context to answer the question."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

        rag_chain = (
                RunnableParallel(
                    context=retriever | format_docs,
                    question=RunnablePassthrough(),
                )
                | prompt
                | llm
                | StrOutputParser()
        )

        set_debug(True)

        result= rag_chain.invoke(query)

        st.header("Answer")
        st.subheader(result)













