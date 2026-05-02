import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."
)


def load_pdf_docs(uploaded_files) -> List:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(temp_pdf_path)
    return all_docs


def build_ensemble_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 4

    return {
        "bm25": bm25_retriever,
        "faiss": faiss_retriever,
    }


def retrieve_ensemble(retrievers, query: str, k: int = 6):
    bm25_docs = retrievers["bm25"].invoke(query)
    faiss_docs = retrievers["faiss"].invoke(query)

    # BM25 + FAISS 결과를 결합하고 중복 문서를 제거합니다.
    merged = []
    seen = set()
    for doc in bm25_docs + faiss_docs:
        key = (
            doc.page_content.strip(),
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break
    return merged


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
st.title("RAG Chatbot")

if not api_key:
    st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일을 여러 개 업로드하세요.",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("RAG 처리", use_container_width=True):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드해 주세요.")
        else:
            with st.spinner("PDF 처리 및 벡터 저장 중..."):
                docs = load_pdf_docs(uploaded_files)
                st.session_state.retriever = build_ensemble_retriever(docs)
            st.success("RAG 처리가 완료되었습니다. 이제 질문해 주세요.")

for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("질문을 입력하세요...")

if question:
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.retriever is None:
        answer = "먼저 왼쪽 사이드바에서 PDF를 업로드하고 RAG 처리를 해주세요."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append(AIMessage(content=answer))
    else:
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                retrieved_docs = retrieve_ensemble(st.session_state.retriever, question)
                context = format_docs(retrieved_docs)

                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                messages = [
                    (
                        "system",
                        f"{SYSTEM_PROMPT}\n\n아래 문맥을 우선 참고해서 답변하세요.\n\n문맥:\n{context}",
                    )
                ]
                for msg in st.session_state.chat_history:
                    if isinstance(msg, HumanMessage):
                        messages.append(("human", msg.content))
                    else:
                        messages.append(("ai", msg.content))

                response = llm.invoke(messages)
                answer = response.content if isinstance(response.content, str) else str(response.content)
                st.markdown(answer)

        st.session_state.chat_history.append(AIMessage(content=answer))
