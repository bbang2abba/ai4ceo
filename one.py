import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."
)


def render_time_app():
    st.title("Streamlit UI")
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:16px; color:#facc15; font-size:24px;">
            최홍석
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <div style="text-align:center; margin-top:8px;">
            <div id="date" style="font-size:30px; color:#fde047; font-family:Consolas, monospace; margin-bottom:8px;"></div>
            <div id="clock" style="font-size:72px; color:#22c55e; font-family:Consolas, monospace; text-shadow:0 0 10px #22c55e, 0 0 20px #22c55e;"></div>
        </div>
        <script>
            function updateClock(){
                const now = new Date();
                const pad = n => String(n).padStart(2, '0');
                const dateStr = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`;
                const timeStr = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
                document.getElementById("date").innerText = dateStr;
                document.getElementById("clock").innerText = timeStr;
            }
            updateClock();
            setInterval(updateClock, 1000);
        </script>
        """,
        height=180,
    )


def render_chatbot_app(client: OpenAI):
    st.title("My First Chatbot")
    if "one_chatbot_messages" not in st.session_state:
        st.session_state.one_chatbot_messages = []

    for message in st.session_state.one_chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("메시지를 입력하세요...")
    if user_prompt:
        st.session_state.one_chatbot_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=st.session_state.one_chatbot_messages,
                )
                answer = response.choices[0].message.content or ""
                st.markdown(answer)

        st.session_state.one_chatbot_messages.append({"role": "assistant", "content": answer})


def render_internet_app(client: OpenAI):
    st.title("Internet Search Chatbot")
    if "one_internet_history" not in st.session_state:
        st.session_state.one_internet_history = []

    for msg in st.session_state.one_internet_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input("인터넷에서 찾아볼 내용을 입력하세요...")
    if user_input:
        st.session_state.one_internet_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        conversation_lines = []
        for message in st.session_state.one_internet_history:
            speaker = "assistant" if isinstance(message, AIMessage) else "user"
            conversation_lines.append(f"{speaker}: {message.content}")
        conversation_text = "\n".join(conversation_lines)

        with st.chat_message("assistant"):
            with st.spinner("웹 검색 중..."):
                response = client.responses.create(
                    model="gpt-5.2",
                    tools=[{"type": "web_search"}],
                    input=(
                        "아래 대화 맥락을 참고하여 최신 사용자 질문에 답변하세요.\n\n"
                        f"{conversation_text}"
                    ),
                )
                answer = response.output_text
                st.markdown(answer)

        st.session_state.one_internet_history.append(AIMessage(content=answer))


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 4

    return {"bm25": bm25_retriever, "faiss": faiss_retriever}


def retrieve_ensemble(retrievers, query: str, k: int = 6):
    bm25_docs = retrievers["bm25"].invoke(query)
    faiss_docs = retrievers["faiss"].invoke(query)
    merged = []
    seen = set()
    for doc in bm25_docs + faiss_docs:
        key = (doc.page_content.strip(), doc.metadata.get("source", ""), doc.metadata.get("page", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break
    return merged


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def render_rag_app():
    st.title("RAG Chatbot")
    if "one_rag_history" not in st.session_state:
        st.session_state.one_rag_history = []
    if "one_rag_retriever" not in st.session_state:
        st.session_state.one_rag_retriever = None

    with st.sidebar:
        st.subheader("RAG 설정")
        uploaded_files = st.file_uploader(
            "PDF 파일을 여러 개 업로드하세요.",
            type=["pdf"],
            accept_multiple_files=True,
            key="one_rag_uploader",
        )
        if st.button("RAG 처리", use_container_width=True):
            if not uploaded_files:
                st.warning("먼저 PDF 파일을 업로드해 주세요.")
            else:
                with st.spinner("PDF 처리 및 벡터 저장 중..."):
                    docs = load_pdf_docs(uploaded_files)
                    st.session_state.one_rag_retriever = build_ensemble_retriever(docs)
                st.success("RAG 처리가 완료되었습니다. 이제 질문해 주세요.")

    for msg in st.session_state.one_rag_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    question = st.chat_input("질문을 입력하세요...")
    if question:
        st.session_state.one_rag_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.one_rag_retriever is None:
            answer = "먼저 왼쪽 사이드바에서 PDF를 업로드하고 RAG 처리를 해주세요."
        else:
            with st.spinner("답변 생성 중..."):
                retrieved_docs = retrieve_ensemble(st.session_state.one_rag_retriever, question)
                context = format_docs(retrieved_docs)

                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                messages = [
                    (
                        "system",
                        f"{SYSTEM_PROMPT}\n\n아래 문맥을 우선 참고해서 답변하세요.\n\n문맥:\n{context}",
                    )
                ]
                for msg in st.session_state.one_rag_history:
                    if isinstance(msg, HumanMessage):
                        messages.append(("human", msg.content))
                    else:
                        messages.append(("ai", msg.content))
                response = llm.invoke(messages)
                answer = response.content if isinstance(response.content, str) else str(response.content)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.one_rag_history.append(AIMessage(content=answer))


load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="One App", page_icon="🧩", layout="wide")

with st.sidebar:
    st.header("통합 앱 메뉴")
    selected_app = st.radio("앱 선택", ["시간", "챗봇", "인터넷검색", "RAG"], index=0)
    if st.button("새로시작하기", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if selected_app == "시간":
    render_time_app()
else:
    if not api_key:
        st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)
    if selected_app == "챗봇":
        render_chatbot_app(openai_client)
    elif selected_app == "인터넷검색":
        render_internet_app(openai_client)
    else:
        render_rag_app()
