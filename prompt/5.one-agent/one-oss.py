import os
import time as time_module
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

APP_TITLE = "3개 통합 앱 (AI Agent)"
LLM_MODEL = "gemma4"
EMBED_MODEL = "bge-m3"

SYSTEM_MESSAGE_RAG = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. "
    "모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."
)

TOOL_LABELS = {
    "time_tool": "시간",
    "chatbot_tool": "챗봇",
    "rag_tool": "RAG",
}

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
h1 { font-size: 1.45rem !important; font-weight: 700 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage p, .stChatMessage li { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { background-color: #ff1493 !important; }
.tool-badge {
    border-radius: 8px;
    padding: 8px 10px;
    background: #111827;
    color: #f9fafb;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "selected_tool": None,
        "tool_history": [],
        "agent_messages": [],
        "chatbot_history": [],
        "rag_history": [],
        "rag_ready": False,
        "rag_retriever": None,
        "vectorstore": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


class ToolTrackCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name")
        if not tool_name:
            return
        label = TOOL_LABELS.get(tool_name, tool_name)
        st.session_state.selected_tool = label
        st.session_state.tool_history.append(label)


@tool
def time_tool() -> str:
    """현재 시간, 현재 날짜, 시계를 물으면 사용."""
    now = datetime.now()
    return f"현재 날짜는 {now.strftime('%Y년 %m월 %d일')}이고 현재 시간은 {now.strftime('%H:%M:%S')}입니다."


@tool
def chatbot_tool(question: str) -> str:
    """일반 지식 질문이나 간단한 대화는 이 도구를 사용."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
    messages: List[Any] = [
        SystemMessage(content="친절한 한국어 어시스턴트로서 정확하고 이해하기 쉽게 답해주세요.")
    ]
    for msg in st.session_state.chatbot_history[-12:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    answer = llm.invoke(messages).content
    st.session_state.chatbot_history.append({"role": "user", "content": question})
    st.session_state.chatbot_history.append({"role": "assistant", "content": answer})
    return answer


@tool
def rag_tool(question: str) -> str:
    """문서 기반 요약/질문은 이 도구를 사용."""
    if not st.session_state.rag_ready or st.session_state.rag_retriever is None:
        return "RAG 문서 준비가 필요합니다. 왼쪽 사이드바에서 PDF를 먼저 처리해주세요."

    docs = st.session_state.rag_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs[:6])

    llm = ChatOllama(model=LLM_MODEL, temperature=0.5)
    messages: List[Any] = [SystemMessage(content=SYSTEM_MESSAGE_RAG)]
    for msg in st.session_state.rag_history[-12:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(
        HumanMessage(
            content=(
                "다음 문맥을 기반으로 질문에 답해주세요.\n\n"
                f"[문맥]\n{context}\n\n"
                f"[질문]\n{question}"
            )
        )
    )
    answer = llm.invoke(messages).content
    st.session_state.rag_history.append({"role": "user", "content": question})
    st.session_state.rag_history.append({"role": "assistant", "content": answer})
    return answer


def process_pdfs(uploaded_files: List[Any]) -> str:
    all_docs = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = tmp.name
        try:
            loaded_docs = PyPDFLoader(temp_path).load()
            for d in loaded_docs:
                d.metadata["source"] = uploaded.name
            all_docs.extend(loaded_docs)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    if not all_docs:
        return "처리할 문서가 없습니다."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    st.session_state.vectorstore = vectorstore
    st.session_state.rag_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    st.session_state.rag_ready = True
    return "저장이 끝났습니다."


def render_clock_top_center() -> None:
    st.markdown(
        """
<style>
.clock-wrap { width: 100%; display: flex; justify-content: center; margin-top: 10px; margin-bottom: 16px; }
.clock-panel { background: #000; border-radius: 12px; padding: 18px 28px; text-align: center; min-width: 420px; }
.clock-date { font-family: 'Courier New', monospace; color: #ffff00; font-size: 34px; font-weight: 700; }
.clock-time { font-family: 'Courier New', monospace; color: #00ff00; font-size: 66px; font-weight: 700; text-shadow: 0 0 16px #00ff00; }
</style>
""",
        unsafe_allow_html=True,
    )
    now = datetime.now()
    st.markdown(
        f"""
<div class="clock-wrap">
  <div class="clock-panel">
    <div class="clock-date">{now.strftime('%Y년 %m월 %d일')}</div>
    <div class="clock-time">{now.strftime('%H:%M:%S')}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    time_module.sleep(1)
    st.rerun()


def build_agent_executor() -> AgentExecutor:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    tools = [time_tool, chatbot_tool, rag_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "사용자 질문을 보고 반드시 아래 중 하나의 도구를 선택해 실행하세요.\n"
                    "1) time_tool: 시간/날짜 질문\n"
                    "2) chatbot_tool: 일반 상식, 일반 대화\n"
                    "3) rag_tool: 업로드 문서에 대한 질문/요약\n"
                    "가능하면 정확히 한 개 도구를 사용하고 한국어로 답하세요."
                ),
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        callbacks=[ToolTrackCallback()],
        max_iterations=4,
    )


with st.sidebar:
    st.title("🤖 AI Agent")
    current_tool = st.session_state.selected_tool or "아직 없음"
    st.markdown(
        f'<div class="tool-badge">선택된 tool: {current_tool}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("PDF 파일 업로드")
    files = st.file_uploader(
        "RAG용 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("PDF 처리하기", use_container_width=True) and files:
        with st.spinner("PDF 처리 중..."):
            try:
                msg = process_pdfs(files)
                st.success(msg)
            except Exception as e:
                st.error(f"PDF 처리 오류: {e}")

    st.markdown("---")
    if st.button("새로시작하기", use_container_width=True):
        for key in [
            "selected_tool",
            "tool_history",
            "agent_messages",
            "chatbot_history",
            "rag_history",
            "rag_ready",
            "rag_retriever",
            "vectorstore",
        ]:
            del st.session_state[key]
        init_state()
        st.rerun()


st.markdown(
    """
<div style="text-align:center; margin-top:-15px; margin-bottom:4px;">
  <h1><span style="color:#1f77b4;">One</span> <span style="color:#ffd700;">OSS</span> <span style="color:#ff69b4;">Agent</span></h1>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.selected_tool == "챗봇":
    st.title("My First Chatbot")
elif st.session_state.selected_tool == "RAG":
    st.title("RAG 챗봇")

for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("질문을 입력하세요"):
    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        wait_box = st.empty()
        wait_box.markdown("AI Agent가 tool을 선택 중입니다...")
        try:
            executor = build_agent_executor()
            history = []
            for msg in st.session_state.agent_messages[:-1]:
                history.append(
                    HumanMessage(content=msg["content"])
                    if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                )
            result = executor.invoke({"input": user_input, "chat_history": history})
            answer = result["output"]
            wait_box.markdown(answer)
            st.session_state.agent_messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            err = f"오류가 발생했습니다: {e}"
            wait_box.markdown(err)
            st.session_state.agent_messages.append({"role": "assistant", "content": err})

if st.session_state.selected_tool == "시간":
    render_clock_top_center()
