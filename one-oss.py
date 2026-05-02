import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

APP_TITLE = "3개 통합 앱 만들기 (one-oss)"
OLLAMA_MODEL = "gemma4"
EMBEDDING_MODEL = "bge-m3"

RAG_SYSTEM_MESSAGE = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. "
    "말투는 존대말 한글로 해줘."
)

TOOL_LABELS = {
    "time_tool": "시간",
    "chatbot_tool": "챗봇",
    "rag_tool": "RAG",
}

st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage p { font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.5rem 0 !important; }
.stChatMessage ul, .stChatMessage ol, .stChatMessage li { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
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
        "selected_tool_name": "아직 없음",
        "agent_messages": [],
        "chatbot_history": [],
        "rag_history": [],
        "rag_retriever": None,
        "processed_file_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


class ToolTrackCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "")
        st.session_state.selected_tool_name = TOOL_LABELS.get(tool_name, tool_name)


def render_live_clock() -> None:
    components.html(
        """
        <div style="width:100%;display:flex;justify-content:center;margin-bottom:14px;">
          <div style="background:#000;border-radius:12px;padding:16px 24px;text-align:center;min-width:420px;">
            <div id="clockDate" style="font-family:'Courier New',monospace;color:#ffff00;font-size:34px;font-weight:700;"></div>
            <div id="clockTime" style="font-family:'Courier New',monospace;color:#00ff00;font-size:66px;font-weight:700;text-shadow:0 0 16px #00ff00;"></div>
          </div>
        </div>
        <script>
          function pad(v){ return String(v).padStart(2, "0"); }
          function tick(){
            const now = new Date();
            const date = now.getFullYear() + "-" + pad(now.getMonth()+1) + "-" + pad(now.getDate());
            const time = pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
            document.getElementById("clockDate").innerText = date;
            document.getElementById("clockTime").innerText = time;
          }
          tick();
          setInterval(tick, 1000);
        </script>
        """,
        height=190,
    )


def get_history_messages(history: List[Dict[str, str]], limit: int = 16) -> List[Any]:
    messages: List[Any] = []
    for row in history[-limit:]:
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    return messages


def process_pdf_files(uploaded_files: List[Any]) -> str:
    all_docs = []
    file_names = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_path = temp_pdf.name
        try:
            docs = PyPDFLoader(temp_path).load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
            file_names.append(uploaded_file.name)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    if not all_docs:
        return "처리할 PDF가 없습니다."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 4

    st.session_state.rag_retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5],
    )
    st.session_state.processed_file_names = file_names
    return "저장이 끝났습니다."


@tool
def time_tool(_: str = "") -> str:
    """시간이나 날짜 질문일 때 호출하는 도구."""
    st.session_state.selected_tool_name = "시간"
    now = datetime.now()
    return f"현재 날짜는 {now.strftime('%Y-%m-%d')}이고 현재 시간은 {now.strftime('%H:%M:%S')}입니다."


@tool
def chatbot_tool(question: str) -> str:
    """일반 질문이나 LLM 자체 지식으로 답할 수 있는 질문에 사용하는 도구."""
    st.session_state.selected_tool_name = "챗봇"
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.6)
    messages: List[Any] = [SystemMessage(content="친절한 한국어 챗봇으로 답하세요.")]
    messages.extend(get_history_messages(st.session_state.chatbot_history))
    messages.append(HumanMessage(content=question))
    answer = str(llm.invoke(messages).content)
    st.session_state.chatbot_history.append({"role": "user", "content": question})
    st.session_state.chatbot_history.append({"role": "assistant", "content": answer})
    return answer


@tool
def rag_tool(question: str) -> str:
    """특정 문서 요약 또는 문서 기반 질문에 사용하는 RAG 도구."""
    st.session_state.selected_tool_name = "RAG"
    if st.session_state.rag_retriever is None:
        return "왼쪽 사이드바에서 PDF 파일을 먼저 업로드해 주세요."

    docs = st.session_state.rag_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs[:6])

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    messages: List[Any] = [SystemMessage(content=RAG_SYSTEM_MESSAGE)]
    messages.extend(get_history_messages(st.session_state.rag_history))
    messages.append(
        HumanMessage(
            content=f"문맥:\n{context}\n\n질문:\n{question}\n\n문맥을 기반으로 정확하게 답변해 주세요."
        )
    )
    answer = str(llm.invoke(messages).content)
    st.session_state.rag_history.append({"role": "user", "content": question})
    st.session_state.rag_history.append({"role": "assistant", "content": answer})
    return answer


def build_agent_executor() -> AgentExecutor:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    tools = [time_tool, chatbot_tool, rag_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "사용자 질문을 보고 반드시 도구 1개만 선택하세요.\n"
                    "- 시간/날짜 관련 질문이면 time_tool\n"
                    "- 일반 질문이면 chatbot_tool\n"
                    "- 특정 문서 요약/문서 기반 질문이면 rag_tool\n"
                    "답변은 한국어 존댓말로 자연스럽게 작성하세요."
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
        callbacks=[ToolTrackCallback()],
        verbose=False,
        max_iterations=4,
        handle_parsing_errors=True,
    )


with st.sidebar:
    st.title("🤖 AI Agent")
    st.markdown(
        f'<div class="tool-badge">선택된 tool: {st.session_state.selected_tool_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "PDF 파일",
        type=["pdf"],
        accept_multiple_files=True,
        key="one_oss_pdf_files",
    )
    if st.button("PDF 처리하기", use_container_width=True):
        if not uploaded_files:
            st.warning("PDF 파일을 선택해 주세요.")
        else:
            with st.spinner("PDF 처리 중..."):
                try:
                    done_message = process_pdf_files(uploaded_files)
                    st.success(done_message)
                except Exception as exc:
                    st.error(f"PDF 처리 중 오류가 발생했습니다: {exc}")
    st.markdown("---")
    if st.button("새로시작하기", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


st.markdown(
    """
<div style="margin-top: -3rem; margin-bottom: 1rem;">
  <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
      <span style="color: #1f77b4;">One</span>
      <span style="color: #ffd700;">OSS</span>
    </h1>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.selected_tool_name == "시간":
    render_live_clock()
elif st.session_state.selected_tool_name == "챗봇":
    st.title("My First Chatbot")
elif st.session_state.selected_tool_name == "RAG":
    st.title("RAG Chatbot")

for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("질문을 입력하세요"):
    st.session_state.agent_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("AI Agent가 tool을 선택하는 중입니다...")
        try:
            executor = build_agent_executor()
            history_messages = get_history_messages(st.session_state.agent_messages[:-1], limit=20)
            result = executor.invoke({"input": user_question, "chat_history": history_messages})
            answer_text = str(result["output"])
        except Exception as exc:
            answer_text = f"오류가 발생했습니다: {exc}"
        placeholder.markdown(answer_text)
        st.session_state.agent_messages.append({"role": "assistant", "content": answer_text})
    st.rerun()
