import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. "
    "말투는 존대말 한글로 해줘."
)

TOOL_LABELS = {
    "time_tool": "시간",
    "chatbot_tool": "챗봇",
    "internet_search_tool": "인터넷검색",
    "rag_tool": "RAG",
}


def init_state() -> None:
    defaults = {
        "selected_tool_name": "아직 없음",
        "route_history": [],
        "chatbot_messages": [],
        "internet_messages": [],
        "rag_messages": [],
        "rag_retriever": None,
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_ref_style() -> None:
    st.markdown(
        """
        <style>
        h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
        h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
        h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
        .stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
        .stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover { background-color: #ff1493 !important; }
        .selected-tool {
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


class ToolTrackCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        del input_str, kwargs
        tool_name = serialized.get("name", "")
        st.session_state.selected_tool_name = TOOL_LABELS.get(tool_name, tool_name)


def render_time_clock() -> None:
    components.html(
        """
        <div style="width:100%; display:flex; justify-content:center; margin-bottom:12px;">
          <div style="background:#000; border-radius:12px; padding:18px 24px; text-align:center; min-width:420px;">
            <div id="dateText" style="font-family:'Courier New',monospace; color:#ffff00; font-size:34px; font-weight:700;"></div>
            <div id="timeText" style="font-family:'Courier New',monospace; color:#00ff00; font-size:66px; font-weight:700; text-shadow:0 0 16px #00ff00;"></div>
          </div>
        </div>
        <script>
          function pad(v){ return String(v).padStart(2, "0"); }
          function tick() {
            const now = new Date();
            const date = now.getFullYear() + "-" + pad(now.getMonth()+1) + "-" + pad(now.getDate());
            const time = pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
            document.getElementById("dateText").innerText = date;
            document.getElementById("timeText").innerText = time;
          }
          tick();
          setInterval(tick, 1000);
        </script>
        """,
        height=200,
    )


def load_pdf_docs(uploaded_files: List[Any]) -> List[Any]:
    all_docs: List[Any] = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_path = temp_pdf.name
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return all_docs


def build_ensemble_retriever(docs: List[Any]) -> EnsembleRetriever:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    faiss_store = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 4
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])


def build_history_for_openai(raw_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in raw_messages]


def build_history_for_langchain(raw_messages: List[Dict[str, str]]) -> List[Any]:
    messages: List[Any] = []
    for item in raw_messages:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def make_agent_executor(api_key: str) -> AgentExecutor:
    router_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

    @tool("time_tool")
    def time_tool(user_question: str) -> str:
        """시간 또는 날짜를 물어보면 실행하는 도구."""
        del user_question
        st.session_state.selected_tool_name = "시간"
        now = datetime.now()
        return (
            f"현재 날짜는 {now.strftime('%Y-%m-%d')}이고 "
            f"현재 시간은 {now.strftime('%H:%M:%S')}입니다."
        )

    @tool("chatbot_tool")
    def chatbot_tool(user_question: str) -> str:
        """일반적인 질문에 대해 LLM 지식으로 답변하는 도구."""
        st.session_state.selected_tool_name = "챗봇"
        client = OpenAI(api_key=api_key)
        st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=build_history_for_openai(st.session_state.chatbot_messages),
        )
        answer = response.choices[0].message.content or ""
        st.session_state.chatbot_messages.append({"role": "assistant", "content": answer})
        return answer

    @tool("internet_search_tool")
    def internet_search_tool(user_question: str) -> str:
        """최신 정보가 필요할 때 gpt-5.2 web_search로 답하는 도구."""
        st.session_state.selected_tool_name = "인터넷검색"
        client = OpenAI(api_key=api_key)
        st.session_state.internet_messages.append({"role": "user", "content": user_question})
        conversation_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in st.session_state.internet_messages
        )
        response = client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=(
                "아래 대화 맥락을 참고해 최신 질문에 답변하세요.\n\n"
                f"{conversation_text}"
            ),
        )
        answer = response.output_text
        st.session_state.internet_messages.append({"role": "assistant", "content": answer})
        return answer

    @tool("rag_tool")
    def rag_tool(user_question: str) -> str:
        """특정 문서 요약 또는 문서 기반 질문에 답변하는 RAG 도구."""
        st.session_state.selected_tool_name = "RAG"
        if st.session_state.rag_retriever is None:
            return "RAG 문서가 준비되지 않았습니다. 왼쪽에서 PDF를 업로드하고 처리해 주세요."

        retrieved_docs = st.session_state.rag_retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs[:6])
        rag_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=api_key)

        messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(build_history_for_langchain(st.session_state.rag_messages))
        messages.append(
            HumanMessage(
                content=(
                    "아래 문맥을 참고해 질문에 답변해주세요.\n\n"
                    f"문맥:\n{context}\n\n"
                    f"질문: {user_question}"
                )
            )
        )
        answer = rag_llm.invoke(messages).content
        answer_text = answer if isinstance(answer, str) else str(answer)
        st.session_state.rag_messages.append({"role": "user", "content": user_question})
        st.session_state.rag_messages.append({"role": "assistant", "content": answer_text})
        return answer_text

    tools = [time_tool, chatbot_tool, internet_search_tool, rag_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 tool 라우팅 에이전트야. 사용자의 질문을 보고 반드시 딱 1개의 tool만 선택해 실행해.\n"
                "- 시간/날짜 질문: time_tool\n"
                "- 일반 질문(LLM 지식 답변 가능): chatbot_tool\n"
                "- 인터넷 최신 정보 필요: internet_search_tool\n"
                "- 특정 문서 요약/문서 질문: rag_tool\n"
                "tool 결과를 한국어 존댓말로 깔끔하게 반환해.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=router_llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[ToolTrackCallback()],
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
    )


def main() -> None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="One Agent App", page_icon="🧩", layout="wide")
    apply_ref_style()
    init_state()

    with st.sidebar:
        st.header("통합 앱 메뉴")
        st.markdown(
            f'<div class="selected-tool">선택된 tool: {st.session_state.selected_tool_name}</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=["pdf"],
            accept_multiple_files=True,
            key="one_agent_pdf_uploader",
        )
        if st.button("파일 처리하기", use_container_width=True):
            if not uploaded_files:
                st.warning("PDF 파일을 업로드해 주세요.")
            else:
                with st.spinner("PDF 텍스트 추출 및 벡터 저장 중..."):
                    docs = load_pdf_docs(uploaded_files)
                    st.session_state.rag_retriever = build_ensemble_retriever(docs)
                    st.session_state.processed_files = [file.name for file in uploaded_files]
                st.success("저장이 끝났습니다.")

        if st.session_state.processed_files:
            st.write(", ".join(st.session_state.processed_files))

        if st.button("새로시작하기", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown(
        """
        <div style="text-align:center; margin-top:-1.0rem; margin-bottom:0.5rem;">
            <h1 style="font-size:4.5rem; font-weight:bold; margin:0; line-height:1.1;">
                <span style="color:#1f77b4;">One</span>
                <span style="color:#ffd700;">Agent</span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.selected_tool_name == "시간":
        render_time_clock()
    elif st.session_state.selected_tool_name == "챗봇":
        st.title("My First Chatbot")
    elif st.session_state.selected_tool_name == "인터넷검색":
        st.title("Internet Search Chatbot")
    elif st.session_state.selected_tool_name == "RAG":
        st.title("RAG Chatbot")

    for msg in st.session_state.route_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not api_key:
        st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        st.stop()

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.route_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("AI Agent가 tool을 선택 중입니다...")
        try:
            executor = make_agent_executor(api_key)
            history = build_history_for_langchain(st.session_state.route_history[:-1])
            result = executor.invoke({"input": user_input, "chat_history": history})
            answer = result.get("output", "")
        except Exception as exc:
            answer = f"오류가 발생했습니다: {exc}"
        placeholder.markdown(answer)

    st.session_state.route_history.append({"role": "assistant", "content": answer})
    st.rerun()


if __name__ == "__main__":
    main()
