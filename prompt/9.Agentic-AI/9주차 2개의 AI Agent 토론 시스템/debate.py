import os
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

MODEL_OPTIONS = ["gpt-5.4", "gemini-3-pro-preview", "claude-opus-4-7"]
TOOL_OPTIONS = ["인터넷 검색", "문서검색", "도구 없음"]


@dataclass
class AgentConfig:
    name: str
    role_description: str
    model_name: str
    retriever: Optional[object] = None


def ensure_session_state() -> None:
    defaults = {
        "debate_history": [],
        "debate_summary": "",
        "running": False,
        "temp_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app_state() -> None:
    for temp_path in st.session_state.get("temp_files", []):
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass
    st.session_state.debate_history = []
    st.session_state.debate_summary = ""
    st.session_state.running = False
    st.session_state.temp_files = []


def create_llm(model_name: str):
    if model_name == "gpt-5.4":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다.")
        return ChatOpenAI(model="gpt-5.4", api_key=api_key, temperature=0.7)
    if model_name == "gemini-3-pro-preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 필요합니다.")
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=0.7)
    if model_name == "claude-opus-4-7":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(model="claude-opus-4-7", anthropic_api_key=api_key, temperature=0.7)
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


def build_retriever_from_pdf(pdf_path: str):
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs)
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vector_store.as_retriever(search_kwargs={"k": 4})


def run_web_search(query: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY가 없어 인터넷 검색을 수행할 수 없습니다."
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5.2",
            input=query,
            tools=[{"type": "web_search", "search_context_size": "high"}],
        )
        return response.output_text or ""
    except Exception as exc:
        return f"인터넷 검색 실패: {exc}"


def fetch_doc_context(retriever, query: str) -> str:
    if retriever is None:
        return ""
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content[:1000] for doc in docs[:3])


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def should_stop_for_repetition(message: str, history: List[Tuple[str, str]], threshold: float = 0.88) -> bool:
    recent_messages = [msg for _, msg in history[-4:]]
    return any(similarity(message, old) >= threshold for old in recent_messages)


def generate_agent_turn(
    agent: AgentConfig,
    other_agent_name: str,
    topic: str,
    tool_type: str,
    history: List[Tuple[str, str]],
) -> str:
    llm = create_llm(agent.model_name)
    history_text = "\n".join([f"{speaker}: {content}" for speaker, content in history]) if history else "(아직 발언 없음)"
    context_block = ""
    if tool_type == "인터넷 검색":
        query = f"토론 주제: {topic}\n현재까지 대화:\n{history_text}\n\n{agent.name}에게 유리한 최신 근거를 찾아줘."
        context_block = run_web_search(query)
    elif tool_type == "문서검색":
        context_block = fetch_doc_context(agent.retriever, f"{topic} 관련 {agent.role_description}")

    prompt = f"""
당신은 토론 에이전트입니다.
- 이름: {agent.name}
- 입장: {agent.role_description}
- 상대: {other_agent_name}
- 토론 주제: {topic}

규칙:
1) 이전 발언과 똑같은 말을 반복하지 마세요.
2) 새 근거 또는 새 반박이 없다면 "더 이상 새로운 주장이 없습니다."로 짧게 말하세요.
3) 이름 prefix("{agent.name}:") 없이 바로 발언하세요.
4) 한국어로 3~6문장 이내로 간결하게 발언하세요.

[도구/참고 정보]
{context_block if context_block else "별도 참고 정보 없음"}

[토론 기록]
{history_text}
"""
    result = llm.invoke(prompt)
    content = getattr(result, "content", str(result)).strip()
    if content.startswith(f"{agent.name}:"):
        content = content[len(agent.name) + 1 :].strip()
    return content


def summarize_debate(topic: str, agent1: str, agent2: str, history: List[Tuple[str, str]]) -> str:
    llm = create_llm("gpt-5.4")
    convo = "\n".join([f"{speaker}: {text}" for speaker, text in history])
    prompt = f"""
너는 사회자다. 아래 토론을 정리해라.
주제: {topic}
참가자: {agent1}, {agent2}

요구사항:
- 두 참가자의 핵심 논리(찬반)를 균형 있게 요약
- 마지막에 사회자의 개인 생각을 명확히 제시
- 한국어로 작성

토론:
{convo}
"""
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def render_header() -> None:
    st.markdown(
        """
        <div style='text-align:center; padding:24px; background:#000; border-radius:16px; margin-bottom:18px;'>
          <h1 style='margin:0; font-size:3rem; font-weight:800;
              background:linear-gradient(90deg,#FFD700,#FF6B6B,#4ECDC4,#45B7D1);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            2개의 AI Agent 토론 시스템
          </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="2개의 AI Agent 토론 시스템", page_icon="🤖", layout="wide")
    ensure_session_state()
    render_header()

    with st.sidebar:
        max_rounds = st.slider("최대 토론횟수", min_value=2, max_value=20, value=6)
        tool_type = st.radio("도구 유형", TOOL_OPTIONS, index=0)
        restart_clicked = st.button("다시 시작하기", use_container_width=True)
        if restart_clicked:
            reset_app_state()
            st.rerun()

    st.subheader("토론 설정")
    topic = st.text_area("토론주제:", placeholder="예: AI 규제는 혁신을 저해하는가?")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 에이전트1 모델 설정")
        agent1_model = st.selectbox("모델 선택", MODEL_OPTIONS, key="agent1_model")
        agent1_name = st.text_input("에이전트 1 이름", value="찬성측", key="agent1_name")
        agent1_desc = st.text_area("에이전트 1 입장설명", key="agent1_desc")
        agent1_pdf = None
        if tool_type == "문서검색":
            agent1_pdf = st.file_uploader("에이전트 1 PDF 문서 선택", type=["pdf"], key="agent1_pdf")

    with col_right:
        st.markdown("### 에이전트2 모델 설정")
        agent2_model = st.selectbox("모델 선택", MODEL_OPTIONS, key="agent2_model")
        agent2_name = st.text_input("에이전트 2 이름", value="반대측", key="agent2_name")
        agent2_desc = st.text_area("에이전트 2 입장설명", key="agent2_desc")
        agent2_pdf = None
        if tool_type == "문서검색":
            agent2_pdf = st.file_uploader("에이전트 2 PDF 문서 선택", type=["pdf"], key="agent2_pdf")

    start_clicked = st.button("토론 시작", type="primary", use_container_width=True)

    # 직전 토론 내용이 있으면 버튼을 누르지 않아도 항상 다시 보여줍니다.
    if st.session_state.debate_history:
        st.markdown("---")
        st.subheader("토론진행")
        for speaker_name, speaker_message in st.session_state.debate_history:
            with st.chat_message("user" if speaker_name == st.session_state.get("last_agent1_name") else "assistant"):
                st.markdown(f"**{speaker_name}**: {speaker_message}")
        if st.session_state.debate_summary:
            st.markdown("---")
            st.subheader("사회자의 정리요약및 개인생각 표시")
            st.markdown(st.session_state.debate_summary)

    if not start_clicked:
        return

    if not topic.strip() or not agent1_desc.strip() or not agent2_desc.strip():
        st.error("토론주제와 두 에이전트의 입장설명을 모두 입력해주세요.")
        return
    if tool_type == "문서검색" and (agent1_pdf is None or agent2_pdf is None):
        st.error("문서검색 모드에서는 에이전트 2명 모두 PDF를 업로드해야 합니다.")
        return

    try:
        agent1_retriever = None
        agent2_retriever = None
        if tool_type == "문서검색":
            path1 = save_uploaded_pdf(agent1_pdf)
            path2 = save_uploaded_pdf(agent2_pdf)
            st.session_state.temp_files.extend([path1, path2])
            agent1_retriever = build_retriever_from_pdf(path1)
            agent2_retriever = build_retriever_from_pdf(path2)

        agent1 = AgentConfig(name=agent1_name, role_description=agent1_desc, model_name=agent1_model, retriever=agent1_retriever)
        agent2 = AgentConfig(name=agent2_name, role_description=agent2_desc, model_name=agent2_model, retriever=agent2_retriever)
    except Exception as exc:
        st.error(f"에이전트 준비 중 오류: {exc}")
        return

    st.markdown("---")
    st.subheader("토론진행")

    history: List[Tuple[str, str]] = []
    st.session_state.debate_summary = ""
    st.session_state.last_agent1_name = agent1.name
    progress = st.progress(0.0)
    live_debate_box = st.container()
    short_count = 0

    for round_index in range(max_rounds):
        speaker = agent1 if round_index % 2 == 0 else agent2
        listener_name = agent2.name if speaker.name == agent1.name else agent1.name
        try:
            message = generate_agent_turn(speaker, listener_name, topic, tool_type, history)
        except Exception as exc:
            st.error(f"{speaker.name} 발언 생성 중 오류: {exc}")
            break

        if should_stop_for_repetition(message, history):
            st.info(f"{speaker.name}이(가) 중복된 발언을 하여 토론을 종료합니다.")
            break

        history.append((speaker.name, message))
        with live_debate_box:
            st.chat_message("user" if speaker.name == agent1.name else "assistant").markdown(
                f"**{speaker.name}**: {message}"
            )
        progress.progress((round_index + 1) / max_rounds)

        if "더 이상 새로운 주장이 없습니다" in message or len(message.split()) < 8:
            short_count += 1
            if short_count >= 2:
                st.info("양측 모두 추가 주장 여지가 적어 토론을 종료합니다.")
                break
        else:
            short_count = 0

    st.session_state.debate_history = history
    progress.empty()

    st.markdown("---")
    st.subheader("사회자의 정리요약및 개인생각 표시")
    if history:
        with st.spinner("사회자가 토론을 정리하는 중입니다..."):
            try:
                summary = summarize_debate(topic, agent1.name, agent2.name, history)
            except Exception as exc:
                summary = f"요약 생성 실패: {exc}"
        st.session_state.debate_summary = summary
        st.markdown(summary)
    else:
        st.warning("표시할 토론 내용이 없습니다.")


if __name__ == "__main__":
    main()

