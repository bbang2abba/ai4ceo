"""
Streamlit SQL Agent: Chinook DB에 질의하고 LangChain AgentExecutor로 SQL을 실행합니다.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents.agent import AgentExecutor
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


# --- 환경 ---
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

APP_TITLE = "SQL Agent"
LLM_MODEL = "gpt-5.2"


def get_openai_api_key() -> str:
    """프로젝트 루트 .env에서 OPENAI_API_KEY를 읽습니다."""
    if os.path.exists(env_path):
        from dotenv import dotenv_values

        env_vars = dotenv_values(env_path)
        key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY")
    return key or ""


def setup_logger() -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"sql_agent_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


LOGGER = setup_logger()


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\n\s*-{3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*={3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*_{3,}\s*\n", "\n\n", text)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*={3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*_{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=api_key)


def candidate_db_paths() -> List[Path]:
    root = Path(__file__).resolve().parent
    cwd = Path.cwd()
    return [
        root / "chinook.db",
        root / "prompt" / "5.SQL-Agent" / "chinook.db",
        cwd / "chinook.db",
        cwd / "prompt" / "5.SQL-Agent" / "chinook.db",
    ]


def locate_chinook_db() -> Optional[Path]:
    for candidate in candidate_db_paths():
        if candidate.exists():
            return candidate
    return None


def init_state() -> None:
    defaults = {
        "chat_history": [],
        "db_connection": None,
        "sql_agent_executor": None,
        "db_initialized": False,
        "db_name": "",
        "db_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_style() -> None:
    st.markdown(
        """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage p, .stChatMessage li { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}
.stButton > button:hover { background-color: #ff1493 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


class SQLProgressCallback(BaseCallbackHandler):
    """SQL 에이전트 실행 단계에 맞춰 진행률을 갱신합니다."""

    def __init__(self, progress_bar: Any, status_box: Any) -> None:
        self.progress_bar = progress_bar
        self.status_box = status_box

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        self.status_box.text("질문 해석 중...")
        self.progress_bar.progress(10)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name", "") or ""
        if "sql" in name.lower() or "database" in name.lower():
            self.status_box.text("SQL 실행 중...")
            self.progress_bar.progress(55)
        else:
            self.status_box.text("도구 실행 중...")
            self.progress_bar.progress(45)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.status_box.text("결과 처리 중...")
        self.progress_bar.progress(82)

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        self.status_box.text("응답 정리 중...")
        self.progress_bar.progress(95)


def build_sql_agent_executor(db_connection: SQLDatabase) -> AgentExecutor:
    """LangChain create_sql_agent → AgentExecutor (요구사항: AgentExecutor 활용)."""
    llm = get_llm(temperature=0.2)
    executor: AgentExecutor = create_sql_agent(
        llm=llm,
        db=db_connection,
        agent_type="tool-calling",
        verbose=False,
        max_iterations=15,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
        },
    )
    return executor


def initialize_database(db_path: Path) -> bool:
    try:
        db_uri = f"sqlite:///{db_path.as_posix()}"
        db_connection = SQLDatabase.from_uri(db_uri)
        sql_agent_executor = build_sql_agent_executor(db_connection)

        st.session_state.db_connection = db_connection
        st.session_state.sql_agent_executor = sql_agent_executor
        st.session_state.db_initialized = True
        st.session_state.db_name = db_path.name
        st.session_state.db_path = str(db_path)
        LOGGER.info("Database initialized: %s", db_path)
        return True
    except Exception as exc:
        LOGGER.error("Database initialization error: %s", exc)
        st.error(f"데이터베이스 초기화 중 오류가 발생했습니다: {exc}")
        return False


def _parse_three_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line:
            lines.append(line)
    return lines[:3]


def generate_followup_questions(user_prompt: str, answer_text: str, db_schema: str) -> List[str]:
    try:
        llm = get_llm(temperature=0.6)
        prompt = f"""사용자 질문:
{user_prompt}

에이전트 답변:
{answer_text}

데이터베이스 스키마(요약):
{db_schema[:12000]}

위 스키마에 실제로 존재하는 테이블·컬럼만 사용해, SQL로 답할 수 있는 후속 질문을 정확히 3개만 작성하세요.

규칙:
- 한 줄에 질문 하나, 총 3줄
- 한국어 존댓말
- 번호, 기호, 따옴표, 마크다운 없이 질문 문장만
- 너무 일반적인 질문 금지
"""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_three_lines(content)
    except Exception as exc:
        LOGGER.warning("Follow-up generation failed: %s", exc)
        return []


def reset_connection() -> None:
    st.session_state.db_connection = None
    st.session_state.sql_agent_executor = None
    st.session_state.db_initialized = False
    st.session_state.db_name = ""
    st.session_state.db_path = ""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗄️", layout="wide")
    init_state()
    apply_style()

    st.markdown(
        """
<div style="margin-top: -2rem; margin-bottom: 0.5rem;">
<div style="text-align: center;">
    <h1 style="font-size: 5.5rem; font-weight: bold; margin: 0; line-height: 1.1;">
        <span style="color: #1f77b4;">SQL</span>
        <span style="color: #ffd700;">Agent</span>
    </h1>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("🗄️ **Chinook** 샘플 DB에 대해 자연어로 질문하면 SQL을 생성·실행한 뒤 결과를 설명합니다.")

    if not st.session_state.db_initialized:
        auto_db = locate_chinook_db()
        if auto_db:
            with st.spinner(f"`{auto_db.name}` 연결 중..."):
                if initialize_database(auto_db):
                    st.rerun()

    if st.session_state.db_initialized:
        st.info(f"연결된 데이터베이스: **{st.session_state.db_name}** (`{st.session_state.db_path}`)")
    else:
        st.warning("`chinook.db`를 찾지 못했습니다. 프로젝트 루트 또는 `prompt/5.SQL-Agent/`에 파일을 두거나 사이드바에서 재연결을 시도하세요.")

    with st.sidebar:
        st.markdown('<h2 style="color: #ffd700;">데이터베이스</h2>', unsafe_allow_html=True)
        if st.session_state.db_initialized:
            st.success(f"연결됨: **{st.session_state.db_name}**")
            st.caption(st.session_state.db_path)
        else:
            st.markdown("탐색 경로:")
            for p in candidate_db_paths():
                st.caption(str(p))

        if st.button("chinook.db 다시 연결", type="primary", use_container_width=True):
            reset_connection()
            db = locate_chinook_db()
            if db is None:
                st.error("chinook.db를 찾지 못했습니다.")
            elif initialize_database(db):
                st.rerun()

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<h3 style="color: #1f77b4;">현재 상태</h3>', unsafe_allow_html=True)
        st.text(f"DB: {st.session_state.db_name or '미연결'}")
        st.text(f"모델: {LLM_MODEL}")
        st.text(f"대화 수: {len(st.session_state.chat_history)}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요. 예: 장르별 트랙 수를 알려주세요."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.db_initialized or st.session_state.sql_agent_executor is None:
            warning = "DB 연결이 필요합니다. `chinook.db`가 있는 경로를 확인한 뒤 다시 시도해주세요."
            with st.chat_message("assistant"):
                st.warning(warning)
            st.session_state.chat_history.append({"role": "assistant", "content": warning})
            st.stop()

        with st.chat_message("assistant"):
            progress = st.progress(0)
            status = st.empty()
            body = st.empty()
            try:
                callback = SQLProgressCallback(progress, status)
                executor = st.session_state.sql_agent_executor
                result = executor.invoke({"input": prompt}, config={"callbacks": [callback]})
                answer = remove_separators(result.get("output", "응답을 생성하지 못했습니다."))

                status.text("추천 질문 생성 중...")
                progress.progress(96)
                db_schema = st.session_state.db_connection.get_table_info()
                suggestions = generate_followup_questions(prompt, answer, db_schema)

                if suggestions:
                    answer += "\n\n### 다음에 물어볼 수 있는 질문\n\n"
                    for idx, question in enumerate(suggestions, start=1):
                        answer += f"{idx}. {question}\n\n"

                progress.progress(100)
                status.empty()
                progress.empty()
                body.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                LOGGER.exception("SQL Agent error: %s", exc)
                progress.empty()
                status.empty()
                err = f"오류가 발생했습니다: {exc}"
                body.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
