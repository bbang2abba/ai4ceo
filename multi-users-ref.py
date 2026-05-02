# -*- coding: utf-8 -*-
"""
PDF 기반 멀티유저 RAG 챗봇
- Supabase Auth 로그인/회원가입(이메일 확인)
- RLS 기준 사용자별 세션·임베딩 분리
- LLM API 키는 사이드바 입력(.env 고정값 없음)
- Supabase URL/키는 os.getenv + Streamlit Secrets 동기화
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import Field, PrivateAttr
from supabase import Client, create_client


def _load_dotenv_multi_paths() -> None:
    """cwd와 무관하게 .env 로드 (Streamlit 실행 경로·Cloud 차이 대응)."""
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2] / ".env",
        here.parents[1] / ".env",
        here.parent / ".env",
        Path.cwd() / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return
    load_dotenv()


_load_dotenv_multi_paths()

# LLM 키는 사이드바에서만 주입(.env·호스트 환경의 잔존 값으로 실행되는 것 방지)
_LLM_ENV_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
for _k in _LLM_ENV_KEYS:
    os.environ.pop(_k, None)

SUPABASE_ERROR: Optional[str] = None


def set_supabase_error(exc: Exception) -> None:
    global SUPABASE_ERROR
    SUPABASE_ERROR = str(exc)


def show_signup_error(exc: Exception) -> None:
    raw = str(exc)
    low = raw.lower()
    if "rate limit" in low or "too many requests" in low:
        st.error("회원가입이 일시적으로 제한되었습니다. (Supabase 이메일·인증 요청 한도 초과)")
        st.caption(
            "잠시 후 다시 시도하거나, Supabase 대시보드 → Authentication → Users에서 사용자를 직접 만든 뒤 "
            "이 화면에서 로그인하세요. 반복 테스트 시 Confirm email을 켠 상태로 가입만 많이 하면 한도에 걸리기 쉽습니다. "
            "(개발 단계에서는 Email → Confirm email을 끄거나, 대시보드에서 사용자 추가를 활용할 수 있습니다.)"
        )
    else:
        st.error(f"회원가입 실패: {raw}")


def show_signin_error(exc: Exception) -> None:
    raw = str(exc)
    low = raw.lower()
    if "invalid login credentials" in low or "invalid_grant" in low or "invalid credentials" in low:
        st.error("로그인 실패: 이메일 또는 비밀번호가 맞지 않습니다.")
        st.caption(
            "· 이메일·비밀번호 오타, 앞뒤 공백을 확인하세요. (이메일은 입력 시 자동으로 공백만 제거합니다.)\n"
            "· 아직 가입·인증을 하지 않았거나 Confirm email 후 메일 링크를 누르지 않았다면 로그인되지 않을 수 있습니다.\n"
            "· Supabase 대시보드 → Authentication → Users에 해당 이메일이 있는지, "
            "**이 앱의 SUPABASE_URL / ANON 키와 같은 프로젝트**인지 확인하세요.\n"
            "· 비밀번호를 잊었다면 대시보드에서 해당 사용자에 대해 비밀번호를 재설정하세요."
        )
    elif "rate limit" in low or "too many requests" in low:
        st.error("로그인 시도가 너무 많습니다. 잠시 후 다시 시도하세요.")
    else:
        st.error(f"로그인 실패: {raw}")


def load_supabase_env_from_secrets() -> None:
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for key in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
            if key in secrets and key not in os.environ:
                os.environ[key] = str(secrets[key])
    except Exception:
        return


def _query_page() -> str:
    raw = st.query_params.get("page", "chat")
    if isinstance(raw, list):
        return (raw[0] if raw else "chat") or "chat"
    return raw if raw else "chat"


def get_supabase() -> Optional[Client]:
    """세션별 클라이언트(전역 캐시 금지 — Streamlit 다중 사용자 간 인증 혼선 방지)."""
    global SUPABASE_ERROR
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    if st.session_state.get("_sb_client") is None:
        st.session_state._sb_client = create_client(url, key)
    client: Client = st.session_state._sb_client
    at = st.session_state.get("sb_access_token")
    rt = st.session_state.get("sb_refresh_token")
    if at and rt:
        try:
            resp = client.auth.set_session(at, rt)
            if resp.session:
                st.session_state.sb_access_token = resp.session.access_token
                st.session_state.sb_refresh_token = resp.session.refresh_token
        except Exception as exc:
            set_supabase_error(exc)
    return client


def kst_now_str() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%m/%d %H:%M")


def ensure_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "processed_files": [],
        "selected_model": "gpt-5.5",
        "retriever": None,
        "current_session_id": str(uuid.uuid4()),
        "last_dropdown_session_id": None,
        "selected_session_id_for_button": None,
        "user_email": None,
        "user_id": None,
        "sessions_bootstrapped": False,
        "enable_web_search": True,
        "signup_email": "",
        "_sb_client": None,
        "sb_access_token": None,
        "sb_refresh_token": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_screen_state() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.retriever = None


def set_api_env(openai_key: str, anthropic_key: str, gemini_key: str) -> None:
    o = (openai_key or "").strip()
    a = (anthropic_key or "").strip()
    g = (gemini_key or "").strip()
    if o:
        os.environ["OPENAI_API_KEY"] = o
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    if a:
        os.environ["ANTHROPIC_API_KEY"] = a
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    if g:
        os.environ["GOOGLE_API_KEY"] = g
    else:
        os.environ.pop("GOOGLE_API_KEY", None)


def llm_api_key_ready(model_name: str) -> bool:
    if model_name == "gpt-5.5":
        return bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if model_name == "claude-opus-4-7":
        return bool((os.getenv("ANTHROPIC_API_KEY") or "").strip())
    return bool((os.getenv("GOOGLE_API_KEY") or "").strip())


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))


def ensure_embedding_dimension(embeddings: OpenAIEmbeddings, expected_dim: int = 1536) -> None:
    vec = embeddings.embed_query("dimension check")
    actual_dim = len(vec)
    if actual_dim != expected_dim:
        raise ValueError(
            f"임베딩 차원 불일치: 현재 {actual_dim}, DB 스키마 vector({expected_dim}). "
            "임베딩 모델 또는 SQL 스키마를 일치시켜주세요."
        )


def build_llm(model_name: str, streaming: bool):
    if model_name == "gpt-5.5":
        return ChatOpenAI(model="gpt-5.5", temperature=1, openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=streaming)
    if model_name == "claude-opus-4-7":
        return ChatAnthropic(
            model="claude-opus-4-7",
            temperature=1,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=streaming,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=streaming,
    )


class SessionRetriever(BaseRetriever):
    k: int = Field(default=8)
    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _user_id: str = PrivateAttr()
    _session_id: str = PrivateAttr()

    def __init__(
        self,
        supabase_client: Client,
        embeddings: OpenAIEmbeddings,
        user_id: str,
        session_id: str,
        k: int = 8,
    ):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._user_id = user_id
        self._session_id = session_id

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        vec = self._embeddings.embed_query(query)
        result = self._supabase.rpc(
            "match_documents",
            {
                "query_embedding": vec,
                "match_threshold": 0.55,
                "match_count": self.k,
                "user_id_filter": self._user_id,
                "session_id_filter": self._session_id,
            },
        ).execute()
        docs: List[Document] = []
        for row in (result.data or []):
            docs.append(
                Document(
                    page_content=row.get("chunk_text", ""),
                    metadata={
                        **(row.get("metadata") or {}),
                        "file_name": row.get("file_name", ""),
                        "chunk_index": row.get("chunk_index", 0),
                    },
                )
            )
        return docs


def _store_auth_session(access_token: str, refresh_token: str, email: Optional[str], uid: Optional[str]) -> None:
    st.session_state.sb_access_token = access_token
    st.session_state.sb_refresh_token = refresh_token
    if email:
        st.session_state.user_email = email
    if uid:
        st.session_state.user_id = uid
    sb = get_supabase()
    if sb:
        try:
            sb.auth.set_session(access_token, refresh_token)
        except Exception as exc:
            set_supabase_error(exc)


def sign_up(email: str, password: str) -> bool:
    sb = get_supabase()
    if not sb:
        st.error("Supabase 연결이 필요합니다.")
        return False
    try:
        result = sb.auth.sign_up({"email": email, "password": password})
        st.session_state.signup_email = email
        if result.session and result.user:
            _store_auth_session(
                result.session.access_token,
                result.session.refresh_token,
                email,
                result.user.id,
            )
            st.session_state.sessions_bootstrapped = False
        return True
    except Exception as exc:
        show_signup_error(exc)
        return False


def sign_in(email: str, password: str) -> bool:
    sb = get_supabase()
    if not sb:
        st.error("Supabase 연결이 필요합니다.")
        return False
    email = (email or "").strip()
    if not email or not password:
        st.warning("이메일과 비밀번호를 입력하세요.")
        return False
    try:
        result = sb.auth.sign_in_with_password({"email": email, "password": password})
        if not result or not result.session or not result.user:
            st.error("로그인 실패: 이메일 인증(Confirm email)을 완료했는지 확인해주세요.")
            st.caption(
                "가입 직후 세션이 없었다면 인증 메일의 링크를 먼저 눌러야 할 수 있습니다. "
                "대시보드 Users에서 사용자 상태(Confirmed)도 확인하세요."
            )
            return False
        _store_auth_session(
            result.session.access_token,
            result.session.refresh_token,
            email,
            result.user.id,
        )
        st.session_state.sessions_bootstrapped = False
        return True
    except Exception as exc:
        show_signin_error(exc)
        return False


def sign_out() -> None:
    sb = get_supabase()
    if sb:
        try:
            sb.auth.sign_out()
        except Exception:
            pass
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None
    st.session_state._sb_client = None
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.sessions_bootstrapped = False
    st.session_state.last_dropdown_session_id = None
    clear_screen_state()


def generate_session_title() -> str:
    user_msgs = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
    ai_msgs = [m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"]
    if not user_msgs or not ai_msgs:
        return f"세션 {kst_now_str()}"
    if not os.getenv("OPENAI_API_KEY"):
        fallback = user_msgs[0]
        return (fallback[:24] + "...") if len(fallback) > 24 else fallback
    try:
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))
        user_block = "\n".join([f"- {x}" for x in user_msgs])[:1400]
        ai_block = "\n".join([f"- {x}" for x in ai_msgs])[:1800]
        prompt = (
            "다음 전체 대화(사용자 질문 전체 + 답변 전체)를 보고 핵심 주제를 1개 세션명으로 만드세요.\n"
            "조건: 한국어, 24자 이내, 따옴표/특수기호 없이 제목만 출력.\n\n"
            f"[사용자 질문 전체]\n{user_block}\n\n"
            f"[AI 답변 전체]\n{ai_block}\n\n"
            "세션명:"
        )
        title = llm.invoke(prompt).content.strip().strip('"').strip("'")
        return title[:24] if title else f"세션 {kst_now_str()}"
    except Exception:
        fallback = user_msgs[0]
        return (fallback[:24] + "...") if len(fallback) > 24 else fallback


def get_sessions() -> List[Dict[str, Any]]:
    sb = get_supabase()
    if not sb or not st.session_state.user_id:
        return []
    try:
        res = (
            sb.table("sessions")
            .select("id,session_id,title,updated_at")
            .eq("user_id", st.session_state.user_id)
            .order("updated_at", desc=True)
            .limit(200)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        set_supabase_error(exc)
        return []


def save_session(session_id: str, clear_after_save: bool = False) -> bool:
    sb = get_supabase()
    if not sb or not st.session_state.user_id:
        st.warning("로그인 후 세션 저장이 가능합니다.")
        return False
    title = generate_session_title()
    payload = {
        "id": session_id,
        "session_id": session_id,
        "user_id": st.session_state.user_id,
        "title": title,
        "chat_history": json.dumps(st.session_state.chat_history, ensure_ascii=False),
        "conversation_memory": json.dumps(st.session_state.conversation_memory, ensure_ascii=False),
        "processed_files": json.dumps(st.session_state.processed_files, ensure_ascii=False),
    }
    try:
        sb.table("sessions").upsert(payload, on_conflict="id").execute()
        if clear_after_save:
            clear_screen_state()
            st.session_state.current_session_id = str(uuid.uuid4())
        return True
    except Exception as exc:
        set_supabase_error(exc)
        st.error(f"세션 저장 실패: {exc}")
        return False


def load_session(session_id: str) -> bool:
    sb = get_supabase()
    if not sb or not st.session_state.user_id:
        st.warning("로그인 후 세션 로드가 가능합니다.")
        return False
    try:
        res = (
            sb.table("sessions")
            .select("chat_history,conversation_memory,processed_files")
            .eq("id", session_id)
            .eq("user_id", st.session_state.user_id)
            .limit(1)
            .execute()
        )
        row = (res.data or [None])[0]
        if not row:
            st.warning("세션을 찾을 수 없습니다.")
            return False
        st.session_state.chat_history = json.loads(row.get("chat_history") or "[]")
        st.session_state.conversation_memory = json.loads(row.get("conversation_memory") or "[]")
        st.session_state.processed_files = json.loads(row.get("processed_files") or "[]")
        st.session_state.current_session_id = session_id
        restore_retriever(session_id)
        return True
    except Exception as exc:
        set_supabase_error(exc)
        st.error(f"세션 로드 실패: {exc}")
        return False


def delete_session(session_id: str) -> bool:
    sb = get_supabase()
    if not sb or not st.session_state.user_id:
        st.warning("로그인 후 세션 삭제가 가능합니다.")
        return False
    try:
        sb.table("sessions").delete().eq("id", session_id).eq("user_id", st.session_state.user_id).execute()
        if st.session_state.current_session_id == session_id:
            clear_screen_state()
            st.session_state.current_session_id = str(uuid.uuid4())
        return True
    except Exception as exc:
        set_supabase_error(exc)
        st.error(f"세션 삭제 실패: {exc}")
        return False


def save_documents(chunks: List[Document], session_id: str) -> None:
    sb = get_supabase()
    if not sb or not st.session_state.user_id:
        raise ValueError("로그인이 필요합니다.")
    embeddings = get_embeddings()
    ensure_embedding_dimension(embeddings, 1536)
    rows = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if not content:
            continue
        source = str((chunk.metadata or {}).get("source", "unknown.pdf"))
        rows.append(
            {
                "user_id": st.session_state.user_id,
                "session_id": session_id,
                "file_name": source,
                "chunk_index": i,
                "chunk_text": content,
                "embedding": embeddings.embed_query(content),
                "metadata": {"source": source, "session_id": session_id},
            }
        )
    if rows:
        sb.table("embeddings").upsert(
            rows,
            on_conflict="user_id,session_id,file_name,chunk_index",
        ).execute()


def restore_retriever(session_id: str) -> None:
    sb = get_supabase()
    if not sb or not st.session_state.user_id or not os.getenv("OPENAI_API_KEY"):
        st.session_state.retriever = None
        return
    try:
        st.session_state.retriever = SessionRetriever(
            supabase_client=sb,
            embeddings=get_embeddings(),
            user_id=st.session_state.user_id,
            session_id=session_id,
            k=8,
        )
    except Exception as exc:
        set_supabase_error(exc)
        st.session_state.retriever = None


def generate_followups(question: str, answer: str, context: str) -> List[str]:
    try:
        llm = build_llm(st.session_state.selected_model, streaming=False)
        prompt = f"질문:{question}\n답변:{answer[:900]}\n참고:{context[:500]}\n향후 더 필요한 질문 3개만 한 줄씩 출력하세요."
        text = llm.invoke(prompt).content.strip()
        rows = [x.strip("-• 1234567890. ") for x in text.splitlines() if x.strip()]
        return rows[:3]
    except Exception:
        return []


def should_use_web_search(question: str, docs: List[Document]) -> bool:
    q = question.lower()
    web_keywords = [
        "오늘",
        "최근",
        "최신",
        "뉴스",
        "발표",
        "업데이트",
        "release",
        "breaking",
        "price",
        "주가",
        "환율",
        "날씨",
        "검색",
        "internet",
        "web",
    ]
    if not docs:
        return True
    return any(k in q for k in web_keywords)


def web_search_answer(question: str, memory: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        client = OpenAI(api_key=api_key)
        recent_mem = "\n".join(memory[-8:]) if memory else ""
        prompt = (
            "다음 질문에 대해 웹 검색 기반으로 사실 정보를 정리하세요.\n"
            "핵심 사실 위주로 간결하게 작성하고, 불확실한 내용은 추정하지 마세요.\n\n"
            f"[이전 대화]\n{recent_mem}\n\n"
            f"[현재 질문]\n{question}"
        )
        resp = client.responses.create(
            model="gpt-5.5",
            input=prompt,
            tools=[{"type": "web_search", "search_context_size": "high"}],
        )
        return (resp.output_text or "").strip()
    except Exception:
        return ""


def render_signup_page() -> None:
    st.markdown("## 회원가입")
    st.write("Supabase에서 Email → Confirm email을 켠 경우, 가입 후 메일의 링크로 인증하고 같은 이메일·비밀번호로 로그인하면 됩니다.")
    st.caption(
        "짧은 시간에 가입을 여러 번 시도하면 Supabase에서 **email rate limit** 으로 막힐 수 있습니다. "
        "그때는 잠시 쉬었다가 하거나, 대시보드에서 사용자를 직접 추가하세요."
    )
    email = st.text_input("Email", key="signup_email_input")
    password = st.text_input("Password", type="password", key="signup_pw_input")
    if st.button("가입하기", use_container_width=True):
        if not email or not password:
            st.warning("이메일과 비밀번호를 입력하세요.")
        elif sign_up(email, password):
            st.success("가입 요청이 처리되었습니다. Confirm email이 켜져 있으면 메일 인증 후 로그인하세요.")
    if st.session_state.signup_email:
        st.info(f"가입 시도 이메일: {st.session_state.signup_email}")
    if st.button("로그인 화면으로 이동", use_container_width=True):
        st.query_params["page"] = "chat"
        st.rerun()


st.set_page_config(page_title="PDF 기반 멀티유저 RAG 챗봇", page_icon="📚", layout="wide")
load_supabase_env_from_secrets()
ensure_state()

st.markdown(
    """
<style>
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important;
}
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
.stChatMessage p, .stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: bold !important;
}
.stButton > button:hover {
    background-color: #ff1493 !important;
}
.stSidebar .stButton > button {
    font-size: 0.76rem !important;
    padding: 0.3rem 0.65rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align: center; margin-top: -2.5rem; margin-bottom: 0.6rem;">
    <h1 style="font-size: 2.4rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span>
        <span style="color: #9b59b6;">기반</span>
        <span style="color: #ffd700;">멀티유저</span>
        <span style="color: #d62728;">RAG 챗봇</span>
    </h1>
</div>
""",
    unsafe_allow_html=True,
)

_page = _query_page()

with st.sidebar:
    st.subheader("LLM API 키 입력")
    openai_key = st.text_input("OpenAI API Key", type="password", key="sb_openai_key")
    anthropic_key = st.text_input("Anthropic API Key", type="password", key="sb_anthropic_key")
    gemini_key = st.text_input("Gemini API Key", type="password", key="sb_gemini_key")
    set_api_env(openai_key, anthropic_key, gemini_key)

    st.divider()
    st.subheader("Supabase 로그인")
    if not os.getenv("SUPABASE_URL") or not (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        st.error("Secrets 또는 환경 변수에 SUPABASE_URL, SUPABASE_ANON_KEY(권장)를 설정하세요.")
    elif os.getenv("SUPABASE_SERVICE_ROLE_KEY") and not os.getenv("SUPABASE_ANON_KEY"):
        st.warning("SERVICE_ROLE만 있으면 RLS가 우회됩니다. 배포 시 ANON 키 사용을 권장합니다.")

    login_id = st.text_input("Login ID (email)", key="sb_login_email")
    login_pw = st.text_input("Password", type="password", key="sb_login_pw")
    if st.button("로그인", use_container_width=True):
        if login_id and login_pw and sign_in(login_id, login_pw):
            st.success("로그인 완료")
            st.rerun()
    if st.button("회원 가입", use_container_width=True):
        st.query_params["page"] = "signup"
        st.rerun()
    if st.button("로그아웃", use_container_width=True):
        sign_out()
        st.success("로그아웃 완료")
        st.rerun()
    if st.session_state.user_email:
        st.info(f"현재 사용자: {st.session_state.user_email}")
    else:
        st.warning("로그인 후 세션·PDF·저장 기능을 사용할 수 있습니다.")

if _page == "signup":
    render_signup_page()
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("LLM 모델 선택")
    _model_choices = ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"]
    _sm = st.session_state.selected_model
    if _sm not in _model_choices:
        _sm = _model_choices[0]
        st.session_state.selected_model = _sm
    st.session_state.selected_model = st.selectbox(
        "모델",
        _model_choices,
        index=_model_choices.index(st.session_state.selected_model),
    )
    st.divider()
    st.session_state.enable_web_search = st.checkbox("인터넷 검색 보강 사용", value=st.session_state.enable_web_search)
    if st.session_state.enable_web_search:
        st.caption("PDF 외 질문도 웹 검색 결과를 반영해 답변합니다.")
    st.divider()
    st.subheader("세션 관리")
    if SUPABASE_ERROR:
        st.caption(f"Supabase 오류(참고): {SUPABASE_ERROR[:200]}")
    sb = get_supabase()
    sessions = get_sessions()
    options = ["(새 세션)"]
    option_map: Dict[str, str] = {}
    for s in sessions:
        label = f"{s.get('title') or '제목 없음'} ({str(s.get('updated_at') or '')[:16]})"
        options.append(label)
        option_map[label] = s["id"]

    selected_label = st.selectbox("세션 풀다운 메뉴", options, index=0)
    selected_session_id = option_map.get(selected_label)
    st.session_state.selected_session_id_for_button = selected_session_id

    if (
        selected_session_id
        and st.session_state.user_id
        and selected_session_id != st.session_state.last_dropdown_session_id
    ):
        if st.session_state.chat_history:
            save_session(st.session_state.current_session_id, clear_after_save=False)
        if load_session(selected_session_id):
            st.session_state.last_dropdown_session_id = selected_session_id
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("세션저장", use_container_width=True):
            if save_session(st.session_state.current_session_id, clear_after_save=True):
                st.success("세션 저장 완료, 화면을 초기화했습니다.")
                st.rerun()
    with c2:
        if st.button("세션로드", use_container_width=True, disabled=selected_session_id is None):
            if selected_session_id and load_session(selected_session_id):
                st.success("세션 로드 완료")
                st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("세션삭제", use_container_width=True, disabled=selected_session_id is None):
            if selected_session_id and delete_session(selected_session_id):
                st.success("세션 삭제 완료")
                st.rerun()
    with c4:
        if st.button("화면초기화", use_container_width=True):
            clear_screen_state()
            st.success("화면을 초기화했습니다.")
            st.rerun()

    if st.button("제목보정", use_container_width=True):
        if save_session(st.session_state.current_session_id, clear_after_save=False):
            st.success("현재 대화 기준으로 세션 제목을 보정했습니다.")

    if st.button("vectordb", use_container_width=True):
        if not sb or not st.session_state.user_id:
            st.warning("로그인이 필요합니다.")
        else:
            try:
                rows = (
                    sb.table("embeddings")
                    .select("file_name")
                    .eq("user_id", st.session_state.user_id)
                    .eq("session_id", st.session_state.current_session_id)
                    .execute()
                    .data
                    or []
                )
            except Exception as exc:
                set_supabase_error(exc)
                rows = []
            names = sorted(list({r.get("file_name") for r in rows if r.get("file_name")}))
            if names:
                st.info("\n".join(names))
            else:
                st.warning("현재 세션의 vectordb 파일이 없습니다.")

    st.divider()
    files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)
    if st.button("파일 처리하기", use_container_width=True, disabled=not files):
        if not st.session_state.user_id:
            st.warning("로그인 후 파일을 처리해주세요.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI API 키를 먼저 입력해주세요.")
        else:
            with st.spinner("PDF 처리 및 벡터 저장 중..."):
                docs: List[Document] = []
                for f in files or []:
                    tmp_path: Optional[str] = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(f.getbuffer())
                            tmp_path = tf.name
                        loaded = PyPDFLoader(tmp_path).load()
                        for d in loaded:
                            d.metadata["source"] = f.name
                        docs.extend(loaded)
                    finally:
                        if tmp_path:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                chunks = splitter.split_documents(docs)
                try:
                    save_documents(chunks, st.session_state.current_session_id)
                except Exception as exc:
                    st.error(f"벡터 저장 실패: {exc}")
                else:
                    st.session_state.processed_files = sorted(
                        list(set(st.session_state.processed_files + [f.name for f in files or []]))
                    )
                    restore_retriever(st.session_state.current_session_id)
                    save_session(st.session_state.current_session_id, clear_after_save=False)
                    st.success("파일 처리 및 자동 세션 저장 완료")

if (
    get_supabase()
    and st.session_state.user_id
    and not st.session_state.sessions_bootstrapped
):
    latest = get_sessions()
    if latest:
        load_session(latest[0]["id"])
        st.session_state.last_dropdown_session_id = latest[0]["id"]
    st.session_state.sessions_bootstrapped = True

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("질문을 입력하세요")
if prompt:
    if not st.session_state.user_id:
        st.warning("로그인 후 질문할 수 있습니다.")
        st.stop()
    if not llm_api_key_ready(st.session_state.selected_model):
        need = (
            "OpenAI"
            if st.session_state.selected_model == "gpt-5.5"
            else "Anthropic"
            if st.session_state.selected_model == "claude-opus-4-7"
            else "Gemini(Google)"
        )
        st.warning(f"사이드바에 선택한 모델에 맞는 API 키({need})를 입력하세요.")
        st.stop()
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs: List[Document] = []
    if st.session_state.retriever is not None:
        docs = st.session_state.retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs[:3]])
    mem = "\n".join(st.session_state.conversation_memory[-20:])

    web_context = ""
    if st.session_state.enable_web_search and should_use_web_search(prompt, docs):
        web_context = web_search_answer(prompt, st.session_state.conversation_memory)

    llm_prompt = (
        f"질문:{prompt}\n\n"
        f"PDF 문서 컨텍스트:\n{context if context else '(없음)'}\n\n"
        f"웹 검색 컨텍스트:\n{web_context if web_context else '(없음)'}\n\n"
        f"이전대화:\n{mem}\n\n"
        "답변 원칙:\n"
        "1) PDF 문서에 근거가 있으면 PDF 내용을 우선 사용하세요.\n"
        "2) PDF 근거가 없거나 최신성이 필요하면 웹 검색 컨텍스트를 사용하세요.\n"
        "3) 두 소스가 모두 없으면 일반 지식으로 답하되 불확실하면 모른다고 말하세요."
    )
    llm = build_llm(st.session_state.selected_model, streaming=True)
    with st.chat_message("assistant"):
        ph = st.empty()
        answer = ""
        for chunk in llm.stream(llm_prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            answer += text
            ph.markdown(answer + "▌")
        followups = generate_followups(prompt, answer, f"{context}\n{web_context}")
        if followups:
            answer += "\n\n### 향후 더 필요한 질문 3개\n"
            for i, q in enumerate(followups, start=1):
                answer += f"{i}. {q}\n"
        ph.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.conversation_memory.append(f"사용자: {prompt}")
    st.session_state.conversation_memory.append(f"AI: {answer}")
    save_session(st.session_state.current_session_id, clear_after_save=False)
