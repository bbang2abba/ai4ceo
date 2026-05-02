import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from pydantic import Field, PrivateAttr
from supabase import Client, create_client

load_dotenv()


@st.cache_resource
def init_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


supabase = init_supabase()
SUPABASE_ERROR: Optional[str] = None


def set_supabase_error(exc: Exception) -> None:
    global SUPABASE_ERROR
    SUPABASE_ERROR = str(exc)


class SessionRetriever(BaseRetriever):
    k: int = Field(default=8)
    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _session_id: str = PrivateAttr()

    def __init__(self, supabase_client: Client, embeddings: OpenAIEmbeddings, session_id: str, k: int = 8):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._session_id = session_id

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        query_embedding = self._embeddings.embed_query(query)
        result = self._supabase.rpc(
            "match_session_documents",
            {
                "target_session_id": self._session_id,
                "query_embedding": query_embedding,
                "match_threshold": 0.55,
                "match_count": self.k,
            },
        ).execute()
        docs: List[Document] = []
        for row in (result.data or []):
            docs.append(Document(page_content=row.get("content", ""), metadata=row.get("metadata") or {}))
        return docs


def kst_now_str() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%m/%d %H:%M")


def get_embeddings() -> OpenAIEmbeddings:
    # SQL schema uses vector(1536), so keep embedding dimension fixed.
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
        return ChatAnthropic(model="claude-opus-4-7", temperature=1, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), streaming=streaming)
    return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=1, google_api_key=os.getenv("GOOGLE_API_KEY"), streaming=streaming)


def ensure_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "processed_files": [],
        "current_session_id": str(uuid.uuid4()),
        "selected_model": "gpt-5.5",
        "retriever": None,
        "sessions_bootstrapped": False,
        "last_dropdown_session_id": None,
        "selected_session_id_for_button": None,
        "enable_web_search": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_screen_state() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.retriever = None


def get_sessions() -> List[Dict[str, Any]]:
    if not supabase:
        return []
    try:
        res = supabase.table("sessions").select("id,title,created_at,updated_at").order("updated_at", desc=True).limit(200).execute()
        return res.data or []
    except Exception as exc:
        set_supabase_error(exc)
        return []


def generate_session_title() -> str:
    user_msgs = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
    ai_msgs = [m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"]
    if not user_msgs or not ai_msgs:
        return f"세션 {kst_now_str()}"
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


def upsert_session(session_id: str, title: str) -> None:
    try:
        supabase.table("sessions").upsert({"id": session_id, "title": title}, on_conflict="id").execute()
    except Exception as exc:
        set_supabase_error(exc)
        raise


def save_messages(session_id: str) -> None:
    try:
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        rows = [{"session_id": session_id, "seq": i, "role": m["role"], "content": m["content"]} for i, m in enumerate(st.session_state.chat_history)]
        if rows:
            supabase.table("messages").insert(rows).execute()
    except Exception as exc:
        set_supabase_error(exc)
        raise


def link_document_to_session(session_id: str, document_id: int, source: str) -> None:
    try:
        supabase.table("session_documents").upsert(
            {"session_id": session_id, "document_id": document_id, "source_name": source},
            on_conflict="session_id,document_id",
        ).execute()
    except Exception as exc:
        set_supabase_error(exc)
        raise


def save_documents(chunks: List[Document], session_id: str) -> None:
    if not supabase:
        raise ValueError("Supabase 연결이 필요합니다.")

    # session_documents has FK to sessions.id, so ensure parent row exists first.
    upsert_session(session_id, f"세션 {kst_now_str()}")

    embeddings = get_embeddings()
    ensure_embedding_dimension(embeddings, 1536)
    for chunk in chunks:
        content = chunk.page_content.strip()
        if not content:
            continue
        source = str((chunk.metadata or {}).get("source", "unknown.pdf"))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        try:
            found = supabase.table("documents").select("id").eq("content_hash", content_hash).limit(1).execute().data
        except Exception as exc:
            set_supabase_error(exc)
            raise
        if found:
            link_document_to_session(session_id, found[0]["id"], source)
            continue

        emb = embeddings.embed_query(content)
        try:
            inserted = supabase.table("documents").insert(
                {
                    "content": content,
                    "metadata": {"source": source},
                    "content_hash": content_hash,
                    "embedding": emb,
                }
            ).execute().data
        except Exception as exc:
            set_supabase_error(exc)
            raise
        if inserted:
            link_document_to_session(session_id, inserted[0]["id"], source)


def save_session(session_id: str, clear_after_save: bool = False) -> bool:
    if not supabase:
        st.error("Supabase 연결이 필요합니다.")
        return False
    try:
        title = generate_session_title()
        upsert_session(session_id, title)
        save_messages(session_id)
    except Exception:
        st.error("세션 저장 실패: Supabase 키 또는 권한을 확인해주세요.")
        return False
    if clear_after_save:
        clear_screen_state()
        st.session_state.current_session_id = str(uuid.uuid4())
    return True


def restore_retriever(session_id: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.retriever = None
        return
    try:
        st.session_state.retriever = SessionRetriever(supabase, get_embeddings(), session_id, 8)
    except Exception as exc:
        set_supabase_error(exc)
        st.session_state.retriever = None


def load_session(session_id: str) -> bool:
    if not supabase:
        return False
    try:
        res = supabase.table("messages").select("seq,role,content").eq("session_id", session_id).order("seq").execute()
    except Exception as exc:
        set_supabase_error(exc)
        st.error("세션 로드 실패: Supabase 키 또는 권한을 확인해주세요.")
        return False
    msgs = res.data or []
    clear_screen_state()
    st.session_state.chat_history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.session_state.conversation_memory.append(f"사용자: {m['content']}")
        elif m["role"] == "assistant":
            st.session_state.conversation_memory.append(f"AI: {m['content']}")
    try:
        docs = supabase.table("session_documents").select("source_name").eq("session_id", session_id).execute().data or []
    except Exception as exc:
        set_supabase_error(exc)
        docs = []
    st.session_state.processed_files = sorted(list({d["source_name"] for d in docs if d.get("source_name")}))
    restore_retriever(session_id)
    st.session_state.current_session_id = session_id
    return True


def delete_session(session_id: str) -> bool:
    if not supabase:
        return False
    try:
        supabase.table("sessions").delete().eq("id", session_id).execute()
    except Exception as exc:
        set_supabase_error(exc)
        st.error("세션 삭제 실패: Supabase 키 또는 권한을 확인해주세요.")
        return False
    if st.session_state.current_session_id == session_id:
        clear_screen_state()
        st.session_state.current_session_id = str(uuid.uuid4())
    return True


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
    # 최신성/외부 정보가 필요할 가능성이 높은 질문은 웹 검색 우선 보강
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


st.set_page_config(page_title="PDF 기반 멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
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
        <span style="color: #ffd700;">멀티세션</span>
        <span style="color: #d62728;">RAG 챗봇</span>
    </h1>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("LLM 모델 선택")
    st.session_state.selected_model = st.selectbox(
        "모델",
        ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"],
        index=["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"].index(st.session_state.selected_model),
    )
    st.divider()
    st.session_state.enable_web_search = st.checkbox("인터넷 검색 보강 사용", value=st.session_state.enable_web_search)
    if st.session_state.enable_web_search:
        st.caption("PDF 외 질문도 웹 검색 결과를 반영해 답변합니다.")
    st.divider()
    st.subheader("세션 관리")
    if SUPABASE_ERROR:
        st.error("Supabase API 오류: .env의 SUPABASE_ANON_KEY 또는 SUPABASE_SERVICE_ROLE_KEY를 확인하세요.")
    sessions = get_sessions()
    option_map: Dict[str, str] = {}
    options = ["(새 세션)"]
    for s in sessions:
        t = s.get("title") or "제목 없음"
        options.append(f"{t} ({s.get('updated_at', '')[:16]})")
        option_map[options[-1]] = s["id"]

    selected_label = st.selectbox("세션 풀다운 메뉴", options, index=0)
    selected_session_id = option_map.get(selected_label)
    st.session_state.selected_session_id_for_button = selected_session_id

    # 풀다운 선택 즉시 자동 로드
    if selected_session_id and selected_session_id != st.session_state.last_dropdown_session_id:
        if st.session_state.chat_history:
            save_session(st.session_state.current_session_id, clear_after_save=False)
        load_session(selected_session_id)
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
            if selected_session_id:
                load_session(selected_session_id)
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

    if st.button("vectordb", use_container_width=True):
        sid = st.session_state.current_session_id
        try:
            rows = supabase.table("session_documents").select("source_name").eq("session_id", sid).execute().data or []
        except Exception as exc:
            set_supabase_error(exc)
            rows = []
        names = sorted(list({r["source_name"] for r in rows if r.get("source_name")}))
        if names:
            st.info("\n".join(names))
        else:
            st.warning("현재 세션의 vectordb 파일이 없습니다.")

    st.divider()
    files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)
    if st.button("파일 처리하기", use_container_width=True, disabled=not files):
        with st.spinner("PDF 처리 및 벡터 저장 중..."):
            docs: List[Document] = []
            for f in files or []:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                    tf.write(f.getbuffer())
                    path = tf.name
                loaded = PyPDFLoader(path).load()
                for d in loaded:
                    d.metadata["source"] = f.name
                docs.extend(loaded)
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
            chunks = splitter.split_documents(docs)
            save_documents(chunks, st.session_state.current_session_id)
            st.session_state.processed_files = sorted(list(set(st.session_state.processed_files + [f.name for f in files or []])))
            restore_retriever(st.session_state.current_session_id)
            save_session(st.session_state.current_session_id, clear_after_save=False)
            st.success("파일 처리 및 자동 세션 저장 완료")


if supabase and not st.session_state.sessions_bootstrapped:
    latest = get_sessions()
    if latest:
        load_session(latest[0]["id"])
    st.session_state.sessions_bootstrapped = True


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("질문을 입력하세요")
if prompt:
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
