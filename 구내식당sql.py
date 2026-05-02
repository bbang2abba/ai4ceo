"""
Streamlit SQL Agent: 구내식당.db 자연어 질의 (sql-agent.py 로직 재사용).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st

# 프로젝트 루트의 sql-agent.py 로드 (파일명에 하이픈이 있어 importlib 사용)
_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("sql_agent_module", _ROOT / "sql-agent.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("sql-agent.py 를 불러올 수 없습니다.")
_sa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_sa)

remove_separators = _sa.remove_separators
get_llm = _sa.get_llm
build_sql_agent_executor = _sa.build_sql_agent_executor
SQLProgressCallback = _sa.SQLProgressCallback
initialize_database = _sa.initialize_database
reset_connection = _sa.reset_connection
generate_followup_questions = _sa.generate_followup_questions
init_state = _sa.init_state
LOGGER = _sa.LOGGER
LLM_MODEL = _sa.LLM_MODEL

APP_TITLE = "구내식당 SQL Agent"
DB_FILENAME = "구내식당.db"
# read_csv_sample.py 기준 테이블명·컬럼(일자, 요일, 인원·승인·재택, 메뉴, 중식계, 석식계)
TABLE_HINT = "구내식당train"

# 사이드바 질문 예시 (실제 컬럼명에 맞춤)
EXAMPLE_QUESTIONS: List[str] = [
    "2016년 월별로 중식계와 석식계 합계를 집계해 주세요.",
    "요일별 평균 중식계와 평균 석식계를 비교해 주세요.",
    "중식계가 가장 많았던 날 상위 10건과 해당 일자의 요일·석식계를 함께 알려 주세요.",
    "본사휴가자수와 본사출장자수의 합이 큰 날 상위 10건을 날짜순으로 보여 주세요.",
    "본사시간외근무명령서승인건수가 높은 상위 10일의 중식계·석식계를 함께 알려 주세요.",
    "일자별로 중식계 대비 석식계 비율을 구하고, 비율이 높은 날 상위 10건을 보여 주세요.",
]

# 분석가 관점의 정적 인사이트 가이드 (데이터 설명용)
ANALYST_INSIGHT_GUIDE = """
- **수요·리듬**: 요일·월별 중식계·석식계 추이로 식수 패턴과 수요 피크를 파악할 수 있습니다.
- **인력 가용성**: 본사정원수 대비 본사휴가자수·본사출장자수·재택근무(현본사소속재택근무자수)와 식수의 관계를 볼 수 있습니다.
- **업무 강도 프록시**: 본사시간외근무명령서승인건수와 식수의 동행 여부를 점검할 수 있습니다(상관은 원인 단정이 아님).
- **메뉴 텍스트**: 조식·중식·석식 메뉴 문자열은 키워드 빈도·계절 메뉴 등 탐색형 분석에 활용할 수 있습니다.
- **운영 의사결정**: 월별 합계·요일별 평균은 식재·인력 배치 시나리오의 출발점이 됩니다.
"""


def candidate_gunaesik_db_paths() -> List[Path]:
    root = Path(__file__).resolve().parent
    cwd = Path.cwd()
    return [
        root / DB_FILENAME,
        root / "prompt" / "5.SQL-Agent" / DB_FILENAME,
        cwd / DB_FILENAME,
        cwd / "prompt" / "5.SQL-Agent" / DB_FILENAME,
        root / "data" / DB_FILENAME,
        cwd / "data" / DB_FILENAME,
    ]


def locate_gunaesik_db() -> Optional[Path]:
    for candidate in candidate_gunaesik_db_paths():
        if candidate.exists():
            return candidate
    return None


def apply_ref_style() -> None:
    """ref.py 와 동일한 헤딩·채팅·버튼 스타일."""
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
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}
.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}
.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def generate_insight_summary(user_prompt: str, answer_text: str, db_schema: str) -> str:
    """답변을 바탕으로 데이터 기반 핵심 인사이트를 생성합니다."""
    try:
        llm = get_llm(temperature=0.35)
        prompt = f"""사용자 질문:
{user_prompt}

에이전트 답변(요약·수치 포함):
{answer_text[:10000]}

DB 스키마 요약(참고):
{db_schema[:6000]}

위 답변에 근거하여, 구내식당·인원·식단 운영 관점에서 활용할 수 있는 핵심 인사이트만 3~5개 작성하세요.

규칙:
- 각 인사이트는 한 문장, 한국어 존댓말
- 번호 목록 형식: 1. 2. 3. (각 줄에 하나)
- 답변에 근거가 없는 추측·단정은 하지 말 것
- SQL이나 테이블명을 나열하지 말고 의미 위주로 작성
"""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        lines: List[str] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
            if line:
                lines.append(line)
        if not lines:
            return ""
        return "\n".join(f"{i}. {t}" for i, t in enumerate(lines[:5], start=1))
    except Exception as exc:
        LOGGER.warning("Insight summary failed: %s", exc)
        return ""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")
    init_state()
    apply_ref_style()

    st.markdown(
        """
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""",
        unsafe_allow_html=True,
    )
    col_title, col_empty = st.columns([4, 1])
    with col_title:
        st.markdown(
            """
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">구내식당</span>
            <span style="color: #ffd700;">SQL</span>
        </h1>
    </div>
            """,
            unsafe_allow_html=True,
        )
    with col_empty:
        st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "🍱 **구내식당** DB에 자연어로 질문하면 SQL을 생성·실행한 뒤 결과를 설명하고, **핵심 인사이트**와 **추천 후속 질문**을 드립니다."
    )

    if not st.session_state.db_initialized:
        auto_db = locate_gunaesik_db()
        if auto_db:
            with st.spinner(f"`{auto_db.name}` 연결 중..."):
                if initialize_database(auto_db):
                    st.rerun()

    if st.session_state.db_initialized:
        st.info(f"연결된 데이터베이스: **{st.session_state.db_name}** (`{st.session_state.db_path}`)")
    else:
        st.warning(
            f"`{DB_FILENAME}` 를 찾지 못했습니다. 프로젝트 루트에서 `python read_csv_sample.py`로 생성하거나, "
            f"`prompt/5.SQL-Agent/`·`data/` 등 경로에 파일을 두고 사이드바에서 재연결하세요."
        )

    with st.sidebar:
        st.markdown('<h2 style="color: #ffd700;">데이터베이스</h2>', unsafe_allow_html=True)
        if st.session_state.db_initialized:
            st.success(f"연결됨: **{st.session_state.db_name}**")
            st.caption(st.session_state.db_path)
        else:
            st.markdown("탐색 경로:")
            for p in candidate_gunaesik_db_paths():
                st.caption(str(p))

        if st.button(f"{DB_FILENAME} 다시 연결", type="primary", use_container_width=True):
            reset_connection()
            db = locate_gunaesik_db()
            if db is None:
                st.error(f"{DB_FILENAME} 을(를) 찾지 못했습니다.")
            elif initialize_database(db):
                st.rerun()

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<h2 style="color: #ff69b4;">질문 예시</h2>', unsafe_allow_html=True)
        st.caption("버튼을 누르면 해당 질문이 입력됩니다.")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            label = q[:40] + ("…" if len(q) > 40 else "")
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["_pending_prompt"] = q

        st.markdown('<h2 style="color: #1f77b4;">데이터 분석 인사이트 가이드</h2>', unsafe_allow_html=True)
        st.markdown(
            "아래는 이 데이터(주로 테이블 `" + TABLE_HINT + "`)로 **분석가가 시도할 수 있는 관점**입니다."
        )
        st.markdown(ANALYST_INSIGHT_GUIDE)

        st.markdown('<h3 style="color: #1f77b4;">현재 상태</h3>', unsafe_allow_html=True)
        st.text(f"DB: {st.session_state.db_name or '미연결'}")
        st.text(f"모델: {LLM_MODEL}")
        st.text(f"대화 수: {len(st.session_state.chat_history)}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt: Optional[str] = None
    if "_pending_prompt" in st.session_state:
        prompt = st.session_state.pop("_pending_prompt")
    elif user_input := st.chat_input("질문을 입력하세요. 예: 요일별 평균 중식계를 알려주세요."):
        prompt = user_input

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.db_initialized or st.session_state.sql_agent_executor is None:
            warning = (
                f"DB 연결이 필요합니다. `{DB_FILENAME}` 경로를 확인한 뒤 사이드바에서 다시 연결해 주세요."
            )
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

                status.text("인사이트 생성 중...")
                progress.progress(92)
                db_schema = st.session_state.db_connection.get_table_info()
                insight_text = generate_insight_summary(prompt, answer, db_schema)

                status.text("추천 질문 생성 중...")
                progress.progress(96)
                suggestions = generate_followup_questions(prompt, answer, db_schema)

                if insight_text:
                    answer += "\n\n### 핵심 인사이트\n\n" + insight_text + "\n"
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
