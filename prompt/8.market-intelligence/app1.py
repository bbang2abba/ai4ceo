"""Samsung Global Market Intelligence Streamlit app."""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


MODEL_NAME = "gpt-5.2"
MAX_RECORDS = 10
APP_TITLE = "삼성전자 Global Market Intelligence"


def _load_env() -> None:
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_openai_api_key() -> Optional[str]:
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        key = dotenv_values(env_path).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY")
    return key.strip() if key else None


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\n\s*-{3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*={3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*_{3,}\s*\n", "\n\n", text)
    text = re.sub(r"^\s*[-=_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_brand_style() -> None:
    st.markdown(
        """
<style>
    .stApp { background: linear-gradient(135deg, #001a3a 0%, #003b7a 100%); }
    .stSidebar { background-color: #001a3a !important; }
    h1, h2, h3, h4, h5, h6, p, div, label, span { color: #FFFFFF !important; }
    .stButton > button {
        background-color: #004C97 !important;
        color: #FFFFFF !important;
        border: 1px solid #7fb5ff !important;
        width: 100%;
        font-weight: 700;
    }
    .stButton > button:hover { background-color: #0060c0 !important; }
    .stDataFrame, .stAlert { background-color: rgba(0, 0, 0, 0.25) !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def call_web_search(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    api_key = get_openai_api_key()
    if not api_key:
        return None, "OPENAI_API_KEY를 `.env` 파일에 설정해주세요."
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            tools=[{"type": "web_search", "search_context_size": "high"}],
        )
        return response.output_text, None
    except Exception as exc:
        return None, f"web_search 호출 오류: {exc}"


def collect_market_data(topic: str, progress, status_box) -> Tuple[Optional[str], Optional[str]]:
    status_box.text("검색 준비 중...")
    progress.progress(10)
    prompt = f"""
당신은 삼성전자 글로벌 마켓 인텔리전스 리서처입니다.
주제: {topic}

반드시 아래 사이트군에서 근거를 찾고, 서로 다른 소스를 섞어 정확히 10건만 수집해주세요.
- IT 전문 매체: The Verge, CNET, Wired, TechRadar, Tom's Guide
- 커뮤니티/SNS: Reddit(r/Samsung, r/Android), X 트렌드, YouTube 테크 인플루언서(MKBHD, Mrwhosetheboss 등)
- 지역 뉴스: Reuters, 동남아/유럽 현지 테크 블로그

반환 형식: 아래 JSON 배열만 반환. 다른 텍스트 금지.
[
  {{
    "media": "매체명",
    "country": "국가/지역",
    "date": "YYYY-MM-DD 또는 Recent",
    "pros": ["혁신성/휴대성/멀티태스킹 관련 장점"],
    "cons": ["내구성/가격/배터리/무게 관련 단점"],
    "comparison": "갤럭시 폴드 시리즈 및 경쟁사(화웨이 등)와 비교",
    "insights": "마케팅 소구점과 차기 모델 수정 제안",
    "website_url": "https://..."
  }}
]
"""
    progress.progress(40)
    status_box.text("글로벌 데이터 수집 중...")
    raw, err = call_web_search(prompt)
    progress.progress(90)
    status_box.text("응답 정리 중...")
    progress.progress(100)
    status_box.text("완료")
    return raw, err


def _try_json_parse(raw_text: str) -> List[Dict]:
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parse_records(raw_text: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for item in _try_json_parse(raw_text)[:MAX_RECORDS]:
        rows.append(
            {
                "매체명": item.get("media", "Unknown"),
                "국가": item.get("country", "Unknown"),
                "리뷰 날짜": item.get("date", "Recent"),
                "긍정 요소": "; ".join(item.get("pros", [])) if isinstance(item.get("pros"), list) else str(item.get("pros", "")),
                "부정 요소": "; ".join(item.get("cons", [])) if isinstance(item.get("cons"), list) else str(item.get("cons", "")),
                "비교 분석": item.get("comparison", ""),
                "전략적 인사이트": item.get("insights", ""),
                "Website URL": item.get("website_url", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_sentiment(df: pd.DataFrame, topic: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    key = get_openai_api_key()
    if not key:
        return None, None, None, None
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.6, api_key=key)

    corpus = "\n".join(
        f"[{r['매체명']}] 국가={r['국가']} | Pros={r['긍정 요소']} | Cons={r['부정 요소']}"
        for _, r in df.iterrows()
    )
    regional = df.groupby("국가").size().to_dict() if not df.empty else {}

    pros = llm.invoke(f"{topic} 장점 4개 이내 요약. 각 문장에 출처(매체명) 포함.\n{corpus}").content
    cons = llm.invoke(f"{topic} 단점 4개 이내 요약. 각 문장에 출처(매체명) 포함.\n{corpus}").content
    durability = llm.invoke(f"{topic}에서 내구성 이슈만 요약하고 출처(매체명) 포함.\n{corpus}").content
    price_region = llm.invoke(
        f"{topic} 가격 반응과 지역별 반응 차이를 함께 요약. 출처(매체명) 포함.\n국가 분포={regional}\n{corpus}"
    ).content
    return (
        remove_separators(pros),
        remove_separators(cons),
        remove_separators(durability),
        remove_separators(price_region),
    )


def generate_exec_report(df: pd.DataFrame, topic: str) -> Optional[str]:
    key = get_openai_api_key()
    if not key:
        return None
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=key)
    brief = "\n".join(
        f"[{r['매체명']}] 국가={r['국가']} Pros={r['긍정 요소']} Cons={r['부정 요소']} 비교={r['비교 분석']} 인사이트={r['전략적 인사이트']}"
        for _, r in df.iterrows()
    )
    prompt = f"""
당신은 삼성전자 DX 부문 임원 보고용 전략 분석가입니다.
주제: {topic}
데이터:
{brief}

아래 형식으로 한국어 보고서를 작성하세요.
1) 시장의 열광 포인트
2) 즉시 대응이 필요한 비판적 여론
3) 경쟁사 대비 우위 요소
4) 마케팅팀 핵심 광고 카피 3개
5) DX 부문 임원 follow-up 이슈 3개

요구: 제목+본문 구조, 서술형, 실행 가능한 문장.
"""
    result = llm.invoke(prompt).content
    return remove_separators(result)


def init_state() -> None:
    if "research_topic" not in st.session_state:
        st.session_state.research_topic = ""
    if "reviews_df" not in st.session_state:
        st.session_state.reviews_df = pd.DataFrame()
    if "sentiment" not in st.session_state:
        st.session_state.sentiment = None
    if "report" not in st.session_state:
        st.session_state.report = None


def main() -> None:
    _load_env()
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    apply_brand_style()
    init_state()

    st.title(APP_TITLE)
    st.caption("gpt-5.2 + web_search 기반 글로벌 마켓 센싱")

    with st.sidebar:
        st.subheader("마켓 리서치 제목")
        with st.form("market_intel_controls", clear_on_submit=False):
            topic = st.text_input(
                "토픽 입력",
                value=st.session_state.research_topic,
                placeholder="예: Galaxy Z Fold 신제품 초기 반응",
            )
            collect_btn = st.form_submit_button("검색과 데이터수집")
            analyze_btn = st.form_submit_button("감성분석")
            report_btn = st.form_submit_button("임원보고서")

        if topic != st.session_state.research_topic:
            st.session_state.research_topic = topic
            st.session_state.reviews_df = pd.DataFrame()
            st.session_state.sentiment = None
            st.session_state.report = None

    if collect_btn:
        current_topic = topic.strip()
        if current_topic and current_topic != st.session_state.research_topic:
            st.session_state.research_topic = current_topic
        if not current_topic:
            st.error("먼저 마켓 리서치 제목을 입력해주세요.")
        else:
            progress = st.progress(0)
            status = st.empty()
            raw, err = collect_market_data(current_topic, progress, status)
            if err:
                st.error(err)
            elif not raw:
                st.error("검색 결과가 비어 있습니다.")
            else:
                df = parse_records(raw)
                if df.empty:
                    st.error("JSON 파싱 실패: 모델 응답 형식을 확인해주세요.")
                else:
                    st.session_state.reviews_df = df
                    st.success(f"{len(df)}개 자료를 수집했습니다.")

    if not st.session_state.reviews_df.empty:
        st.subheader("글로벌 실시간 검색 결과")
        st.dataframe(st.session_state.reviews_df, hide_index=True, width="stretch")

    if analyze_btn:
        if st.session_state.reviews_df.empty:
            st.warning("먼저 '검색과 데이터수집'을 실행해주세요.")
        else:
            with st.spinner("감성 분석 중..."):
                st.session_state.sentiment = summarize_sentiment(
                    st.session_state.reviews_df, st.session_state.research_topic
                )

    if st.session_state.sentiment:
        pros, cons, durability, price_region = st.session_state.sentiment
        st.subheader("감성 분석 및 시각화")
        if pros:
            st.markdown("### 주요 장점")
            st.markdown(pros)
        if cons:
            st.markdown("### 주요 단점")
            st.markdown(cons)
        if durability:
            st.markdown("### 내구성 언급 요약")
            st.markdown(durability)
        if price_region:
            st.markdown("### 가격/지역 반응 요약")
            st.markdown(price_region)

    if report_btn:
        if st.session_state.reviews_df.empty:
            st.warning("먼저 데이터 수집을 실행해주세요.")
        else:
            with st.spinner("임원 보고서 생성 중..."):
                st.session_state.report = generate_exec_report(
                    st.session_state.reviews_df, st.session_state.research_topic
                )

    if st.session_state.report:
        st.subheader("임원용 전략 보고서")
        st.markdown(st.session_state.report)
        safe_topic = st.session_state.research_topic.replace(" ", "_") or "market-intel"
        st.download_button(
            "보고서 다운로드 (TXT)",
            data=st.session_state.report,
            file_name=f"{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
