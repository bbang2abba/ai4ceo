import os
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import ArxivLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

MODEL_OPTIONS = ["gpt-5.4", "gemini-3-pro-preview", "claude-opus-4-7"]


class ResearchState(TypedDict):
    topic: str
    agent1_research: str
    agent2_counter_research: str
    agent3_final_report: str
    research_history: List[str]


def ensure_session_state() -> None:
    defaults = {
        "selected_model": MODEL_OPTIONS[0],
        "research_state": None,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_llm_model(model_name: str, temperature: float = 0.4):
    if model_name == "gpt-5.4":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다.")
        return ChatOpenAI(model="gpt-5.4", api_key=api_key, temperature=temperature)
    if model_name == "gemini-3-pro-preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 필요합니다.")
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            google_api_key=api_key,
            temperature=temperature,
        )
    if model_name == "claude-opus-4-7":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(model="claude-opus-4-7", anthropic_api_key=api_key, temperature=temperature)
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def search_internet_with_gpt52_web_search(query: str) -> str:
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
        return response.output_text or "인터넷 검색 결과가 비어 있습니다."
    except Exception as exc:
        return f"인터넷 검색 중 오류 발생: {exc}"


def search_arxiv_tool(query: str, max_docs: int = 3) -> str:
    try:
        docs = ArxivLoader(query=query, load_max_docs=max_docs, load_all_available_meta=True).load()
    except Exception as exc:
        return f"Arxiv 검색 중 오류 발생: {exc}"

    if not docs:
        return "Arxiv에서 관련 논문을 찾지 못했습니다."

    lines: List[str] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        lines.append(
            "\n".join(
                [
                    f"[논문 {index}]",
                    f"제목: {meta.get('Title', '제목 없음')}",
                    f"저자: {meta.get('Authors', '저자 정보 없음')}",
                    f"발행일: {meta.get('Published', '날짜 정보 없음')}",
                    f"요약: {meta.get('Summary', doc.page_content[:600])[:700]}",
                ]
            )
        )
    return "\n\n".join(lines)


def agent1_research(state: ResearchState, llm, progress_bar, status_text) -> Dict[str, Any]:
    topic = state["topic"]
    progress_bar.progress(0.1, text="Agent 1: 인터넷 검색 중")
    status_text.text("Agent 1이 최신 웹 정보를 수집하고 있습니다.")
    internet_results = search_internet_with_gpt52_web_search(
        f"{topic} 핵심 동향, 최신 데이터, 실무 사례, 쟁점, 출처 링크를 조사해줘."
    )

    progress_bar.progress(0.25, text="Agent 1: Arxiv 검색 중")
    status_text.text("Agent 1이 Arxiv 논문을 찾고 있습니다.")
    arxiv_results = search_arxiv_tool(topic, max_docs=4)

    progress_bar.progress(0.33, text="Agent 1: 자료 종합 중")
    status_text.text("Agent 1이 수집 자료를 종합하고 있습니다.")
    prompt = f"""
주제: {topic}

역할: 너는 리서치 1번 에이전트다.
아래 인터넷/Arxiv 자료를 종합해서 '근거 중심 중립 보고서'를 작성해라.

[인터넷 검색 결과]
{internet_results}

[Arxiv 검색 결과]
{arxiv_results}

요구사항:
1) 핵심 사실/데이터/사례를 구조화
2) 출처를 본문에 명시
3) 주장과 근거를 분리
4) 한국어로 작성
"""
    result = llm.invoke(prompt)
    content = getattr(result, "content", str(result))
    return {
        **state,
        "agent1_research": content,
        "research_history": state["research_history"] + [f"Agent 1 완료 ({datetime.now().strftime('%H:%M:%S')})"],
    }


def agent2_counter_research(state: ResearchState, llm, progress_bar, status_text) -> Dict[str, Any]:
    topic = state["topic"]
    seed = state.get("agent1_research", "")[:1800]

    progress_bar.progress(0.45, text="Agent 2: 반대 관점 쿼리 설계")
    status_text.text("Agent 2가 반대 의견/대안 철학 검색 쿼리를 만들고 있습니다.")
    query_gen = llm.invoke(
        f"""
주제: {topic}
1번 에이전트 요약:
{seed}

다른 방식/철학/패러다임 중심의 반대 관점을 찾기 위한 검색 쿼리 3개를 줄바꿈으로 출력해라.
번호/설명 없이 쿼리만 출력.
"""
    )
    queries_raw = getattr(query_gen, "content", str(query_gen))
    queries = [q.strip("-• \t") for q in queries_raw.splitlines() if q.strip()][:3]
    if not queries:
        queries = [f"{topic} criticism", f"{topic} alternative approach", f"{topic} philosophical debate"]

    progress_bar.progress(0.58, text="Agent 2: 웹 반론 검색")
    status_text.text("Agent 2가 웹에서 반론 근거를 수집하고 있습니다.")
    internet_blocks = []
    for idx, query in enumerate(queries, start=1):
        block = search_internet_with_gpt52_web_search(f"{query} {topic}")
        internet_blocks.append(f"[쿼리 {idx}] {query}\n{block}")

    progress_bar.progress(0.66, text="Agent 2: Arxiv 대안 연구 검색")
    status_text.text("Agent 2가 Arxiv에서 대안 접근 논문을 찾고 있습니다.")
    arxiv_results = search_arxiv_tool(f"{topic} alternative method criticism comparative study", max_docs=3)

    progress_bar.progress(0.74, text="Agent 2: 반대 관점 종합")
    status_text.text("Agent 2가 반대 관점을 종합하고 있습니다.")
    prompt = f"""
주제: {topic}

역할: 너는 2번 에이전트다.
1번 에이전트 결과를 비판하고, 철학/방법론이 다른 대안을 제시하라.

[1번 에이전트 결과]
{state.get("agent1_research", "")}

[반대 관점 웹 자료]
{chr(10).join(internet_blocks)}

[반대 관점 Arxiv 자료]
{arxiv_results}

요구사항:
1) 핵심 반박 포인트 정리
2) 다른 접근 철학/방법론 제시
3) 반대 논리의 한계도 균형있게 기술
4) 출처 명시
5) 한국어 작성
"""
    result = llm.invoke(prompt)
    content = getattr(result, "content", str(result))
    return {
        **state,
        "agent2_counter_research": content,
        "research_history": state["research_history"] + [f"Agent 2 완료 ({datetime.now().strftime('%H:%M:%S')})"],
    }


def agent3_final_report(state: ResearchState, llm, progress_bar, status_text) -> Dict[str, Any]:
    progress_bar.progress(0.86, text="Agent 3: 최종 토론형 리포트 작성")
    status_text.text("Agent 3가 양측 주장을 비교하고 결론을 도출하고 있습니다.")

    prompt = f"""
주제: {state["topic"]}

[Agent 1 결과]
{state.get("agent1_research", "")}

[Agent 2 결과]
{state.get("agent2_counter_research", "")}

역할: 너는 3번 에이전트(총괄 분석가)다.
아래 순서를 반드시 지켜라.
1) 양측 핵심 주장 소개
2) 장점/단점/이슈 비교
3) 너의 결론
4) 향후 영향(산업/기술/사회), 새 트렌드
5) 추가 리서치 과제 제안

한국어로 작성하고, 균형 잡힌 관점을 유지하라.
"""
    result = llm.invoke(prompt)
    content = getattr(result, "content", str(result))
    progress_bar.progress(1.0, text="리서치 완료")
    status_text.text("3개 에이전트 리서치가 완료되었습니다.")
    return {
        **state,
        "agent3_final_report": content,
        "research_history": state["research_history"] + [f"Agent 3 완료 ({datetime.now().strftime('%H:%M:%S')})"],
    }


def run_research_pipeline(topic: str, model_name: str, progress_bar, status_text) -> ResearchState:
    llm = get_llm_model(model_name=model_name, temperature=0.4)
    state: ResearchState = {
        "topic": topic,
        "agent1_research": "",
        "agent2_counter_research": "",
        "agent3_final_report": "",
        "research_history": [],
    }
    state = agent1_research(state, llm, progress_bar, status_text)
    state = agent2_counter_research(state, llm, progress_bar, status_text)
    state = agent3_final_report(state, llm, progress_bar, status_text)
    return state


def render_style() -> None:
    st.markdown(
        """
<style>
h1, h2, h3 { font-weight: 700 !important; }
.stButton > button {
    border-radius: 10px !important;
    border: 0 !important;
    font-weight: 700 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
<div style="text-align:center; margin-top:-2rem; margin-bottom:1rem; padding:18px; border-radius:16px;
background: linear-gradient(90deg, rgba(255,215,0,0.15), rgba(69,183,209,0.15), rgba(255,105,180,0.15));">
  <h1 style="font-size:3rem; margin:0;
  background:linear-gradient(90deg,#ffd700,#45b7d1,#ff69b4);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    Deep Researcher
  </h1>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Deep Researcher", page_icon="🔎", layout="wide")
    ensure_session_state()
    render_style()
    render_header()

    with st.sidebar:
        st.subheader("설정")
        selected_model = st.selectbox("모델 선택", MODEL_OPTIONS, key="model_selector")
        st.session_state.selected_model = selected_model
        topic = st.text_area("리서치 주제", height=120, placeholder="예: 멀티에이전트 AI의 산업 적용과 리스크")

        if st.button("리서치 시작", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("리서치 주제를 입력해주세요.")
                st.stop()

            st.session_state.research_state = None
            st.session_state.chat_history = []

            progress_bar = st.progress(0.0, text="리서치를 시작합니다.")
            status_text = st.empty()
            try:
                st.session_state.research_state = run_research_pipeline(
                    topic.strip(), selected_model, progress_bar, status_text
                )
                st.success("리서치가 완료되었습니다.")
            except Exception as exc:
                st.error(f"리서치 중 오류가 발생했습니다: {exc}")

        st.markdown("---")
        st.caption("Agent 1: 자료 수집 (웹 + Arxiv)")
        st.caption("Agent 2: 반대/대안 관점 분석 (웹 + Arxiv)")
        st.caption("Agent 3: 종합 토론, 결론, 향후 전망")

    state = st.session_state.research_state
    if state:
        tab1, tab2, tab3, tab4 = st.tabs(["최종 Report", "Agent 1", "Agent 2", "진행 이력"])
        with tab1:
            st.markdown(state.get("agent3_final_report", ""))
            st.download_button(
                "최종 report 다운로드",
                data=state.get("agent3_final_report", ""),
                file_name=f"deep_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with tab2:
            st.markdown(state.get("agent1_research", ""))
        with tab3:
            st.markdown(state.get("agent2_counter_research", ""))
        with tab4:
            for item in state.get("research_history", []):
                st.write(f"- {item}")
    else:
        st.info("사이드바에서 주제를 입력하고 `리서치 시작`을 눌러주세요.")

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("리포트 토론")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_prompt := st.chat_input("리포트에 대해 추가 질문을 입력하세요"):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if not st.session_state.research_state:
            answer = "먼저 리서치를 완료해 주세요."
        else:
            try:
                llm = get_llm_model(st.session_state.selected_model, temperature=0.5)
                rs = st.session_state.research_state
                prompt = f"""
다음 리서치 결과를 기반으로 사용자 질문에 답변해라.

주제: {rs["topic"]}
Agent1: {rs.get("agent1_research", "")[:1500]}
Agent2: {rs.get("agent2_counter_research", "")[:1500]}
Final Report: {rs.get("agent3_final_report", "")[:2500]}

사용자 질문: {user_prompt}
"""
                response = llm.invoke(prompt)
                answer = getattr(response, "content", str(response))
            except Exception as exc:
                answer = f"답변 생성 중 오류가 발생했습니다: {exc}"

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

