import os
import re
import tempfile
import logging
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv, dotenv_values
from google import genai

MODEL_NAME = "gemini-pro-latest"
QUOTA_ERROR_HINT = (
    "Gemini API 할당량(Quota)을 초과했습니다. "
    "잠시 후 다시 시도하거나, Google AI Studio/GCP에서 과금 및 쿼터 설정을 확인해주세요."
)


load_dotenv()


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"google_rag_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_google_api_key() -> str:
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file)
    env_path = os.path.join(project_root, ".env")

    if os.path.exists(env_path):
        env_vars = dotenv_values(env_path)
        api_key = env_vars.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")

    return api_key or ""


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


def build_system_instruction() -> str:
    return """당신은 전문적인 AI 어시스턴트입니다. 답변을 전문적으로 해줘.

답변 형식:
- 답변은 반드시 제목과 본문으로 구분하여 작성하세요
- 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
- 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
- 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
- 본문은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요
- 파일에 없는 내용은 추측하지 말고, 불확실하면 불확실하다고 명시하세요
- 답변 마지막에는 반드시 '## 출처' 섹션을 만들고 실제로 참고한 파일명만 bullet로 적으세요"""


def build_sources_section(file_names: List[str]) -> str:
    if not file_names:
        return "\n\n## 출처\n- 업로드된 파일 없음"
    lines = "\n".join([f"- {name}" for name in file_names])
    return f"\n\n## 출처\n{lines}"


def format_gemini_error(error: Exception) -> str:
    error_text = str(error)
    lowered = error_text.lower()
    is_quota_error = (
        "resource_exhausted" in lowered
        or "quota exceeded" in lowered
        or "too many requests" in lowered
        or "429" in lowered
    )
    if is_quota_error:
        return f"{QUOTA_ERROR_HINT}\n\n원본 오류: {error_text}"
    return error_text


def upload_pdf_files(client: genai.Client, uploaded_files) -> List[object]:
    uploaded_gemini_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            gemini_file = client.files.upload(file=temp_file_path)
            uploaded_gemini_files.append(gemini_file)

    return uploaded_gemini_files


def answer_with_file_search(client: genai.Client, prompt: str, history: list, file_names: List[str]) -> str:
    history_text = ""
    if history:
        recent_history = history[-6:] if len(history) > 6 else history
        history_text = "\n\n이전 대화 맥락:\n"
        for msg in recent_history:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            history_text += f"{role}: {msg['content']}\n"

    instruction = build_system_instruction()
    user_prompt = (
        f"{instruction}\n\n"
        f"참고 파일 목록: {', '.join(file_names)}\n"
        f"{history_text}\n"
        f"현재 질문: {prompt}\n\n"
        "업로드된 PDF 파일들을 바탕으로 답변해주세요."
    )

    try:
        files_for_context = st.session_state.get("gemini_files", [])
        contents = files_for_context + [user_prompt]
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )
        return response.text or "응답을 생성하지 못했습니다."
    except Exception as e:
        logger.error(f"Gemini File Search 응답 생성 오류: {e}")
        return f"Gemini File Search 응답 생성 중 오류: {format_gemini_error(e)}"


def generate_next_questions(client: genai.Client, prompt: str, answer_text: str) -> List[str]:
    next_questions_prompt = f"""
질문자가 한 질문: {prompt}

생성된 답변:
{answer_text}

위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.

요구사항:
- 답변 내용을 더 깊이 이해하기 위한 후속 질문
- 답변에서 언급된 내용을 구체화하거나 확장하는 질문
- 관련된 다른 주제나 관점을 탐색할 수 있는 질문
- 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
- 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성

형식:
질문1
질문2
질문3

참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[next_questions_prompt],
        )
        text = response.text or ""
        questions = [q.strip() for q in text.split("\n") if q.strip() and not q.strip().startswith("#")]
        return questions[:3]
    except Exception as e:
        logger.warning(f"다음 질문 생성 실패: {format_gemini_error(e)}")
        return []


st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "gemini_files" not in st.session_state:
    st.session_state.gemini_files = []

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage p, .stChatMessage ul, .stChatMessage ol, .stChatMessage li {
    font-size: 0.95rem !important; line-height: 1.5 !important;
}
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

st.markdown(
    """
<div style="margin-top: -3rem; margin-bottom: 1rem;">
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">RAG</span>
            <span style="color: #ffd700;">챗봇</span>
        </h1>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("📚 **Gemini File Search 모드**: PDF 파일을 업로드해 Gemini 내장 검색으로 답변합니다.")

google_api_key = get_google_api_key()
if not google_api_key:
    st.error("`.env` 파일에 `GOOGLE_API_KEY`를 설정해주세요.")
    st.stop()

client = genai.Client(api_key=google_api_key)

with st.sidebar:
    st.markdown('<h2 style="color: #ff69b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("파일 처리하기"):
        with st.spinner("PDF 파일을 Gemini에 업로드 중입니다..."):
            try:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                if not new_files:
                    st.success("모든 파일이 이미 처리되었습니다.")
                else:
                    uploaded = upload_pdf_files(client, new_files)
                    st.session_state.gemini_files.extend(uploaded)
                    st.session_state.processed_files.extend([f.name for f in new_files])
                    st.success(f"✅ {len(new_files)}개 파일 처리 완료!")
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
                logger.error(f"PDF 업로드 오류: {e}")

    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file_name in st.session_state.processed_files:
            st.write(f"- {file_name}")

    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<h3 style="color: #1f77b4;">현재 설정</h3>', unsafe_allow_html=True)
    st.text("모드: Gemini File Search")
    st.text(f"모델: {MODEL_NAME}")
    st.text(f"처리된 파일: {len(st.session_state.processed_files)}개")
    st.text(f"대화 기록: {len(st.session_state.chat_history)}개")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if not st.session_state.gemini_files:
        warning_msg = "📚 먼저 PDF 파일을 업로드하고 '파일 처리하기'를 눌러주세요."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": warning_msg})
    else:
        with st.spinner("Gemini가 파일을 검색해 답변을 생성 중입니다..."):
            history = [m for m in st.session_state.chat_history[:-1] if m["role"] in ("user", "assistant")]
            answer_text = answer_with_file_search(
                client=client,
                prompt=prompt,
                history=history,
                file_names=st.session_state.processed_files,
            )
            answer_text = remove_separators(answer_text)
            if "## 출처" not in answer_text:
                answer_text += build_sources_section(st.session_state.processed_files)

            next_questions = generate_next_questions(client, prompt, answer_text)
            if next_questions:
                answer_text += "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n"
                for i, q in enumerate(next_questions, 1):
                    answer_text += f"{i}. {q}\n\n"

            with st.chat_message("assistant"):
                st.markdown(answer_text)

            st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
