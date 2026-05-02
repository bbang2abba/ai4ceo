import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Any
from datetime import datetime
import logging
import re

# 환경 변수 로드
load_dotenv()

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"sql_agent_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# HTTP 요청 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# 구분선 및 취소선 제거 함수
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    # 취소선 마크다운 제거 (~~텍스트~~ -> 텍스트)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # 여러 줄에 걸친 구분선 제거 (공백 포함)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    # 단독 라인의 구분선 제거
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    # 연속된 빈 줄 정리 (최대 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)
    elif model_name == "claude-sonnet-4-5":
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    elif model_name == "gemini-3-pro-preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY가 환경변수에 설정되어 있지 않습니다.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=temperature)
    else:
        # 기본값: gpt-5.1
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)

# 페이지 설정
st.set_page_config(
    page_title="SQL Agent",
    page_icon="🗄️",
    layout="wide"
)

# 초기 상태 설정
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_connection" not in st.session_state:
    st.session_state.db_connection = None

if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = None

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

if "db_name" not in st.session_state:
    st.session_state.db_name = None

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
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

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
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

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
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
""", unsafe_allow_html=True)

# 로고 및 제목 영역
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_logo, col_title, col_empty = st.columns([1, 4, 1])

with col_logo:
    st.markdown("""
    <div style="margin-top: 0.5rem;">
        <div style="width: 120px; height: 120px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1f77b4 0%, #ffd700 100%); border-radius: 50%; margin: 0 auto;">
            <span style="color: white; font-size: 1.5rem; font-weight: bold;">SQL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">SQL</span> 
            <span style="color: #ffd700;">Agent</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.db_initialized and st.session_state.db_name:
    st.markdown(f"**{st.session_state.db_name}** 데이터베이스에 대해 자연어로 질문해보세요!")
else:
    st.markdown("데이터베이스에 대해 자연어로 질문해보세요!")

# 데이터베이스 초기화 함수
def initialize_database():
    """Chinook 데이터베이스 초기화"""
    try:
        # 여러 경로에서 chinook.db 파일 찾기
        current_dir = os.getcwd()
        possible_paths = [
            os.path.join(current_dir, "04.Langchain", "11.Agent", "chinook.db"),
            os.path.join(current_dir, "data", "Chinook.db"),
            os.path.join(current_dir, "data", "chinook.db"),
            os.path.join(current_dir, "chinook.db"),
        ]
        
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if not db_path:
            st.error(f"chinook.db 파일을 찾을 수 없습니다. 다음 경로를 확인했습니다: {possible_paths}")
            return False
        
        # 데이터베이스 이름 추출 (파일명에서 확장자 제거)
        db_name = os.path.splitext(os.path.basename(db_path))[0]
        st.session_state.db_name = db_name
        
        # 데이터베이스 연결
        db_uri = f"sqlite:///{db_path}"
        db_connection = SQLDatabase.from_uri(db_uri)
        st.session_state.db_connection = db_connection
        
        # LLM 초기화 (현재 선택된 모델 사용)
        llm = get_llm(st.session_state.llm_model, temperature=0.7)
        
        # SQL Agent 생성
        sql_agent = create_sql_agent(
            llm,
            db=db_connection,
            agent_type="openai-tools",
            verbose=False
        )
        st.session_state.sql_agent = sql_agent
        
        st.session_state.db_initialized = True
        logger.info(f"데이터베이스 초기화 완료: {db_path}")
        return True
        
    except Exception as e:
        st.error(f"데이터베이스 초기화 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"데이터베이스 초기화 오류: {e}")
        return False

# 사이드바 설정
with st.sidebar:
    # 1. LLM 모델 선택
    st.markdown('<h2 style="color: #1f77b4;">1. LLM 모델 선택</h2>', unsafe_allow_html=True)
    all_models = ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"]
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = all_models[0]
    
    try:
        current_index = all_models.index(st.session_state.llm_model)
    except ValueError:
        current_index = 0
    
    selected_model = st.radio(
        "사용할 언어모델을 선택하세요",
        options=all_models,
        index=current_index,
        key='llm_model_radio'
    )
    
    # 모델이 변경되면 데이터베이스 재초기화
    if st.session_state.llm_model != selected_model:
        st.session_state.llm_model = selected_model
        st.session_state.db_initialized = False
        st.session_state.sql_agent = None
    
    # 데이터베이스 초기화 버튼
    st.markdown('<h2 style="color: #ffd700;">2. 데이터베이스 연결</h2>', unsafe_allow_html=True)
    if not st.session_state.db_initialized:
        if st.button("데이터베이스 초기화", type="primary", use_container_width=True):
            with st.spinner("Chinook 데이터베이스에 연결 중..."):
                if initialize_database():
                    st.success("데이터베이스가 성공적으로 초기화되었습니다!")
                    st.rerun()
    else:
        if st.session_state.db_name:
            st.success(f"데이터베이스 연결됨: **{st.session_state.db_name}**")
        else:
            st.success("데이터베이스 연결됨")
        if st.button("재연결", use_container_width=True):
            st.session_state.db_initialized = False
            st.session_state.sql_agent = None
            st.session_state.db_name = None
            st.rerun()
    
    # 대화 초기화 버튼
    st.markdown('<h2 style="color: #ff69b4;">3. 대화 관리</h2>', unsafe_allow_html=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # 현재 설정 표시
    st.markdown('<h3 style="color: #1f77b4;">현재 설정</h3>', unsafe_allow_html=True)
    st.text(f"모델: {st.session_state.llm_model}")
    if st.session_state.db_initialized and st.session_state.db_name:
        st.text(f"데이터베이스: {st.session_state.db_name}")
    else:
        st.text(f"데이터베이스: {'연결됨' if st.session_state.db_initialized else '연결 안 됨'}")
    st.text(f"대화 기록: {len(st.session_state.chat_history)}개")

# 데이터베이스가 초기화되지 않은 경우 자동 초기화
if not st.session_state.db_initialized:
    with st.spinner("데이터베이스를 초기화하는 중..."):
        if initialize_database():
            st.success("데이터베이스가 성공적으로 초기화되었습니다!")
            st.rerun()

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("데이터베이스에 대해 질문하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if not st.session_state.db_initialized or st.session_state.sql_agent is None:
        with st.chat_message("assistant"):
            st.write("데이터베이스에 먼저 연결해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "데이터베이스에 먼저 연결해주세요."})
    else:
        # SQL Agent 실행 (progress 표시)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # SQL 생성 및 실행 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1단계: SQL 쿼리 생성 중
                status_text.text("SQL 쿼리 생성 중...")
                progress_bar.progress(20)
                
                # SQL Agent 실행
                status_text.text("SQL 쿼리 실행 중...")
                progress_bar.progress(50)
                
                agent_response = st.session_state.sql_agent.invoke({"input": prompt})
                agent_output = agent_response["output"]
                
                progress_bar.progress(80)
                status_text.text("답변 생성 중...")
                
                # 답변 정리
                response_text = remove_separators(agent_output)
                
                progress_bar.progress(90)
                status_text.text("다음 질문 추천 생성 중...")
                
                # 다음 질문 3개 생성 (데이터베이스 구조 기반)
                try:
                    # 데이터베이스 스키마 정보 가져오기
                    db_schema = st.session_state.db_connection.get_table_info()
                    
                    llm = get_llm(st.session_state.llm_model, temperature=1)
                    next_questions_prompt = f"""
                    질문자가 한 질문: {prompt}
                    
                    생성된 답변:
                    {response_text}
                    
                    데이터베이스 구조 정보:
                    {db_schema}
                    
                    위 정보를 바탕으로, 데이터베이스 구조를 분석하여 실제로 답변할 수 있는 질문 중에서 가장 적절한 3가지 질문을 선택해주세요.
                    
                    요구사항:
                    - 반드시 데이터베이스 구조에 존재하는 테이블과 컬럼만 사용하여 답변 가능한 질문이어야 합니다
                    - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                    - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                    - 데이터베이스 구조를 활용하여 실제로 SQL로 답변할 수 있는 질문
                    - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                    - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                    - 데이터베이스에 존재하지 않는 테이블이나 컬럼을 언급하는 질문은 생성하지 마세요
                    
                    형식:
                    질문1
                    질문2
                    질문3
                    
                    참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요. 반드시 데이터베이스 구조를 확인하여 실제로 답변 가능한 질문만 생성하세요.
                    """
                    next_questions_response = llm.invoke(next_questions_prompt).content
                    next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                    next_questions = next_questions[:3]
                    
                    progress_bar.progress(100)
                    status_text.text("완료!")
                    
                    if next_questions:
                        response_text += "\n\n"
                        response_text += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                        for i, question in enumerate(next_questions, 1):
                            response_text += f"{i}. {question}\n\n"
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")
                
                # 진행 상황 표시 제거
                progress_bar.empty()
                status_text.empty()
                
                # 최종 답변 표시
                response_placeholder.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                progress_bar.empty()
                status_text.empty()
                response_placeholder.write(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                logger.error(f"SQL Agent 실행 오류: {e}")

# 하단 정보
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>SQL Agent - Chinook 데이터베이스에 자연어로 질문하고 답변을 받아보세요</p>
</div>
""", unsafe_allow_html=True)

