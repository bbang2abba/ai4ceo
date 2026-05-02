import os
import streamlit as st
import tempfile
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

log_filename = os.path.join(log_dir, f"sql_agent2_{datetime.now().strftime('%Y%m%d')}.log")
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

# CSV를 SQLite DB로 변환하는 함수
def csv_to_sqlite(csv_file, db_path: str) -> bool:
    """CSV 파일을 SQLite 데이터베이스로 변환합니다."""
    try:
        # 여러 인코딩으로 CSV 파일 읽기 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                # 파일을 처음부터 읽기 (BytesIO 객체인 경우 seek 필요)
                if hasattr(csv_file, 'seek'):
                    csv_file.seek(0)
                # 임시 파일로 저장 후 읽기 (인코딩 문제 해결)
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(csv_file.read())
                    tmp_path = tmp_file.name
                
                # 임시 파일에서 읽기
                df = pd.read_csv(tmp_path, encoding=encoding)
                used_encoding = encoding
                logger.info(f"CSV 파일을 {encoding} 인코딩으로 성공적으로 읽음")
                
                # 임시 파일 삭제
                os.unlink(tmp_path)
                break
            except Exception as e:
                logger.warning(f"{encoding} 인코딩 실패: {str(e)[:50]}")
                # 임시 파일이 남아있으면 삭제
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                continue
        
        if df is None:
            st.error("CSV 파일을 읽을 수 없습니다. 인코딩 문제일 수 있습니다.")
            logger.error("모든 인코딩 시도 실패")
            return False
        
        # 기존 DB 파일이 있으면 삭제
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # SQLite 연결
        conn = sqlite3.connect(db_path)
        
        # 테이블 이름은 'data'로 고정 (CSV 파일명이 복잡할 수 있으므로)
        table_name = 'data'
        
        # 데이터프레임을 SQLite 테이블로 저장
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
        logger.info(f"SQLite 데이터베이스 생성 완료: {db_path}, 테이블: {table_name}, 행 수: {len(df)}")
        return True
        
    except Exception as e:
        st.error(f"CSV를 데이터베이스로 변환하는 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"CSV 변환 오류: {e}")
        return False

# 페이지 설정
st.set_page_config(
    page_title="SQL Agent 2",
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

if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "db_info" not in st.session_state:
    st.session_state.db_info = None

if "db_description" not in st.session_state:
    st.session_state.db_description = None

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
    /* 배경색은 변경하지 않음 (기본값 유지) */
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    color: inherit !important; /* 부모의 색상 상속 */
}

/* assistant 메시지의 텍스트 색상 변경 (배경은 유지) */
.stChatMessage[data-testid="assistant"] {
    /* 배경색은 변경하지 않음 */
}

.stChatMessage[data-testid="assistant"] p,
.stChatMessage[data-testid="assistant"] li,
.stChatMessage[data-testid="assistant"] {
    color: #1f1f1f !important; /* 텍스트 색상만 변경 */
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

/* 코드 스타일 - 배경색은 변경하지 않음 */
.stChatMessage code {
    font-size: 0.9rem !important;
    /* background-color는 기본값 유지 (변경하지 않음) */
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
    color: #1f77b4 !important; /* 텍스트 색상만 변경 */
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
            <span style="color: #ffd700;">Agent 2</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# 기존 데이터베이스 연결 닫기 함수
def close_all_databases():
    """모든 열려있는 데이터베이스 연결을 닫습니다."""
    try:
        # SQLDatabase 연결 닫기
        if st.session_state.db_connection is not None:
            try:
                # SQLDatabase 객체는 내부적으로 연결을 관리하므로 명시적으로 닫을 수 없을 수 있음
                # 하지만 세션 상태를 초기화하면 연결이 해제됨
                logger.info("기존 데이터베이스 연결을 닫는 중...")
            except Exception as e:
                logger.warning(f"데이터베이스 연결 닫기 중 오류: {e}")
        
        # SQLite 직접 연결이 있다면 닫기
        # (현재는 SQLDatabase를 통해 관리되므로 별도 처리 불필요)
        
    except Exception as e:
        logger.error(f"데이터베이스 연결 닫기 오류: {e}")

# 데이터베이스 설명 생성 함수
def generate_database_description(db_info: dict, db_name: str) -> str:
    """데이터베이스 정보를 바탕으로 LLM을 사용하여 설명을 생성합니다."""
    try:
        if not db_info or not db_info.get("table_details"):
            return "데이터베이스 정보가 없습니다."
        
        table_details = db_info.get("table_details", [])
        total_rows = db_info.get("total_rows", 0)
        
        # 테이블 정보 요약
        tables_summary = []
        for table in table_details:
            tables_summary.append({
                "name": table["name"],
                "columns": table["columns"],
                "row_count": table["row_count"]
            })
        
        # LLM을 사용하여 설명 생성
        llm = get_llm(st.session_state.llm_model, temperature=0.7)
        
        prompt = f"""다음 데이터베이스 정보를 바탕으로 한국어로 간결하고 명확한 설명을 작성해주세요.

데이터베이스 이름: {db_name}
전체 행 수: {total_rows:,}개
테이블 수: {len(tables_summary)}개

테이블 정보:
"""
        for table in tables_summary:
            prompt += f"""
- 테이블명: {table['name']}
  - 행 수: {table['row_count']:,}개
  - 컬럼명: {', '.join(table['columns'])}
"""
        
        prompt += """
요구사항:
- 데이터베이스의 목적과 주요 내용을 설명하세요
- 주요 테이블과 컬럼에 대해 간략히 설명하세요
- 데이터 규모(행 수)를 언급하세요
- 간결하고 명확하게 작성하세요 (3-5문단 정도)
- 마크다운 형식으로 작성하세요 (# ## ### 사용 가능)
"""
        
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
        
    except Exception as e:
        logger.error(f"데이터베이스 설명 생성 오류: {e}")
        return f"데이터베이스 설명 생성 중 오류가 발생했습니다: {str(e)}"

# 데이터베이스 정보 가져오기 함수
def get_database_info(db_connection, db_path: str) -> dict:
    """데이터베이스의 테이블, 컬럼, row 수 정보를 가져옵니다."""
    try:
        info = {
            "tables": [],
            "total_rows": 0,
            "table_details": []
        }
        
        # 테이블 목록 가져오기
        tables = db_connection.get_usable_table_names()
        info["tables"] = tables
        
        # SQLite 직접 연결하여 상세 정보 가져오기
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 각 테이블의 정보 가져오기
        table_details = []
        for table in tables:
            try:
                # 컬럼명 추출
                cursor.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()
                columns = [col[1] for col in columns_info]  # col[1]은 컬럼명
                
                # Row 수 가져오기
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                info["total_rows"] += row_count
                
                table_details.append({
                    "name": table,
                    "columns": columns,
                    "row_count": row_count
                })
            except Exception as e:
                logger.warning(f"테이블 {table} 정보 가져오기 실패: {e}")
                continue
        
        conn.close()
        info["table_details"] = table_details
        return info
        
    except Exception as e:
        logger.error(f"데이터베이스 정보 가져오기 오류: {e}")
        return {"tables": [], "total_rows": 0, "table_details": []}

# 데이터베이스 정보 표시
if st.session_state.db_initialized and st.session_state.db_name:
    st.markdown(f"**{st.session_state.db_name}** 데이터베이스에 대해 자연어로 질문해보세요!")
    
    # 데이터베이스 설명 표시
    if st.session_state.db_description:
        st.markdown("---")
        st.markdown('<h3 style="color: #1f77b4;">📝 데이터베이스 설명</h3>', unsafe_allow_html=True)
        st.markdown(st.session_state.db_description)
    
    # 데이터베이스 정보 표시
    if st.session_state.db_info is not None:
        db_info = st.session_state.db_info
        table_details = db_info.get("table_details", [])
        total_rows = db_info.get("total_rows", 0)
        
        # db_info가 있지만 table_details가 비어있으면 다시 가져오기 시도
        if not table_details and st.session_state.db_connection and st.session_state.db_path:
            logger.info("데이터베이스 정보를 다시 가져오는 중...")
            db_info = get_database_info(st.session_state.db_connection, st.session_state.db_path)
            st.session_state.db_info = db_info
            table_details = db_info.get("table_details", [])
            total_rows = db_info.get("total_rows", 0)
            
            # 설명도 다시 생성
            if not st.session_state.db_description and table_details:
                db_description = generate_database_description(db_info, st.session_state.db_name)
                st.session_state.db_description = db_description
        
        if table_details:
            st.markdown("---")
            st.markdown('<h3 style="color: #1f77b4;">📊 데이터베이스 정보</h3>', unsafe_allow_html=True)
            
            # 전체 요약 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("테이블 수", len(table_details))
            with col2:
                st.metric("전체 행 수", f"{total_rows:,}")
            with col3:
                total_columns = sum(len(t["columns"]) for t in table_details)
                st.metric("전체 컬럼 수", total_columns)
            
            # 각 테이블 상세 정보
            st.markdown('<h4 style="color: #ffd700;">테이블 상세 정보</h4>', unsafe_allow_html=True)
            
            for table_detail in table_details:
                with st.expander(f"📋 {table_detail['name']} (행 수: {table_detail['row_count']:,})", expanded=False):
                    st.markdown(f"**컬럼 수:** {len(table_detail['columns'])}")
                    st.markdown("**컬럼 목록:**")
                    # 컬럼을 그리드로 표시
                    cols_per_row = 4
                    columns = table_detail['columns']
                    for i in range(0, len(columns), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(columns):
                                col.code(columns[i + j])
        else:
            st.warning("데이터베이스 정보를 가져올 수 없습니다. 데이터베이스를 다시 생성해주세요.")
    else:
        # db_info가 None이면 다시 가져오기 시도
        if st.session_state.db_connection and st.session_state.db_path:
            logger.info("데이터베이스 정보를 처음 가져오는 중...")
            db_info = get_database_info(st.session_state.db_connection, st.session_state.db_path)
            st.session_state.db_info = db_info
            
            # 설명도 생성
            if db_info.get("table_details"):
                db_description = generate_database_description(db_info, st.session_state.db_name)
                st.session_state.db_description = db_description
            
            st.rerun()
else:
    st.markdown("CSV 파일을 선택하고 데이터베이스에 대해 자연어로 질문해보세요!")

# 데이터베이스 초기화 함수
def initialize_database(db_path: str) -> bool:
    """SQLite 데이터베이스를 초기화하고 SQL Agent를 생성합니다."""
    try:
        # 데이터베이스 이름 추출 (파일명에서 확장자 제거)
        db_name = os.path.splitext(os.path.basename(db_path))[0]
        st.session_state.db_name = db_name
        
        # 데이터베이스 연결
        db_uri = f"sqlite:///{db_path}"
        db_connection = SQLDatabase.from_uri(db_uri)
        st.session_state.db_connection = db_connection
        
        # 데이터베이스 정보 가져오기
        db_info = get_database_info(db_connection, db_path)
        st.session_state.db_info = db_info
        
        # 데이터베이스 설명 생성
        db_description = generate_database_description(db_info, db_name)
        st.session_state.db_description = db_description
        
        # LLM 초기화 (현재 선택된 모델 사용)
        llm = get_llm(st.session_state.llm_model, temperature=0.7)
        
        # SQL Agent 생성 (개선된 프롬프트 포함)
        # SQL Agent용 커스텀 프롬프트
        sql_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query assistant. Your task is to answer questions about the database by writing and executing SQL queries.

IMPORTANT RULES:
1. ALWAYS examine the database schema first using the available tools before answering
2. NEVER say "I don't know" or "I cannot answer" - always try to query the database
3. If you're unsure about the schema, use the tools to explore the database structure
4. Write accurate SQL queries based on the actual table and column names in the database
5. Execute the queries and provide clear, helpful answers in Korean
6. If a query returns no results, explain what you found (or didn't find) rather than saying you don't know
7. Always provide the SQL query you used and explain the results

When answering:
- First, check the database schema to understand available tables and columns
- Write appropriate SQL queries to answer the question
- Execute the queries and interpret the results
- Provide a clear, informative answer in Korean based on the query results
- If the data doesn't exist or query returns empty, explain that clearly instead of saying "I don't know"

Remember: The database schema is available through the tools. Always use them to understand the database structure before answering."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # SQL Agent 생성 (prompt 파라미터 지원 여부에 따라 다르게 처리)
        try:
            sql_agent = create_sql_agent(
                llm,
                db=db_connection,
                agent_type="openai-tools",
                verbose=False,
                prompt=sql_agent_prompt
            )
        except TypeError:
            # prompt 파라미터를 지원하지 않는 경우 기본 방식 사용
            logger.warning("create_sql_agent가 prompt 파라미터를 지원하지 않습니다. 기본 프롬프트를 사용합니다.")
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
        if st.session_state.db_initialized:
            st.session_state.db_initialized = False
            st.session_state.sql_agent = None
            # DB 재초기화
            if st.session_state.db_path:
                initialize_database(st.session_state.db_path)
    
    # 2. CSV 파일 업로드
    st.markdown('<h2 style="color: #ffd700;">2. CSV 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type="csv")
    
    if uploaded_file:
        process_button = st.button("데이터베이스 생성", type="primary", use_container_width=True)
        
        if process_button:
            with st.spinner("CSV 파일을 데이터베이스로 변환 중입니다..."):
                try:
                    # 기존 데이터베이스 연결 닫기
                    close_all_databases()
                    
                    # 기존 데이터베이스 상태 초기화
                    st.session_state.db_initialized = False
                    st.session_state.db_connection = None
                    st.session_state.sql_agent = None
                    st.session_state.db_name = None
                    st.session_state.db_path = None
                    st.session_state.db_info = None
                    if "db_description" in st.session_state:
                        st.session_state.db_description = None
                    
                    # 프로젝트 디렉토리에 DB 파일 저장
                    current_dir = os.getcwd()
                    db_dir = os.path.join(current_dir, "dbs")
                    if not os.path.exists(db_dir):
                        os.makedirs(db_dir)
                    
                    # DB 파일명: CSV 파일명에서 확장자 제거
                    csv_filename = uploaded_file.name
                    db_filename = os.path.splitext(csv_filename)[0] + ".db"
                    db_path = os.path.join(db_dir, db_filename)
                    
                    # CSV를 SQLite로 변환
                    if csv_to_sqlite(uploaded_file, db_path):
                        st.session_state.db_path = db_path
                        
                        # 데이터베이스 초기화
                        if initialize_database(db_path):
                            st.success(f"데이터베이스가 성공적으로 생성되었습니다!")
                            st.rerun()
                        else:
                            st.error("데이터베이스 초기화에 실패했습니다.")
                    else:
                        st.error("CSV 파일 변환에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    logger.error(f"CSV 파일 처리 오류: {e}")
    
    # 데이터베이스 연결 상태 표시
    if st.session_state.db_initialized and st.session_state.db_name:
        st.markdown('<h2 style="color: #ff69b4;">3. 데이터베이스 상태</h2>', unsafe_allow_html=True)
        st.success(f"데이터베이스 연결됨: **{st.session_state.db_name}**")
        if st.button("재연결", use_container_width=True):
            # 기존 연결 닫기
            close_all_databases()
            
            st.session_state.db_initialized = False
            st.session_state.sql_agent = None
            st.session_state.db_name = None
            st.session_state.db_path = None
            st.session_state.db_info = None
            st.session_state.db_description = None
            st.rerun()
    
    # 대화 초기화 버튼
    st.markdown('<h2 style="color: #ff69b4;">4. 대화 관리</h2>', unsafe_allow_html=True)
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
            st.write("먼저 CSV 파일을 업로드하고 데이터베이스를 생성해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 CSV 파일을 업로드하고 데이터베이스를 생성해주세요."})
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
                    # 데이터베이스 구조를 더 상세하게 가져오기
                    table_names = st.session_state.db_connection.get_usable_table_names()
                    schema_details = ""
                    for table in table_names:
                        try:
                            table_info = st.session_state.db_connection.get_table_info_no_throw([table])
                            schema_details += f"\n{table_info}\n"
                        except:
                            pass
                    
                    next_questions_prompt = f"""당신은 데이터베이스 질문 추천 전문가입니다. 다음 정보를 바탕으로 실제로 답변 가능한 질문 3개를 생성해주세요.

사용자의 질문: {prompt}

생성된 답변:
{response_text}

데이터베이스 구조 정보:
{schema_details}

사용 가능한 테이블 목록: {', '.join(table_names)}

중요한 요구사항:
1. 반드시 위에 제공된 데이터베이스 구조 정보를 정확히 확인하세요
2. 존재하는 테이블과 컬럼만 사용하여 SQL로 답변 가능한 질문만 생성하세요
3. 각 질문은 데이터베이스의 실제 데이터를 조회할 수 있어야 합니다
4. 답변 내용을 더 깊이 이해하기 위한 후속 질문을 우선적으로 생성하세요
5. 답변에서 언급된 내용을 구체화하거나 확장하는 질문을 포함하세요
6. 데이터베이스 구조를 활용하여 실제로 SQL 쿼리로 답변할 수 있는 질문만 생성하세요
7. 각 질문은 완전한 한국어 문장으로 작성하되, 간결하고 명확하게 작성하세요
8. 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성하세요
9. 데이터베이스에 존재하지 않는 테이블이나 컬럼을 언급하는 질문은 절대 생성하지 마세요
10. 생성된 질문은 반드시 SQL 쿼리로 실행 가능해야 합니다

생성 형식 (번호 없이):
질문1
질문2
질문3

중요: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요. 반드시 데이터베이스 구조를 확인하여 실제로 답변 가능한 질문만 생성하세요."""
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
                    progress_bar.progress(100)
                    status_text.text("완료!")
                
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
    <p>SQL Agent 2 - CSV 파일을 데이터베이스로 변환하고 자연어로 질문하고 답변을 받아보세요</p>
</div>
""", unsafe_allow_html=True)

