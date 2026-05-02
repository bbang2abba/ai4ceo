import os
import streamlit as st
import tempfile
import ssl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from typing import Any
from datetime import datetime
import logging
import re

# SSL 인증서 검증 문제 해결을 위한 전역 설정
# (api.openai.com은 뚫려 있으나, SDK 초기화 시 발생할 수 있는 부가 통신 차단 방지)
os.environ["PYTHONHTTPSVERIFY"] = "0"
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 환경 변수 로드 (.env 파일을 프로젝트 루트에서 찾기)
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)  # ref.py가 프로젝트 루트에 있다고 가정
env_path = os.path.join(project_root, ".env")

# 프로젝트 루트의 .env 파일 로드
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # 프로젝트 루트에 없으면 현재 작업 디렉토리에서 로드
    load_dotenv()

# OpenAI API 키 로드 함수
def get_openai_api_key() -> str:
    """OpenAI API 키를 .env 파일에서 명시적으로 로드합니다."""
    # 프로젝트 루트 디렉토리 찾기
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file)
    env_path = os.path.join(project_root, ".env")
    
    # .env 파일이 없으면 현재 작업 디렉토리에서 찾기
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
    
    if os.path.exists(env_path):
        from dotenv import dotenv_values
        env_vars = dotenv_values(env_path)
        api_key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY를 찾을 수 없음")
    
    return api_key


# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"chatbot_{datetime.now().strftime('%Y%m%d')}.log")
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

# LLM 모델 함수 (gpt-5.2 고정)
def get_llm(temperature: float = 0.7, api_key: str = None) -> Any:
    """gpt-5.2 모델을 반환합니다."""
    # API 키가 제공되지 않으면 .env 파일에서 로드
    if api_key is None:
        api_key = get_openai_api_key()
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    
    return ChatOpenAI(model="gpt-5.2", temperature=temperature, api_key=api_key)

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "search_model" not in st.session_state:
    st.session_state.search_model = "gpt-5.2 web_search 사용"  # 기본값: 인터넷 검색 모드

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

# 제목 영역 (상단에 배치)
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_title, col_empty = st.columns([4, 1])

with col_title:
    # 제목 (더 크게)
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">RAG</span> 
            <span style="color: #ffd700;">챗봇</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    # 오른쪽 여백
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.search_model == "gpt-5.2 web_search 사용":
    st.markdown("🌐 **인터넷 검색 모드**: gpt-5.2와 web_search를 사용하여 최신 정보를 검색합니다.")
else:
    st.markdown("📚 **RAG 모드**: PDF 파일을 업로드하여 문서 기반 답변을 받을 수 있습니다.")

# gpt-5.2 web_search 검색 함수 (OpenAI Responses API 사용)
def search_with_web_search(prompt: str, history: list, temperature: float = 0.7) -> str:
    """OpenAI Responses API를 사용하여 web_search를 수행합니다."""
    api_key = get_openai_api_key()
    if not api_key:
        return "OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다. 프로젝트 루트 디렉토리의 .env 파일에 OPENAI_API_KEY를 추가해주세요."
    
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=api_key)
        
        # 대화 기록을 고려한 입력 구성
        # 최근 대화 맥락을 포함하여 더 정확한 검색 수행
        context_prompt = prompt
        if history:
            # 최근 3개 대화만 포함 (너무 길어지지 않도록)
            recent_history = history[-3:] if len(history) > 3 else history
            context = "\n\n이전 대화 맥락:\n"
            for msg in recent_history:
                if msg["role"] == "user":
                    context += f"사용자: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    context += f"어시스턴트: {msg['content']}\n"
            context_prompt = context + f"\n현재 질문: {prompt}"
        
        # 시스템 프롬프트 추가
        system_instruction = """당신은 전문적인 AI 어시스턴트입니다. 답변을 전문적으로 해줘.

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
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
        
        full_prompt = f"{system_instruction}\n\n{context_prompt}"
        
        # Responses API (2026년 표준) 호출
        # Built-in web_search 도구를 사용하여 OpenAI 서버에서 검색을 대행함
        response = client.responses.create(
            model="gpt-5.2",
            input=full_prompt,
            tools=[
                {
                    "type": "web_search",  # OpenAI 내장 검색 도구
                    "search_context_size": "high"  # 상세한 검색 결과 참조
                }
            ]
        )
        
        # 결과 및 인용(Citations) 처리
        # GPT-5.2는 답변 내에 검색 출처를 구조화하여 포함합니다.
        answer = response.output_text
        
        return answer
        
    except Exception as e:
        logger.error(f"web_search 검색 중 오류: {e}")
        return f"web_search 검색 중 오류: {e}"

# 사이드바 설정
with st.sidebar:
    # 모드 선택 (인터넷 검색 또는 RAG)
    st.markdown('<h2 style="color: #ffd700;">모드 선택</h2>', unsafe_allow_html=True)
    mode = st.radio(
        "사용할 모드를 선택하세요:",
        [
            "🌐 인터넷 검색 (gpt-5.2 web_search)",
            "📚 RAG (PDF 검색)"
        ],
        index=0 if st.session_state.search_model == "gpt-5.2 web_search 사용" else 1,
        key="mode_selection"
    )
    
    # 선택에 따라 상태 업데이트
    if mode == "🌐 인터넷 검색 (gpt-5.2 web_search)":
        st.session_state.search_model = "gpt-5.2 web_search 사용"
        st.session_state.use_rag = False
        st.info("💡 gpt-5.2와 web_search를 사용하여 인터넷에서 최신 정보를 검색합니다.")
    else:
        st.session_state.search_model = "사용 안 함"
        st.session_state.use_rag = True
        st.info("💡 PDF 파일을 업로드하여 문서 기반 답변을 받을 수 있습니다.")

    # RAG 모드일 때만 PDF 파일 업로드 표시
    if st.session_state.use_rag:
        st.markdown("---")
        st.markdown('<h2 style="color: #ff69b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # 임시 파일 생성 및 처리
                    temp_dir = tempfile.TemporaryDirectory()
                    
                    all_docs = []
                    new_files = []
                    
                    # 각 파일 처리
                    for uploaded_file in uploaded_files:
                        # 이미 처리된 파일 스킵
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                            
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        
                        # 업로드된 파일을 임시 파일로 저장
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # PDF 로더 생성 및 문서 로드
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        # 메타데이터에 파일 이름 추가
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # 모든 청크를 벡터 데이터베이스에 저장
                        total_chunks = len(chunks)
                        
                        # API 키 확인 (.env 파일에서 명시적으로 로드)
                        openai_api_key = get_openai_api_key()
                        
                        if not openai_api_key:
                            st.error("**OpenAI API 키가 설정되지 않았습니다.**\n\n"
                                   "`.env` 파일에 `OPENAI_API_KEY=your_api_key_here` 형식으로 API 키를 추가해주세요.")
                            logger.error("OPENAI_API_KEY가 환경 변수에 설정되지 않음")
                        else:
                            # 임베딩 및 벡터 스토어 생성 (명시적으로 API 키 전달)
                            try:
                                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                            except Exception as e:
                                st.error("**OpenAI API 키 초기화 중 오류가 발생했습니다.**\n\n"
                                       "API 키를 확인해주세요.")
                                logger.error(f"OpenAIEmbeddings 초기화 오류: {e}")
                                embeddings = None
                            
                            if embeddings is not None:
                                if st.session_state.vectorstore is None:
                                    # 새 벡터 스토어 생성
                                    batch_size = 30
                                    vectorstore = None
                                    api_key_error = False
                                    first_error = None
                                    
                                    for i in range(0, len(chunks), batch_size):
                                        batch_chunks = chunks[i:i + batch_size]
                                        
                                        try:
                                            if vectorstore is None:
                                                vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                                            else:
                                                vectorstore.add_documents(batch_chunks)
                                        except Exception as e:
                                            error_str = str(e)
                                            logger.warning(f"벡터 스토어 생성 중 오류 (배치 {i//batch_size + 1}): {e}")
                                            
                                            # 첫 번째 오류 저장
                                            if first_error is None:
                                                first_error = error_str
                                            
                                            # API 키 오류인지 확인
                                            if "401" in error_str or "invalid_api_key" in error_str.lower() or "incorrect api key" in error_str.lower():
                                                api_key_error = True
                                                logger.error("OpenAI API 키 오류 감지 - 처리 중단")
                                                break  # API 키 오류면 더 이상 시도하지 않음
                                            continue
                                    
                                    # 벡터 스토어가 성공적으로 생성되었는지 확인
                                    if vectorstore is None:
                                        if api_key_error:
                                            # API 키 오류 메시지 간소화
                                            with st.container():
                                                st.error("**🔑 OpenAI API 키 오류**")
                                                
                                                if first_error and "incorrect api key" in first_error.lower():
                                                    st.warning("❌ 제공된 API 키가 올바르지 않습니다.")
                                                else:
                                                    st.warning("❌ API 키 인증에 실패했습니다.")
                                                
                                                with st.expander("📋 해결 방법 보기", expanded=True):
                                                    st.markdown("""
                                                    **1. `.env` 파일 확인**
                                                    - 프로젝트 루트 디렉토리에 `.env` 파일이 있는지 확인
                                                    - 파일 내용: `OPENAI_API_KEY=sk-...` 형식으로 설정
                                                    
                                                    **2. API 키 유효성 확인**
                                                    - [OpenAI API Keys](https://platform.openai.com/account/api-keys)에서 키가 활성화되어 있는지 확인
                                                    - 키가 만료되었거나 삭제된 경우 새로 생성
                                                    
                                                    **3. 크레딧 확인**
                                                    - [Usage](https://platform.openai.com/account/usage)에서 충분한 크레딧이 있는지 확인
                                                    
                                                    **4. 앱 재시작**
                                                    - `.env` 파일 수정 후 Streamlit 앱을 재시작하세요
                                                    """)
                                                
                                                # 오류 상세 정보 (접을 수 있게)
                                                if first_error:
                                                    with st.expander("🔍 오류 상세 정보"):
                                                        st.code(first_error[:500], language=None)
                                            
                                            logger.error("벡터 스토어 생성 실패: OpenAI API 키 오류")
                                        else:
                                            st.error("**벡터 스토어 생성 실패**\n\nPDF 파일을 확인해주세요.")
                                            if first_error:
                                                logger.error(f"벡터 스토어 생성 실패: {first_error}")
                                            else:
                                                logger.error("벡터 스토어 생성 실패: 모든 배치에서 오류 발생")
                                    else:
                                        st.session_state.vectorstore = vectorstore
                                else:
                                    # 기존 벡터 스토어에 추가
                                    batch_size = 30
                                    
                                    for i in range(0, len(chunks), batch_size):
                                        batch_chunks = chunks[i:i + batch_size]
                                        
                                        try:
                                            st.session_state.vectorstore.add_documents(batch_chunks)
                                        except Exception as e:
                                            logger.warning(f"벡터 스토어에 문서 추가 중 오류 (배치 {i//batch_size + 1}): {e}")
                                            continue
                        
                        # 검색기 생성 (벡터 스토어가 존재할 때만)
                        if st.session_state.vectorstore is not None:
                            st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 10}  # 검색 결과 수 증가
                            )
                            
                            # 처리된 파일 목록 업데이트
                            st.session_state.processed_files.extend(new_files)
                            st.success(f"✅ {len(new_files)}개 파일 처리 완료!")
                        else:
                            st.error("벡터 스토어가 생성되지 않아 파일을 처리할 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
                    logger.error(f"PDF 파일 처리 오류: {e}")

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
    
    # 현재 설정 표시
    st.markdown('<h3 style="color: #1f77b4;">현재 설정</h3>', unsafe_allow_html=True)
    if st.session_state.search_model == "gpt-5.2 web_search 사용":
        st.text("모드: 인터넷 검색 (gpt-5.2 web_search)")
        st.text("모델: OpenAI gpt-5.2")
    else:
        st.text("모드: RAG (PDF 검색)")
        st.text("모델: OpenAI gpt-5.2")
    if st.session_state.processed_files:
        st.text(f"처리된 파일: {len(st.session_state.processed_files)}개")
        st.text(f"대화 기록: {len(st.session_state.chat_history)}개")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # --- 1순위: 인터넷 검색 모델 처리 ---
    if st.session_state.search_model == "gpt-5.2 web_search 사용":
        with st.spinner("gpt-5.2 web_search 검색 중..."):
            chat_history_for_search = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] in ["user", "assistant"]:
                    chat_history_for_search.append({"role": msg["role"], "content": msg["content"]})

            response_text = ""
            try:
                response_text = search_with_web_search(prompt, chat_history_for_search)
                if not response_text or not isinstance(response_text, str):
                    response_text = "web_search 검색 결과가 없습니다."
                
                # 답변에서 구분선 제거
                response_text = remove_separators(response_text)
                
                # 다음 질문 3개 생성 (gpt-5.2 사용)
                try:
                    openai_api_key = get_openai_api_key()
                    if openai_api_key:
                        llm = get_llm(temperature=1, api_key=openai_api_key)
                        next_questions_prompt = f"""
                        질문자가 한 질문: {prompt}
                        
                        생성된 답변:
                        {response_text}
                        
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
                        next_questions_response = llm.invoke([{"role": "user", "content": next_questions_prompt}])
                        if hasattr(next_questions_response, 'content'):
                            next_questions_text = next_questions_response.content
                        else:
                            next_questions_text = str(next_questions_response)
                        next_questions = [q.strip() for q in next_questions_text.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                        next_questions = next_questions[:3]
                        
                        if next_questions:
                            response_text += "\n\n"
                            response_text += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                            for i, question in enumerate(next_questions, 1):
                                response_text += f"{i}. {question}\n\n"
                    else:
                        logger.warning("OPENAI_API_KEY가 없어 다음 질문 생성을 건너뜁니다.")
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"web_search 검색 중 오류: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                logger.error(f"web_search 검색 오류: {e}")

    # --- RAG 모드: OpenAI 모델 사용 ---
    elif st.session_state.use_rag:
        # RAG 사용: PDF 파일이 있는 경우
        if st.session_state.retriever is not None:
            with st.spinner("PDF 기반 RAG 답변을 생성 중입니다..."):
                try:
                    # RAG 검색 (상위 3개 문서만 사용)
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    
                    if not retrieved_docs:
                        response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                    else:
                        # 상위 3개 문서만 사용
                        top_docs = retrieved_docs[:3]
                        
                        # 컨텍스트 구성
                        context_text = ""
                        max_context_length = 8000
                        current_length = 0
                        
                        for i, doc in enumerate(top_docs):
                            doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                            if current_length + len(doc_text) > max_context_length:
                                st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                                break
                            context_text += doc_text
                            current_length += len(doc_text)
                        
                        # 과거 대화 맥락 구성
                        conversation_context = ""
                        if st.session_state.conversation_memory:
                            conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                            # 최근 50개 대화 사용
                            recent_conversations = st.session_state.conversation_memory[-50:]
                            for conv in recent_conversations:
                                conversation_context += f"{conv}\n"
                            conversation_context += "=== 대화 맥락 끝 ===\n"
                        
                        # 시스템 프롬프트 구성
                        system_prompt = f"""
                        질문: {prompt}
                        
                        관련 문서:
                        {context_text}{conversation_context}
                        
                        위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
                        이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.
                        
                        답변 형식:
                        - 답변은 반드시 제목과 본문으로 구분하여 작성하세요
                        - 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
                        - 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
                        - 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
                        - 본문은 서술형으로 작성하되 존대말을 사용하세요
                        - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                        
                        주의사항:
                        - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                        - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                        - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                        - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                        - 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
                        - 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
                        - 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
                        - 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요
                        """
                        
                        # LLM으로 답변 생성 (스트리밍 모드)
                        llm = get_llm(temperature=1)
                        
                        response = ""
                        with st.chat_message("assistant"):
                            stream_placeholder = st.empty()
                            # 스트리밍으로 답변 생성
                            for chunk in llm.stream(system_prompt):
                                if hasattr(chunk, 'content'):
                                    chunk_text = chunk.content
                                else:
                                    chunk_text = str(chunk)
                                response += chunk_text
                                # 실시간으로 표시 (구분선 제거 포함)
                                cleaned_response = remove_separators(response)
                                stream_placeholder.markdown(cleaned_response)
                        
                        # 답변에서 구분선 제거
                        response = remove_separators(response)
                    
                        # 다음 질문 3개 생성
                        next_questions_prompt = f"""
                        질문자가 한 질문: {prompt}
                        
                        생성된 답변:
                        {response}
                        
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
                            next_questions_response = llm.invoke(next_questions_prompt).content
                            # 질문들을 리스트로 파싱
                            next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                            # 최대 3개만 선택
                            next_questions = next_questions[:3]
                            
                            # 답변 끝에 다음 질문 추가
                            if next_questions:
                                response += "\n\n"
                                response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                                for i, question in enumerate(next_questions, 1):
                                    response += f"{i}. {question}\n\n"
                                # 다음 질문 추가 후 다시 표시
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                        except Exception as e:
                            # 다음 질문 생성 실패 시 무시하고 원래 답변만 표시
                            logger.warning(f"다음 질문 생성 실패: {e}")
                        
                        # 대화 기록에 추가
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # 대화 맥락 메모리에 추가 (최근 50개 대화 유지)
                        st.session_state.conversation_memory.append(f"사용자: {prompt}")
                        st.session_state.conversation_memory.append(f"AI: {response}")
                        if len(st.session_state.conversation_memory) > 100:  # 50개 대화 = 100개 메시지
                            st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                    
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.write(f"오류가 발생했습니다: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
                    logger.error(f"RAG 답변 생성 오류: {e}")

        # RAG 모드: PDF 파일이 없는 경우
        else:
            with st.chat_message("assistant"):
                st.warning("📚 RAG 모드를 사용하려면 먼저 PDF 파일을 업로드하고 '파일 처리하기' 버튼을 클릭해주세요.")
            st.session_state.chat_history.append({"role": "assistant", "content": "RAG 모드를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요."})
            logger.warning("RAG 모드 선택되었으나 PDF 파일이 없음")