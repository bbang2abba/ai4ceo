import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from typing import Any
from langchain_perplexity import ChatPerplexity
from datetime import datetime
import logging
import re

# 환경 변수 로드
load_dotenv()

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

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)
    elif model_name == "claude-sonnet-4-5":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    elif model_name == "gemini-3-pro-preview":
        from langchain_google_genai import ChatGoogleGenerativeAI
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
    page_title="기재부 잿봇",
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
    st.session_state.use_rag = True

if "search_model" not in st.session_state:
    st.session_state.search_model = "사용 안 함"

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

# 로고 및 제목 영역 (상단에 배치)
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_logo, col_title, col_empty = st.columns([1, 4, 1])

with col_logo:
    # 기획재정부 로고 이미지 파일 경로 (우선 사용)
    logo_file = "기획재정부.png"
    
    # 대체 로고 파일 경로들
    logo_paths = [
        logo_file,  # 최우선: 기획재정부.png
        "logo.png",
        "logo.jpg",
        "assets/logo.png",
        "images/logo.png",
        "static/logo.png",
        "기재부로고.png",
        "기재부_로고.png",
        "기획재정부로고.png",
        "기획재정부_로고.png"
    ]
    
    logo_found = False
    # 로컬 파일 확인 (기획재정부.png 우선)
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            try:
                st.image(logo_path, width=120)
                logo_found = True
                break
            except Exception as e:
                logger.warning(f"로고 파일 로드 실패 ({logo_path}): {e}")
                continue
    
    # 로컬 파일이 없으면 기본 아이콘 표시
    if not logo_found:
        st.markdown("""
        <div style="margin-top: 0.5rem;">
            <div style="width: 120px; height: 120px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1f77b4 0%, #ffd700 100%); border-radius: 50%; margin: 0 auto;">
                <span style="color: white; font-size: 1.5rem; font-weight: bold;">기재부</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        logger.warning(f"로고 파일을 찾을 수 없습니다: {logo_file}")

with col_title:
    # 제목 (더 크게)
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">기획재정부</span> 
            <span style="color: #ffd700;">챗봇</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    # 오른쪽 여백
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("모델을 선택하고 인터넷검색과 RAG 선택해주세요.")

# Perplexity 검색 함수
def search_with_perplexity_chat(prompt: str, history: list, temperature: float = 0.7) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "PERPLEXITY_API_KEY가 환경변수에 설정되어 있지 않습니다."
    
    messages = []
    system_content = """당신은 전문적인 AI 어시스턴트입니다. 답변을 전문적으로 해줘.

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
    messages.append({"role": "system", "content": system_content})

    filtered_history = []
    last_role = "system"
    for msg in history:
        if msg["role"] not in ["user", "assistant"]:
            continue
        if last_role == "system" and msg["role"] != "user":
            continue
        if last_role == msg["role"]:
            continue
        filtered_history.append(msg)
        last_role = msg["role"]

    messages.extend(filtered_history)

    if not filtered_history or filtered_history[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": prompt})

    llm = ChatPerplexity(model="sonar-pro", temperature=temperature, api_key=api_key)
    try:
        response = llm.invoke(messages)
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Perplexity 검색 중 오류: {e}"

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
    st.session_state.llm_model = selected_model

    # 2. 인터넷 검색 선택
    st.markdown('<h2 style="color: #ffd700;">2. 인터넷 검색</h2>', unsafe_allow_html=True)
    search_model = st.radio(
        "인터넷 검색을 사용하시겠습니까?",
        [
            "사용 안 함",
            "Perplexity 사용"
        ],
        index=0 if st.session_state.search_model == "사용 안 함" else 1
    )
    st.session_state.search_model = search_model

    # 3. RAG 선택
    st.markdown('<h2 style="color: #ff69b4;">3. RAG (PDF 검색)</h2>', unsafe_allow_html=True)
    use_rag = st.radio(
        "RAG를 사용하시겠습니까?",
        [
            "사용 안 함",
            "RAG 사용"
        ],
        index=0 if not st.session_state.use_rag else 1
    )
    st.session_state.use_rag = (use_rag == "RAG 사용")

    # 4. PDF 파일 업로드
    st.markdown('<h2 style="color: #d62728;">4. PDF 파일 업로드</h2>', unsafe_allow_html=True)
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
                        
                        # 임베딩 및 벡터 스토어 생성
                        embeddings = OpenAIEmbeddings()
                        
                        if st.session_state.vectorstore is None:
                            # 새 벡터 스토어 생성
                            batch_size = 30
                            vectorstore = None
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    if vectorstore is None:
                                        vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                                    else:
                                        vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                            
                            st.session_state.vectorstore = vectorstore
                        else:
                            # 기존 벡터 스토어에 추가
                            batch_size = 30
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    st.session_state.vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                        
                        # 검색기 생성 (더 많은 결과와 정확한 검색)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10}  # 검색 결과 수 증가
                        )
                        
                        # 처리된 파일 목록 업데이트
                        st.session_state.processed_files.extend(new_files)
                        
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
    st.text(f"모델: {st.session_state.llm_model}")
    st.text(f"인터넷 검색: {st.session_state.search_model}")
    st.text(f"RAG: {'사용' if st.session_state.use_rag else '사용 안 함'}")
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
    if st.session_state.search_model == "Perplexity 사용":
        with st.spinner("Perplexity 검색 중..."):
            chat_history_for_perplexity = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] in ["user", "assistant"]:
                    chat_history_for_perplexity.append({"role": msg["role"], "content": msg["content"]})

            response_text = ""
            try:
                response_text = search_with_perplexity_chat(prompt, chat_history_for_perplexity)
                if not response_text or not isinstance(response_text, str):
                    response_text = "Perplexity 검색 결과가 없습니다."
                
                # 답변에서 구분선 제거
                response_text = remove_separators(response_text)
                
                # 다음 질문 3개 생성
                try:
                    llm = get_llm(st.session_state.llm_model, temperature=1)
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
                    next_questions_response = llm.invoke(next_questions_prompt).content
                    next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                    next_questions = next_questions[:3]
                    
                    if next_questions:
                        response_text += "\n\n"
                        response_text += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                        for i, question in enumerate(next_questions, 1):
                            response_text += f"{i}. {question}\n\n"
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"Perplexity 검색 중 오류: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                logger.error(f"Perplexity 검색 오류: {e}")

    # --- 2순위: RAG 또는 직접 LLM 답변 (인터넷 검색 '사용 안 함'일 때만 실행) ---
    elif st.session_state.search_model == "사용 안 함":
        # --- 2-1: RAG 사용이 선택되었고 PDF 파일이 있는 경우 ---
        if st.session_state.use_rag and st.session_state.retriever is not None:
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
                        - 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
                        - 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
                        - 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
                        - 답변은 서술형으로 작성하되 존대말을 사용하세요
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
                        llm = get_llm(st.session_state.llm_model, temperature=1)
                        
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

        # --- 2-2: RAG 사용이 선택되지 않았거나 PDF 파일이 없는 경우 (직접 LLM 사용) ---
        else:
            if st.session_state.use_rag and st.session_state.retriever is None:
                with st.chat_message("assistant"):
                    st.warning("RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요.")
                st.session_state.chat_history.append({"role": "assistant", "content": "RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요."})
                logger.warning("RAG 선택되었으나 PDF 파일이 없음")
            else:
                try:
                    llm = get_llm(st.session_state.llm_model, temperature=1)
                    direct_prompt = f"""당신은 유능한 AI 어시스턴트입니다. 반드시 한국어로 답변해주세요.

질문: {prompt}

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
                    
                    response = ""
                    with st.chat_message("assistant"):
                        stream_placeholder = st.empty()
                        # 스트리밍으로 답변 생성
                        for chunk in llm.stream(direct_prompt):
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
                    try:
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
                        next_questions_response = llm.invoke(next_questions_prompt).content
                        next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                        next_questions = next_questions[:3]
                        
                        if next_questions:
                            response += "\n\n"
                            response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                            for i, question in enumerate(next_questions, 1):
                                response += f"{i}. {question}\n\n"
                            # 다음 질문 추가 후 다시 표시
                            with st.chat_message("assistant"):
                                st.markdown(response)
                    except Exception as e:
                        logger.warning(f"다음 질문 생성 실패: {e}")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"LLM 생성 중 오류 발생: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    logger.error(f"LLM 답변 생성 오류: {e}")