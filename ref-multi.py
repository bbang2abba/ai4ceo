import os
import streamlit as st
import tempfile
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from supabase import create_client, Client

# 환경 변수 로드 (기존 값 덮어쓰기)
load_dotenv(override=True)

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 초기화"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        st.warning("⚠️ Supabase 환경 변수가 설정되지 않았습니다. .env 파일에 SUPABASE_URL과 SUPABASE_ANON_KEY를 추가해주세요.")
        return None
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Supabase 연결 오류: {str(e)}")
        return None

supabase: Client = init_supabase()

# OpenAI API 키 검증
def check_openai_api_key():
    """OpenAI API 키가 설정되어 있는지 확인"""
    # 환경 변수 다시 로드 (최신 값 확인)
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다."
    
    # 공백 제거
    api_key = api_key.strip()
    
    # 따옴표 제거 (혹시 따옴표로 감싸져 있을 경우)
    api_key = api_key.strip('"').strip("'")
    
    # API 키 형식 검증 (sk- 또는 sk-proj-로 시작하는지 확인)
    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        return False, f"OPENAI_API_KEY 형식이 올바르지 않습니다. 'sk-' 또는 'sk-proj-'로 시작해야 합니다.\n현재 키: {api_key[:10]}... (처음 10자만 표시)\n전체 길이: {len(api_key)}자"
    
    # 최소 길이 확인 (일반적으로 50자 이상)
    if len(api_key) < 20:
        return False, f"OPENAI_API_KEY가 너무 짧습니다. 올바른 API 키인지 확인해주세요.\n현재 길이: {len(api_key)}자"
    
    return True, None

# 세션 제목 생성 함수
def generate_session_title(chat_history: list) -> str:
    """대화 내용을 기반으로 세션 제목 생성"""
    if not chat_history or len(chat_history) == 0:
        return "새 세션"
    
    # 최근 대화만 사용 (처음 3개 대화)
    recent_chats = chat_history[:6] if len(chat_history) > 6 else chat_history
    
    # 대화 내용 요약
    conversation_text = ""
    for msg in recent_chats:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"사용자: {content[:100]}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content[:200]}\n"
    
    if not conversation_text.strip():
        return "새 세션"
    
    try:
        # API 키 확인
        api_key_valid, _ = check_openai_api_key()
        if not api_key_valid:
            # API 키가 없으면 첫 번째 사용자 메시지의 일부를 제목으로 사용
            first_user_msg = next((msg.get("content", "") for msg in chat_history if msg.get("role") == "user"), "")
            if first_user_msg:
                return first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
            return "새 세션"
        
        # LLM으로 제목 생성
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = f"""다음 대화 내용을 기반으로 간결하고 명확한 세션 제목을 생성해주세요.

대화 내용:
{conversation_text}

요구사항:
- 제목은 20자 이내로 작성
- 대화의 주요 주제를 반영
- 한글로 작성
- 따옴표나 특수문자 없이 작성
- 제목만 출력 (설명 없이)

제목:"""
        
        title = llm.invoke(prompt).content.strip()
        # 따옴표 제거
        title = title.strip('"').strip("'").strip()
        
        # 너무 길면 자르기
        if len(title) > 30:
            title = title[:27] + "..."
        
        return title if title else "새 세션"
    except Exception as e:
        # 오류 발생 시 첫 번째 사용자 메시지 사용
        first_user_msg = next((msg.get("content", "") for msg in chat_history if msg.get("role") == "user"), "")
        if first_user_msg:
            return first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
        return "새 세션"

# 세션 관리 함수
def save_session_to_supabase(session_id: str):
    """현재 세션을 Supabase에 저장"""
    if supabase is None:
        return False
    
    try:
        # 세션 제목 생성 (대화 내용이 있을 때만)
        title = None
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            title = generate_session_title(st.session_state.chat_history)
        
        session_data = {
            "session_id": session_id,
            "chat_history": json.dumps(st.session_state.chat_history, ensure_ascii=False),
            "conversation_memory": json.dumps(st.session_state.conversation_memory, ensure_ascii=False),
            "processed_files": json.dumps(st.session_state.processed_files, ensure_ascii=False),
            "metadata": json.dumps({
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, ensure_ascii=False)
        }
        
        # 제목이 있으면 추가
        if title:
            session_data["title"] = title
        
        # 기존 세션이 있는지 확인
        existing = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
        
        if existing.data:
            # 업데이트
            supabase.table("sessions").update(session_data).eq("session_id", session_id).execute()
        else:
            # 새로 생성
            supabase.table("sessions").insert(session_data).execute()
        
        return True
    except Exception as e:
        st.error(f"세션 저장 오류: {str(e)}")
        return False

def load_session_from_supabase(session_id: str):
    """Supabase에서 세션 로드"""
    if supabase is None:
        return False
    
    try:
        result = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
        
        if result.data:
            session_data = result.data[0]
            
            # 세션 데이터 복원
            if session_data.get("chat_history"):
                st.session_state.chat_history = json.loads(session_data["chat_history"])
            if session_data.get("conversation_memory"):
                st.session_state.conversation_memory = json.loads(session_data["conversation_memory"])
            if session_data.get("processed_files"):
                st.session_state.processed_files = json.loads(session_data["processed_files"])
            
            return True
        return False
    except Exception as e:
        st.error(f"세션 로드 오류: {str(e)}")
        return False

def list_sessions_from_supabase():
    """Supabase에서 모든 세션 목록 가져오기"""
    if supabase is None:
        return []
    
    try:
        result = supabase.table("sessions").select("session_id, title, created_at, updated_at").order("updated_at", desc=True).limit(50).execute()
        sessions = result.data if result.data else []
        
        # 제목이 없는 세션에 대해 제목 생성
        for session in sessions:
            if not session.get("title"):
                # 세션 데이터를 가져와서 제목 생성
                session_data = supabase.table("sessions").select("chat_history").eq("session_id", session["session_id"]).execute()
                if session_data.data and session_data.data[0].get("chat_history"):
                    try:
                        chat_history = json.loads(session_data.data[0]["chat_history"])
                        title = generate_session_title(chat_history)
                        # 제목 업데이트
                        supabase.table("sessions").update({"title": title}).eq("session_id", session["session_id"]).execute()
                        session["title"] = title
                    except:
                        session["title"] = "제목 없음"
                else:
                    session["title"] = "새 세션"
        
        return sessions
    except Exception as e:
        st.error(f"세션 목록 조회 오류: {str(e)}")
        return []

def delete_session_from_supabase(session_id: str):
    """Supabase에서 세션 삭제"""
    if supabase is None:
        return False
    
    try:
        supabase.table("sessions").delete().eq("session_id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"세션 삭제 오류: {str(e)}")
        return False

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 RAG 챗봇",
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

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = False

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

# 제목
st.markdown("""
<div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span> 
        <span style="color: #ffffff; font-size: 0.7em;">기반</span> 
        <span style="color: #ffd700;">RAG</span> 
        <span style="color: #d62728; font-size: 0.7em;">챗봇</span>
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!")

# 사이드바 설정
with st.sidebar:
    # 세션 관리 섹션
    st.markdown('<h2 style="color: #1f77b4;">세션 관리</h2>', unsafe_allow_html=True)
    
    if supabase:
        # 세션 목록 가져오기
        sessions = list_sessions_from_supabase()
        # 세션 옵션 생성 (제목 + 날짜)
        session_options = ["새 세션 생성"]
        for s in sessions:
            title = s.get("title", "제목 없음")
            date = s.get("updated_at", "")[:10] if s.get("updated_at") else ""
            session_options.append(f"{title} ({date})")
        
        selected_session = st.selectbox(
            "세션 선택",
            options=session_options,
            index=0 if not st.session_state.session_loaded else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("세션 로드", use_container_width=True):
                if selected_session != "새 세션 생성":
                    # 선택된 세션 찾기 (제목으로 매칭)
                    selected_index = session_options.index(selected_session) - 1  # "새 세션 생성" 제외
                    if 0 <= selected_index < len(sessions):
                        selected_session_data = sessions[selected_index]
                        full_session_id = selected_session_data['session_id']
                        
                        if load_session_from_supabase(full_session_id):
                            st.session_state.current_session_id = full_session_id
                            st.session_state.session_loaded = True
                            st.success(f"세션 '{selected_session_data.get('title', '제목 없음')}'이 로드되었습니다!")
                            st.rerun()
                        else:
                            st.error("세션 로드에 실패했습니다.")
                    else:
                        st.error("세션을 찾을 수 없습니다.")
                else:
                    st.info("새 세션을 생성합니다.")
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.session_state.session_loaded = False
        
        with col2:
            if st.button("세션 저장", use_container_width=True):
                if save_session_to_supabase(st.session_state.current_session_id):
                    st.success("세션이 저장되었습니다!")
                else:
                    st.error("세션 저장에 실패했습니다.")
        
        # 세션 삭제
        if selected_session != "새 세션 생성" and sessions:
            if st.button("선택한 세션 삭제", use_container_width=True):
                # 선택된 세션 찾기
                selected_index = session_options.index(selected_session) - 1  # "새 세션 생성" 제외
                if 0 <= selected_index < len(sessions):
                    selected_session_data = sessions[selected_index]
                    full_session_id = selected_session_data['session_id']
                    session_title = selected_session_data.get('title', '제목 없음')
                    
                    if delete_session_from_supabase(full_session_id):
                        st.success(f"세션 '{session_title}'이 삭제되었습니다!")
                        st.rerun()
                    else:
                        st.error("세션 삭제에 실패했습니다.")
                else:
                    st.error("세션을 찾을 수 없습니다.")
        
        st.markdown("---")
        # 현재 세션 제목 표시
        current_session_title = "새 세션"
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            # 현재 세션의 제목 생성 또는 가져오기
            try:
                session_data = supabase.table("sessions").select("title").eq("session_id", st.session_state.current_session_id).execute()
                if session_data.data and session_data.data[0].get("title"):
                    current_session_title = session_data.data[0]["title"]
                else:
                    # 제목이 없으면 생성
                    current_session_title = generate_session_title(st.session_state.chat_history)
            except:
                current_session_title = generate_session_title(st.session_state.chat_history)
        
        st.info(f"📌 현재 세션: **{current_session_title}**")
        st.caption(f"세션 ID: `{st.session_state.current_session_id[:8]}...`")
    else:
        st.warning("Supabase가 연결되지 않았습니다. 세션 저장 기능을 사용할 수 없습니다.")
    
    st.markdown("---")
    st.markdown('<h2 style="color: #1f77b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
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
                        st.info(f"총 {total_chunks}개의 청크를 처리합니다.")
                        
                        # OpenAI API 키 확인
                        api_key_valid, api_key_error = check_openai_api_key()
                        if not api_key_valid:
                            api_key_preview = os.getenv("OPENAI_API_KEY", "")[:20] if os.getenv("OPENAI_API_KEY") else "없음"
                            raise Exception(
                                f"OpenAI API 키 오류: {api_key_error}\n\n"
                                f"**현재 설정된 키 (처음 20자):** `{api_key_preview}...`\n\n"
                                "**해결 방법:**\n"
                                "1. .env 파일을 열어주세요\n"
                                "2. OPENAI_API_KEY 값을 확인해주세요\n"
                                "3. API 키는 'sk-' 또는 'sk-proj-'로 시작해야 합니다\n"
                                "4. 공백이나 따옴표가 포함되지 않았는지 확인해주세요\n"
                                "5. API 키는 https://platform.openai.com/account/api-keys 에서 발급받을 수 있습니다\n"
                                "6. .env 파일 수정 후 Streamlit 앱을 재시작해주세요"
                            )
                        
                        # 임베딩 및 벡터 스토어 생성
                        try:
                            embeddings = OpenAIEmbeddings()
                        except Exception as e:
                            raise Exception(f"OpenAI 임베딩 초기화 실패: {str(e)}\n\nAPI 키를 확인해주세요.")
                        
                        if st.session_state.vectorstore is None:
                            # 새 벡터 스토어 생성
                            batch_size = 30
                            vectorstore = None
                            success_count = 0
                            error_messages = []
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    if vectorstore is None:
                                        vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                                        success_count += 1
                                    else:
                                        vectorstore.add_documents(batch_chunks)
                                        success_count += 1
                                except Exception as e:
                                    error_msg = str(e)
                                    # API 키 오류인 경우 특별 처리
                                    if "401" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                                        raise Exception(
                                            "❌ OpenAI API 키 오류가 발생했습니다.\n\n"
                                            "해결 방법:\n"
                                            "1. .env 파일을 열어주세요\n"
                                            "2. OPENAI_API_KEY 값을 확인해주세요\n"
                                            "3. API 키는 https://platform.openai.com/account/api-keys 에서 발급받을 수 있습니다\n"
                                            "4. API 키는 'sk-'로 시작해야 합니다\n"
                                            "5. .env 파일 수정 후 Streamlit 앱을 재시작해주세요"
                                        )
                                    else:
                                        error_messages.append(f"청크 {i//batch_size + 1} 배치: {error_msg}")
                                        st.warning(f"청크 {i//batch_size + 1} 배치 처리 중 오류: {error_msg}")
                                        continue
                            
                            # 벡터스토어가 성공적으로 생성되었는지 확인
                            if vectorstore is None:
                                raise Exception("벡터스토어 생성에 실패했습니다. 모든 배치 처리 중 오류가 발생했습니다.")
                            
                            if success_count == 0:
                                raise Exception("벡터스토어 생성에 실패했습니다. 처리된 청크가 없습니다.")
                            
                            st.session_state.vectorstore = vectorstore
                            st.success(f"벡터스토어가 성공적으로 생성되었습니다. ({success_count}개 배치 처리됨)")
                        else:
                            # 기존 벡터 스토어에 추가
                            batch_size = 30
                            success_count = 0
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    st.session_state.vectorstore.add_documents(batch_chunks)
                                    success_count += 1
                                except Exception as e:
                                    error_msg = str(e)
                                    # API 키 오류인 경우 특별 처리
                                    if "401" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                                        raise Exception(
                                            "❌ OpenAI API 키 오류가 발생했습니다.\n\n"
                                            "해결 방법:\n"
                                            "1. .env 파일을 열어주세요\n"
                                            "2. OPENAI_API_KEY 값을 확인해주세요\n"
                                            "3. API 키는 https://platform.openai.com/account/api-keys 에서 발급받을 수 있습니다\n"
                                            "4. API 키는 'sk-'로 시작해야 합니다\n"
                                            "5. .env 파일 수정 후 Streamlit 앱을 재시작해주세요"
                                        )
                                    else:
                                        st.warning(f"청크 {i//batch_size + 1} 배치 추가 중 오류: {error_msg}")
                                        continue
                            
                            if success_count > 0:
                                st.success(f"{success_count}개 배치가 기존 벡터스토어에 추가되었습니다.")
                        
                        # 벡터스토어가 존재하는지 확인 후 검색기 생성
                        if st.session_state.vectorstore is None:
                            raise Exception("벡터스토어가 생성되지 않았습니다.")
                        
                        # 검색기 생성 (더 많은 결과와 정확한 검색)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10}  # 검색 결과 수 증가
                        )
                        
                        # 처리된 파일 목록 업데이트
                        st.session_state.processed_files.extend(new_files)
                        
                        # 파일 처리 후 세션 저장
                        if supabase:
                            save_session_to_supabase(st.session_state.current_session_id)
                        
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        # 초기화 후 세션 저장
        if supabase:
            save_session_to_supabase(st.session_state.current_session_id)
        st.rerun()
    
    # API 키 상태 표시
    st.markdown("---")
    st.subheader("🔑 API 키 상태")
    
    # 환경 변수 재로드 버튼
    if st.button("🔄 환경 변수 재로드", use_container_width=True):
        load_dotenv(override=True)
        st.success("환경 변수를 재로드했습니다!")
        st.rerun()
    
    api_key_valid, api_key_error = check_openai_api_key()
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if api_key_valid:
        # API 키의 처음과 끝 부분만 표시 (보안)
        if len(api_key) > 20:
            masked_key = f"{api_key[:10]}...{api_key[-4:]}"
        else:
            masked_key = "***"
        st.success("✅ OpenAI API 키가 설정되어 있습니다")
        st.caption(f"키 형식: `{masked_key}`")
    else:
        st.error(f"❌ {api_key_error}")
        if api_key:
            st.warning(f"⚠️ 현재 설정된 값 (처음 20자): `{api_key[:20]}...`")
            st.warning(f"⚠️ 전체 길이: {len(api_key)}자")
        st.info("💡 .env 파일에 올바른 OPENAI_API_KEY를 추가해주세요")
        with st.expander("📝 .env 파일 설정 예시"):
            st.code("""
# .env 파일 예시
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

# 주의사항:
# - 따옴표 없이 직접 입력
# - 공백 없이 입력
# - 'sk-' 또는 'sk-proj-'로 시작해야 함
# - 전체 키를 복사해서 붙여넣기
# - .env 파일 수정 후 위의 '환경 변수 재로드' 버튼 클릭
            """, language="bash")
        
        # .env 파일 경로 표시
        import pathlib
        env_path = pathlib.Path(".env")
        if env_path.exists():
            st.info(f"📁 .env 파일 위치: `{env_path.absolute()}`")
        else:
            st.warning("⚠️ .env 파일을 찾을 수 없습니다. 프로젝트 루트 디렉토리에 .env 파일을 생성해주세요.")
    
    # 메모리 사용량 표시
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF 파일을 업로드하고 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF 파일을 업로드하고 처리해주세요."})
    else:
        with st.spinner("답변을 생성 중입니다..."):
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
                    """
                    
                    # LLM으로 답변 생성
                    try:
                        # API 키 확인
                        api_key_valid, api_key_error = check_openai_api_key()
                        if not api_key_valid:
                            raise Exception(f"OpenAI API 키 오류: {api_key_error}")
                        
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
                        response = llm.invoke(system_prompt).content
                    except Exception as e:
                        error_msg = str(e)
                        # API 키 오류인 경우 특별 처리
                        if "401" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg or "API key" in error_msg:
                            response = (
                                "❌ OpenAI API 키 오류가 발생했습니다.\n\n"
                                "**해결 방법:**\n"
                                "1. .env 파일을 열어주세요\n"
                                "2. OPENAI_API_KEY 값을 확인해주세요\n"
                                "3. API 키는 https://platform.openai.com/account/api-keys 에서 발급받을 수 있습니다\n"
                                "4. API 키는 'sk-'로 시작해야 합니다\n"
                                "5. .env 파일 수정 후 Streamlit 앱을 재시작해주세요"
                            )
                        else:
                            raise e
                    
                
                # 답변 표시
                with st.chat_message("assistant"):
                    st.write(response)
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 대화 맥락 메모리에 추가 (최근 50개 대화 유지)
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:  # 50개 대화 = 100개 메시지
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                
                # 자동으로 Supabase에 세션 저장
                if supabase:
                    save_session_to_supabase(st.session_state.current_session_id)
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"오류가 발생했습니다: {str(e)}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})