"""
AI Agent를 활용한 통합 앱
4개의 tools(시간, 챗봇, 인터넷검색, RAG)을 AI Agent가 자동으로 선택하여 실행
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import time
from typing import List, Any
import logging

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

# Perplexity 임포트
try:
    from langchain_perplexity import ChatPerplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

# OpenAI 임포트
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# 페이지 설정
st.set_page_config(
    page_title="AI Agent 통합 앱",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None

if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = []

if "internet_messages" not in st.session_state:
    st.session_state.internet_messages = []

if "internet_chat_history" not in st.session_state:
    st.session_state.internet_chat_history = []

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "rag_vectorstore" not in st.session_state:
    st.session_state.rag_vectorstore = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "rag_processing_complete" not in st.session_state:
    st.session_state.rag_processing_complete = False

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Tool 추적을 위한 Callback Handler
class ToolTrackingHandler(BaseCallbackHandler):
    """Tool 호출을 추적하는 Callback Handler"""
    def __init__(self):
        self.tool_name = None
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Tool이 시작될 때 호출"""
        self.tool_name = serialized.get("name", None)
    
    def on_tool_end(self, output, **kwargs):
        """Tool이 끝날 때 호출"""
        pass

# ==================== Tool 정의 ====================

@tool
def show_time() -> str:
    """현재 시간과 날짜를 표시합니다. 사용자가 시간이나 날짜를 물어볼 때 사용합니다."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y년 %m월 %d일")
    return f"현재 날짜: {current_date}, 현재 시간: {current_time}"

@tool
def chatbot(query: str) -> str:
    """일반적인 대화나 질문에 답변합니다. 시간, 인터넷 검색, 문서 질문이 아닌 경우 사용합니다.
    
    Args:
        query: 사용자의 질문이나 대화 내용
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다."
    
    client = OpenAI(api_key=openai_api_key)
    
    # 대화 히스토리 구성
    messages = []
    for msg in st.session_state.chatbot_messages[-10:]:  # 최근 10개만 사용
        messages.append(msg)
    
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

@tool
def internet_search(query: str) -> str:
    """최신 정보나 인터넷 검색이 필요한 질문에 답변합니다. 뉴스, 시사, 최신 정보를 물어볼 때 사용합니다.
    
    Args:
        query: 인터넷 검색이 필요한 질문
    """
    if not PERPLEXITY_AVAILABLE:
        return "langchain_perplexity가 설치되지 않았습니다."
    
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        return "PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다."
    
    try:
        llm = ChatPerplexity(
            api_key=perplexity_api_key,
            model="sonar-pro"
        )
        
        # 대화 히스토리 구성
        messages = []
        for msg in st.session_state.internet_chat_history[-10:]:
            messages.append(msg)
        
        messages.append(HumanMessage(content=query))
        response = llm.invoke(messages)
        
        # 히스토리 업데이트
        st.session_state.internet_chat_history.append(HumanMessage(content=query))
        st.session_state.internet_chat_history.append(AIMessage(content=response.content))
        
        return response.content
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

@tool
def rag_search(query: str) -> str:
    """업로드된 PDF 문서에 대한 질문에 답변합니다. 문서 내용을 물어볼 때 사용합니다.
    
    Args:
        query: PDF 문서에 대한 질문
    """
    if not st.session_state.rag_processing_complete:
        return "먼저 PDF 파일을 업로드하고 처리해주세요."
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다."
    
    try:
        retriever = st.session_state.rag_retriever
        relevant_docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        system_message = """너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. 
그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."""
        
        prompt_template = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context}

질문: {query}

답변:"""
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt_template)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# ==================== AI Agent 초기화 ====================

def initialize_agent():
    """AI Agent를 초기화합니다."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=openai_api_key
    )
    
    # Tools 리스트
    tools = [show_time, chatbot, internet_search, rag_search]
    
    # Agent 프롬프트 직접 정의 (chat_history 포함)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자의 질문에 따라 적절한 tool을 선택하여 답변하는 AI Agent입니다.

사용 가능한 tools:
1. show_time: 현재 시간과 날짜를 표시합니다. 사용자가 시간이나 날짜를 물어볼 때 사용합니다.
2. chatbot: 일반적인 대화나 질문에 답변합니다. 시간, 인터넷 검색, 문서 질문이 아닌 경우 사용합니다.
3. internet_search: 최신 정보나 인터넷 검색이 필요한 질문에 답변합니다. 뉴스, 시사, 최신 정보를 물어볼 때 사용합니다.
4. rag_search: 업로드된 PDF 문서에 대한 질문에 답변합니다. 문서 내용을 물어볼 때 사용합니다.

사용자의 질문을 분석하여 가장 적절한 tool을 선택하고 사용하세요."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agent 생성
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Tool 추적을 위한 Callback Handler 생성
    tool_handler = ToolTrackingHandler()
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True,
        callbacks=[tool_handler]
    )
    
    # tool_handler를 agent_executor에 저장하여 나중에 접근 가능하도록
    agent_executor._tool_handler = tool_handler
    
    return agent_executor

# ==================== 시간 앱 함수 ====================
def show_time_app():
    """실시간 시간 표시 앱"""
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .main .block-container {
        padding: 0;
        max-width: 100%;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .time-container {
        background-color: #000000;
        padding: 50px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .time-display {
        font-family: 'Courier New', monospace;
        font-size: 72px;
        color: #00ff00;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 0 0 20px #00ff00;
    }
    .date-display {
        font-family: 'Courier New', monospace;
        font-size: 36px;
        color: #ffff00;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 0 0 10px #ffff00;
    }
    </style>
    """, unsafe_allow_html=True)
    
    placeholder = st.empty()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y년 %m월 %d일")
    
    placeholder.markdown(f"""
    <div class="time-container">
        <div class="date-display">{current_date}</div>
        <div class="time-display">{current_time}</div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(1)
    st.rerun()

# ==================== 챗봇 앱 함수 ====================
def show_chatbot_app():
    """간단한 챗봇 앱"""
    st.title("💬 챗봇")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return
    
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("무엇이든 물어보세요!"):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=st.session_state.chatbot_messages,
                    max_tokens=2048,
                    temperature=0.7,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# ==================== 인터넷 검색 앱 함수 ====================
def show_internet_search_app():
    """인터넷 검색 챗봇 앱"""
    st.title("🌐 인터넷 검색")
    
    if not PERPLEXITY_AVAILABLE:
        st.error("langchain_perplexity가 설치되지 않았습니다.")
        return
    
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        st.error("PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return
    
    @st.cache_resource
    def get_perplexity_llm():
        return ChatPerplexity(
            api_key=perplexity_api_key,
            model="sonar-pro"
        )
    
    llm = get_perplexity_llm()
    
    for message in st.session_state.internet_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("인터넷 검색 질문을 입력하세요!"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 인터넷 검색 중...")
            
            try:
                messages = []
                for msg in st.session_state.internet_chat_history:
                    messages.append(msg)
                
                messages.append(HumanMessage(content=prompt))
                response = llm.invoke(messages)
                full_response = response.content
                
                message_placeholder.markdown(full_response)
                
                st.session_state.internet_messages.append({"role": "user", "content": prompt})
                st.session_state.internet_messages.append({"role": "assistant", "content": full_response})
                
                st.session_state.internet_chat_history.append(HumanMessage(content=prompt))
                st.session_state.internet_chat_history.append(AIMessage(content=full_response))
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# ==================== RAG 앱 함수 ====================
def show_rag_app():
    """RAG 챗봇 앱"""
    st.title("📚 RAG 챗봇")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return
    
    if not st.session_state.rag_processing_complete:
        st.info("왼쪽 사이드바에서 PDF 파일을 업로드하고 처리해주세요.")
    
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요!"):
        if not st.session_state.rag_processing_complete:
            st.warning("먼저 PDF 파일을 업로드하고 처리해주세요.")
        else:
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤖 답변 생성 중...")
                
                try:
                    retriever = st.session_state.rag_retriever
                    relevant_docs = retriever.invoke(prompt)
                    
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    system_message = """너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. 
그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."""
                    
                    prompt_template = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context}

질문: {prompt}

답변:"""
                    
                    llm = ChatOpenAI(
                        model="gpt-4o",
                        temperature=0.7,
                        api_key=openai_api_key,
                        streaming=True
                    )
                    
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=prompt_template)
                    ]
                    
                    full_response = ""
                    for chunk in llm.stream(messages):
                        if chunk.content:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.rag_messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

# ==================== AI Agent 모드 함수 ====================
def show_agent_mode():
    """AI Agent가 자동으로 tool을 선택하는 모드"""
    st.title("🤖 AI Agent 모드")
    st.info("💡 질문을 입력하면 AI Agent가 자동으로 적절한 tool을 선택하여 답변합니다.")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return
    
    # Agent 초기화
    if "agent_executor" not in st.session_state:
        with st.spinner("AI Agent를 초기화하는 중..."):
            st.session_state.agent_executor = initialize_agent()
    
    if st.session_state.agent_executor is None:
        st.error("AI Agent 초기화에 실패했습니다.")
        return
    
    # 대화 히스토리 표시
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요 (AI Agent가 자동으로 tool을 선택합니다)..."):
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤖 AI Agent가 적절한 tool을 선택하는 중...")
            
            try:
                # Tool 추적을 위한 Callback Handler 생성 (매번 새로 생성)
                tool_handler = ToolTrackingHandler()
                
                # Agent 실행 (chat_history를 빈 리스트로 전달, callback 포함)
                result = st.session_state.agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": []
                    },
                    config={"callbacks": [tool_handler]}
                )
                
                # 사용된 tool 정보를 먼저 확인하여 sidebar에 표시
                tool_used = None
                display_name = None
                
                # Callback Handler에서 tool 이름 가져오기
                if hasattr(tool_handler, 'tool_name') and tool_handler.tool_name:
                    tool_used = tool_handler.tool_name
                
                # intermediate_steps 확인 (백업 방법)
                if not tool_used and "intermediate_steps" in result:
                    if result["intermediate_steps"]:
                        for step in result["intermediate_steps"]:
                            if step and len(step) > 0:
                                # step[0]은 AgentAction 객체
                                action = step[0]
                                
                                # tool 속성 확인 (여러 방법 시도)
                                try:
                                    if hasattr(action, 'tool'):
                                        tool_used = action.tool
                                    elif hasattr(action, 'tool_name'):
                                        tool_used = action.tool_name
                                    elif hasattr(action, 'name'):
                                        tool_used = action.name
                                    elif isinstance(action, dict):
                                        tool_used = action.get('tool', action.get('tool_name', action.get('name', None)))
                                    
                                    # tool_used가 문자열이 아니면 변환 시도
                                    if tool_used and not isinstance(tool_used, str):
                                        tool_used = str(tool_used)
                                    
                                    if tool_used:
                                        break
                                except Exception:
                                    continue
                
                # Tool이 선택되었으면 sidebar에 즉시 표시 (답변 전에)
                if tool_used:
                    st.session_state.selected_tool = tool_used
                    # Tool 이름을 한글로 변환
                    tool_names = {
                        "show_time": "⏰ 시간",
                        "chatbot": "💬 챗봇",
                        "internet_search": "🌐 인터넷 검색",
                        "rag_search": "📚 RAG"
                    }
                    display_name = tool_names.get(tool_used, tool_used)
                    # Tool 이름을 메시지에 표시하고 답변 시작
                    message_placeholder.markdown(f"🔧 **선택된 Tool: {display_name}**\n\n답변 생성 중...")
                else:
                    display_name = None
                
                # 답변 표시
                response = result["output"]
                if tool_used and display_name:
                    message_placeholder.markdown(f"🔧 **선택된 Tool: {display_name}**\n\n{response}")
                else:
                    message_placeholder.markdown(response)
                
                st.session_state.agent_messages.append({"role": "assistant", "content": response})
                
                # Sidebar 업데이트를 위해 새로고침
                if tool_used:
                    st.rerun()
                
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})

# ==================== 메인 앱 ====================
# 사이드바
with st.sidebar:
    st.title("🤖 AI Agent 통합 앱")
    st.markdown("---")
    
    # 선택된 tool 표시
    st.subheader("🔧 선택된 Tool")
    tool_name = st.session_state.selected_tool
    # Tool 이름을 한글로 변환
    tool_names = {
        "show_time": "⏰ 시간",
        "chatbot": "💬 챗봇",
        "internet_search": "🌐 인터넷 검색",
        "rag_search": "📚 RAG"
    }
    if tool_name:
        display_name = tool_names.get(tool_name, tool_name)
        st.success(f"✅ {display_name}")
    else:
        st.info("선택된 tool 없음")
    
    st.markdown("---")
    
    # RAG PDF 업로드 및 처리 (항상 표시)
    st.subheader("📚 PDF 파일 업로드 (RAG용)")
    uploaded_files = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_file_uploader"
    )
    
    if uploaded_files and st.button("PDF 처리하기"):
        with st.spinner("PDF 파일을 처리하는 중..."):
            all_docs = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"파일 {uploaded_file.name} 처리 중 오류: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(all_docs)
                
                embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                texts = [doc.page_content for doc in chunks]
                metadatas = [doc.metadata for doc in chunks]
                bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
                bm25_retriever.k = 4
                
                vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )
                
                st.session_state.rag_vectorstore = vectorstore
                st.session_state.rag_retriever = ensemble_retriever
                st.session_state.rag_processing_complete = True
                st.success("✅ PDF 파일 처리가 완료되었습니다!")
            else:
                st.error("처리할 문서가 없습니다.")
    
    st.markdown("---")
    
    # 새로시작하기 버튼
    if st.button("🔄 새로시작하기", type="primary", use_container_width=True):
        st.session_state.chatbot_messages = []
        st.session_state.internet_messages = []
        st.session_state.internet_chat_history = []
        st.session_state.rag_messages = []
        st.session_state.rag_vectorstore = None
        st.session_state.rag_retriever = None
        st.session_state.rag_processing_complete = False
        st.session_state.agent_messages = []
        st.session_state.selected_tool = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ 안내")
    st.info("""
    질문을 입력하면 AI Agent가 자동으로 적절한 tool을 선택합니다:
    - ⏰ **시간**: 시간/날짜 질문
    - 💬 **챗봇**: 일반 대화
    - 🌐 **인터넷 검색**: 최신 정보/뉴스
    - 📚 **RAG**: PDF 문서 질문
    """)

# 메인 화면 - 항상 AI Agent 모드만 사용
show_agent_mode()

