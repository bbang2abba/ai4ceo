#
# RAG 기능이 있는 챗봇을 만들어 봅시다
# PDF 파일을 업로드하고 질문할 수 있는 기능을 추가합니다
# 여러 LLM 모델을 선택할 수 있는 기능을 추가합니다
# 자동 응답 방식 선택 기능을 추가합니다 (인터넷 검색, RAG, 일반 LLM)
#

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="💬",
    layout="wide"
)

# Streamlit 앱 제목 설정
st.header("RAG 챗봇 만들기")

# 사이드바에 PDF 업로드 기능 및 LLM 모델 선택 기능 추가
with st.sidebar:
    st.subheader("설정")
    
    # LLM 모델 선택
    llm_model = st.selectbox(
        "LLM 모델 선택",
        [
            "gpt-4o",
            "claude-3-7-sonnet-latest",
            "o3-mini",
            "exaone3.5:32b"
        ],
        index=0  # 기본값은 gpt-4o
    )
    
    # 응답 방식 선택
    response_mode = st.radio(
        "응답 방식",
        ["자동 선택", "인터넷 검색 사용", "RAG 사용", "일반 LLM 사용"],
        index=0  # 기본값은 자동 선택
    )
    
    # 구분선 추가
    st.markdown("---")
    
    st.subheader("문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        with st.spinner("PDF 파일을 처리 중입니다..."):
            # PDF 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # 임베딩 생성 및 벡터 저장소 생성
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # 세션 상태에 벡터 저장소 저장
            st.session_state.vectorstore = vectorstore
            st.success(f"PDF 파일 '{uploaded_file.name}'이(가) 성공적으로 처리되었습니다!")
            
            # 임시 파일 삭제
            os.unlink(pdf_path)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 대화 내용 표시
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# 채팅 입력 (항상 맨 아래에 고정됨)
input_text = st.chat_input("질문을 해보세요")

if input_text:
    # 사용자 입력을 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": input_text})
    
    # 사용자 메시지 표시
    st.chat_message("user").write(input_text)
    
    # 선택한 LLM 모델에 따라 LLM 초기화
    if llm_model == "gpt-4o":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    elif llm_model == "claude-3-7-sonnet-latest":
        llm = ChatAnthropic(model_name="claude-3-7-sonnet-latest", temperature=0)
    elif llm_model == "o3-mini":
        llm = ChatOllama(model="o3-mini-2025-01-31")
    elif llm_model == "exaone3.5:32b":
        llm = ChatOllama(model="exaone3.5:32b")
    
    # 응답 생성 중 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(f"응답을 생성 중입니다... (모델: {llm_model})")
        
        # 자동 응답 방식 선택 (gpt-4o 사용)
        if response_mode == "자동 선택":
            # 응답 방식 선택을 위한 LLM 초기화 (항상 gpt-4o 사용)
            selector_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
            
            # 응답 방식 선택 프롬프트
            selector_prompt = ChatPromptTemplate.from_template("""
            당신은 사용자 질문을 분석하여 가장 적절한 응답 방식을 선택하는 AI입니다.
            
            다음 세 가지 응답 방식 중 하나를 선택하세요:
            1. "인터넷 검색": 최신 정보, 뉴스, 시사 이슈, 실시간 데이터가 필요한 질문
            2. "RAG": 업로드된 PDF 문서에 관한 질문 (문서가 업로드된 경우에만)
            3. "일반 LLM": 일반적인 지식, 개념 설명, 창의적인 작업에 관한 질문
            
            사용자 질문: {question}
            
            PDF 문서 업로드 여부: {has_pdf}
            
            응답 방식(인터넷 검색, RAG, 일반 LLM 중 하나만 정확히 답변):
            """)
            
            # 응답 방식 선택
            has_pdf = "있음" if "vectorstore" in st.session_state else "없음"
            response_type = selector_llm.invoke(
                selector_prompt.format(question=input_text, has_pdf=has_pdf)
            ).content.strip()
            
            # 응답 방식에 따라 처리
            if "인터넷 검색" in response_type:
                selected_mode = "인터넷 검색 사용"
            elif "RAG" in response_type and has_pdf == "있음":
                selected_mode = "RAG 사용"
            else:
                selected_mode = "일반 LLM 사용"
                
            message_placeholder.markdown(f"응답을 생성 중입니다... (모델: {llm_model}, 방식: {selected_mode})")
        else:
            selected_mode = response_mode
        
        # 인터넷 검색 사용
        if selected_mode == "인터넷 검색 사용":
            try:
                # Tavily 검색 도구 초기화
                search_tool = TavilySearchResults()
                
                # 검색 수행
                search_results = search_tool.invoke({"query": input_text})
                
                # 검색 결과를 기반으로 응답 생성
                search_prompt = ChatPromptTemplate.from_template("""
                다음 검색 결과를 바탕으로 사용자의 질문에 답변해주세요.
                
                검색 결과:
                {search_results}
                
                사용자 질문: {question}
                
                답변:
                """)
                
                # 검색 결과 문자열로 변환
                search_results_str = "\n\n".join([f"제목: {result['title']}\n내용: {result['content']}\n출처: {result['url']}" for result in search_results])
                
                # 응답 생성
                response_text = llm.invoke(
                    search_prompt.format(search_results=search_results_str, question=input_text)
                ).content
                
                # 출처 정보 추가
                sources = set([result['url'] for result in search_results])
                if sources:
                    response_text += "\n\n**출처:**\n" + "\n".join([f"- {source}" for source in sources])
                
            except Exception as e:
                response_text = f"인터넷 검색 중 오류가 발생했습니다: {str(e)}\n일반 LLM을 사용하여 응답합니다.\n\n"
                
                # 일반 LLM 응답으로 대체
                messages = [
                    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 일반적인 지식으로 답변합니다."},
                ]
                
                # 이전 대화 내용 추가
                for msg in st.session_state.messages:
                    messages.append(msg)
                
                # LangChain을 사용하여 챗봇 응답 생성
                response_text += llm.invoke(messages).content
        
        # RAG 사용 (벡터 저장소가 있는 경우)
        elif selected_mode == "RAG 사용" and "vectorstore" in st.session_state:
            # 이전 대화 내용을 메시지 형식으로 변환
            chat_history = []
            for i in range(0, len(st.session_state.chat_history), 2):
                if i+1 < len(st.session_state.chat_history):
                    chat_history.append(HumanMessage(content=st.session_state.chat_history[i]))
                    chat_history.append(AIMessage(content=st.session_state.chat_history[i+1]))
            
            # RAG 체인 생성 (최신 LangChain API 사용)
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # 프롬프트 템플릿 생성
            prompt = ChatPromptTemplate.from_template("""
            다음 대화 내용과 문맥을 바탕으로 질문에 답변해주세요.
            
            문맥: {context}
            
            질문: {input}
            """)
            
            # 문서 체인 생성
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            # 검색 체인 생성
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # 대화 기록과 함께 질문 처리
            response = retrieval_chain.invoke({
                "input": input_text,
                "chat_history": chat_history
            })
            
            response_text = response["answer"]
            
            # 대화 기록 업데이트
            st.session_state.chat_history.append(input_text)
            st.session_state.chat_history.append(response_text)
        
        # 일반 LLM 사용 또는 RAG를 사용하려 했으나 벡터 저장소가 없는 경우
        else:
            # 일반 챗봇 응답 (RAG 없이)
            messages = [
                {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 일반적인 지식으로 답변합니다."},
            ]
            
            # 이전 대화 내용 추가
            for msg in st.session_state.messages:
                messages.append(msg)
            
            # LangChain을 사용하여 챗봇 응답 생성
            response_text = llm.invoke(messages).content
        
        # 응답 표시
        message_placeholder.markdown(response_text)
    
    # 응답을 세션 상태에 추가
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 입력창 초기화를 위한 재실행
    st.rerun()
