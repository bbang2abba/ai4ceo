import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="SQL Agent 데모",
    page_icon="🗄️",
    layout="wide"
)

def initialize_database():
    """데이터베이스 연결 초기화"""
    try:
        # 현재 작업 디렉토리에서 chinook.db 파일 찾기
        current_dir = os.getcwd()
        db_path = os.path.join(current_dir, "data", "chinook.db")
        
        # data 폴더에 없으면 현재 디렉토리에서 찾기
        if not os.path.exists(db_path):
            db_path = os.path.join(current_dir, "chinook.db")
        
        if not os.path.exists(db_path):
            st.error(f"chinook.db 파일을 찾을 수 없습니다. 경로: {db_path}")
            return None, None, None
        
        # SQLite 데이터베이스 URI 생성
        db_uri = f"sqlite:///{db_path}"
        
        # 데이터베이스 연결
        db = SQLDatabase.from_uri(db_uri)
        
        # LLM 초기화 (Gemini 2.5 Pro 사용)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1  # 낮은 temperature로 일관된 결과 보장
        )
        
        # SQL Agent 생성
        agent_executor = create_sql_agent(
            llm,
            db=db,
            agent_type="openai-tools",
            verbose=True
        )
        
        # SQL 쿼리 생성 체인
        query_chain = create_sql_query_chain(llm, db)
        
        return db, agent_executor, query_chain
        
    except Exception as e:
        st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        return None, None, None

def reset_session():
    """세션 상태 초기화"""
    for key in ['question', 'sql_query', 'result', 'question_asked']:
        if key in st.session_state:
            del st.session_state[key]

def main():
    # 제목
    st.title("🗄️ SQL Agent 데모")
    st.markdown("---")
    
    # 왼쪽 사이드바 - 재시작 버튼
    with st.sidebar:
        st.header("제어판")
        if st.button("🔄 화면 재시작", type="primary", use_container_width=True):
            reset_session()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 데이터베이스 정보")
        st.info("Chinook 샘플 데이터베이스\n\n- 음악 스토어 데이터\n- 고객, 직원, 아티스트, 앨범, 트랙 등")
        
        st.markdown("---")
        st.markdown("### 📊 빠른 질문 예시")
        example_questions = [
            "직원은 모두 몇명인가요?",
            "가장 많이 팔린 곡 5개는?",
            "고객 중에서 가장 구매를 많이한 top 10명은?",
            "가장 많이 팔린 장르는?",
            "총 매출액은 얼마인가요?"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state['question_asked'] = example
                st.session_state['question'] = example
                st.rerun()
    
    # 메인 컨텐츠 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 질문 입력
        st.subheader("💬 질문하기")
        question = st.text_input(
            "자연어로 질문을 입력하세요:",
            placeholder="예: 직원은 모두 몇명인가요?",
            value=st.session_state.get('question', ''),
            key="question_input"
        )
        
        # 질문 제출 버튼
        if st.button("🔍 실행", type="primary"):
            if question:
                st.session_state['question_asked'] = question
                st.session_state['question'] = question
            else:
                st.warning("질문을 입력해주세요.")
    
    with col2:
        pass  # 오른쪽 컬럼은 비워둠
    
    # 질문이 있으면 처리
    if st.session_state.get('question_asked'):
        st.markdown("---")
        
        # 데이터베이스 초기화
        with st.spinner("데이터베이스 연결 중..."):
            db, agent_executor, query_chain = initialize_database()
        
        if db and agent_executor and query_chain:
            try:
                # SQL 쿼리 생성
                with st.spinner("SQL 쿼리 생성 중..."):
                    sql_query = query_chain.invoke({"question": st.session_state['question_asked']})
                    st.session_state['sql_query'] = sql_query
                
                # 생성된 SQL 쿼리 표시
                st.subheader("🔧 생성된 SQL 쿼리")
                st.code(sql_query, language="sql")
                
                # SQL 실행 및 결과 표시
                with st.spinner("쿼리 실행 중..."):
                    try:
                        # SQL Agent를 사용하여 질문 처리
                        result = agent_executor.invoke(st.session_state['question_asked'])
                        st.session_state['result'] = result["output"]
                        
                        # Verbose 실행 과정 표시
                        if "intermediate_steps" in result:
                            st.subheader("🔍 실행 과정")
                            for i, step in enumerate(result["intermediate_steps"]):
                                with st.expander(f"단계 {i+1}: {step.get('action', 'Unknown')}"):
                                    if 'action' in step:
                                        st.write(f"**액션:** {step['action']}")
                                    if 'action_input' in step:
                                        st.write(f"**입력:** {step['action_input']}")
                                    if 'observation' in step:
                                        st.write(f"**관찰:** {step['observation']}")
                                    if 'tool' in step:
                                        st.write(f"**도구:** {step['tool']}")
                    except Exception as e:
                        # Agent 실행 실패 시 직접 SQL 실행
                        try:
                            raw_result = db.run(sql_query)
                            st.session_state['result'] = raw_result
                        except Exception as sql_error:
                            st.session_state['result'] = f"SQL 실행 오류: {str(sql_error)}"
                
                # 실행 결과 표시
                st.subheader("📋 실행 결과")
                if isinstance(st.session_state['result'], str):
                    st.write(st.session_state['result'])
                else:
                    st.write(st.session_state['result'])
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {str(e)}")
        else:
            st.error("데이터베이스 연결에 실패했습니다.")
    

if __name__ == "__main__":
    main()