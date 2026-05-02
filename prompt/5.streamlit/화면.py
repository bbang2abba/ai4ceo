import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(layout="wide", page_title="Streamlit Component Sampler")

st.title("Streamlit UI 컴포넌트 도감")
st.caption("Vibe Coding을 위한 화면 용어 및 시각적 예시 정리")

# 1. 사이드바 (Layout & Control)
with st.sidebar:
    st.header("1. 사이드바 영역 (st.sidebar)")
    st.write("이곳은 고정된 메뉴 영역입니다.")
    mode = st.radio("모드 선택 (st.radio)", ["학습 모드", "실전 모드"])
    st.divider()
    st.info("여기는 st.info 박스입니다.")

# 2. 메인 화면 탭 구성 (Layout)
tab1, tab2, tab3, tab4 = st.tabs(["입력 도구 (Input)", "데이터 (Data)", "레이아웃 (Layout)", "피드백 (Status)"])

# 탭 1: 입력 도구
with tab1:
    st.subheader("사용자 입력 (Input Widgets)")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("이름 입력 (st.text_input)", placeholder="여기에 입력하세요")
        age = st.number_input("나이 입력 (st.number_input)", min_value=0, max_value=100, value=25)
        is_agree = st.checkbox("동의하시겠습니까? (st.checkbox)")
        
    with col2:
        # 태그 선택 (호환성 개선)
        tags = st.multiselect("관심 태그 (st.multiselect)", ["AI", "Python", "Streamlit", "Data"])
        level = st.slider("난이도 조절 (st.slider)", 0, 100, 50)
        # 팝업 대신 expander 사용 (호환성 개선)
        with st.expander("팝업 열기 (st.expander)"):
            st.write("이것은 팝업 내부에 있는 내용입니다.")
            st.text_input("팝업 안의 입력창")

    st.divider()
    st.write("현재 입력된 값 실시간 확인 (st.write):")
    st.json({"이름": name, "나이": age, "동의": is_agree, "태그": tags})

# 탭 2: 데이터 및 차트
with tab2:
    st.subheader("데이터 시각화 (Data & Charts)")
    
    # 가상 데이터 생성
    df = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("데이터프레임 (st.dataframe)")
        st.dataframe(df, use_container_width=True)
        
        st.write("정적 테이블 (st.table)")
        st.table(df.head(3))
        
    with c2:
        st.write("라인 차트 (st.line_chart)")
        st.line_chart(df)
        
        st.write("지표 표시 (st.metric)")
        m1, m2, m3 = st.columns(3)
        m1.metric("온도", "24 °C", "1.2 °C")
        m2.metric("습도", "45 %", "-5 %")
        m3.metric("풍속", "3 m/s", "0.1 m/s")

# 탭 3: 레이아웃 구조
with tab3:
    st.subheader("구조 잡기 (Layouts)")
    
    with st.expander("눌러서 내용 보기 (st.expander)"):
        st.write("숨겨진 내용이 펼쳐집니다. 상세 설명이나 옵션을 넣을 때 좋습니다.")
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    
    st.write("컨테이너 (st.container) 예시:")
    with st.container(border=True):
        st.write("이 내용은 테두리가 있는 컨테이너 안에 있습니다.")
        st.button("컨테이너 내부 버튼")

# 탭 4: 상태 및 피드백
with tab4:
    st.subheader("알림 및 상태 (Status)")
    
    btn_success = st.button("성공 메시지 보기")
    if btn_success:
        st.success("작업이 성공했습니다! (st.success)")
        st.toast("우측 하단을 보세요! (st.toast)")
        
    if st.button("로딩 효과 보기"):
        with st.spinner("처리 중입니다... (st.spinner)"):
            import time
            time.sleep(2)
        st.write("완료되었습니다.")

    st.warning("주의 사항입니다 (st.warning)")
    st.error("에러가 발생했습니다 (st.error)")