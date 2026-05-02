import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.title("My First Chatbot")

if not api_key:
    st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("메시지를 입력하세요...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages,
            )
            answer = response.choices[0].message.content or ""
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
