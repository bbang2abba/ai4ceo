import time
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Digital Clock", page_icon="⏰", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #020617 45%, #000000 100%);
    }
    .clock-wrap {
        margin-top: 30px;
        text-align: center;
        color: #22d3ee;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .clock-title {
        font-size: 28px;
        color: #f8fafc;
        letter-spacing: 2px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }
    .clock-name {
        font-size: 20px;
        color: #facc15;
        margin-bottom: 14px;
    }
    .digital-time {
        font-size: 78px;
        font-weight: 700;
        letter-spacing: 4px;
        color: #22d3ee;
        text-shadow:
            0 0 7px #22d3ee,
            0 0 18px #22d3ee,
            0 0 35px #0891b2;
        margin: 0;
        line-height: 1.2;
    }
    .digital-date {
        font-size: 24px;
        letter-spacing: 2px;
        color: #fde047;
        text-shadow:
            0 0 6px #facc15,
            0 0 14px #ca8a04;
        margin-top: 8px;
    }
    .caption {
        margin-top: 18px;
        color: #cbd5e1;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

clock_box = st.empty()

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")

    clock_box.markdown(
        f"""
        <div class="clock-wrap">
            <div class="clock-title">Streamlit UI</div>
            <div class="clock-name">최홍석</div>
            <p class="digital-time">{current_time}</p>
            <div class="digital-date">{current_date}</div>
            <div class="caption">Neon Digital Clock</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1)
