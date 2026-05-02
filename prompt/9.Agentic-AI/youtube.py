import os
import re
import shutil
import tempfile
from html import unescape
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydub import AudioSegment
from yt_dlp import YoutubeDL


load_dotenv()

MODEL_OPTIONS = {
    "gpt-5.4": "gpt-5.4",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "claude-opus-4-7": "claude-opus-4-7",
}


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def init_state() -> None:
    defaults = {
        "selected_model": "gpt-5.4",
        "youtube_url": "",
        "processing_done": False,
        "processing_error": "",
        "summary": "",
        "transcript": "",
        "chat_history": [],
        "topical_chunks": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def get_llm(model_label: str):
    if model_label == "gpt-5.4":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
        return ChatOpenAI(model=MODEL_OPTIONS[model_label], api_key=api_key, temperature=0.2)

    if model_label == "gemini-3-pro-preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 .env에 없습니다.")
        return ChatGoogleGenerativeAI(model=MODEL_OPTIONS[model_label], google_api_key=api_key, temperature=0.2)

    if model_label == "claude-opus-4-7":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 .env에 없습니다.")
        return ChatAnthropic(model=MODEL_OPTIONS[model_label], anthropic_api_key=api_key, temperature=0.2)

    raise ValueError(f"지원하지 않는 모델입니다: {model_label}")


def download_youtube_audio(url: str, output_dir: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = sanitize_filename(info.get("title", "youtube_audio"))
        return os.path.join(output_dir, f"{title}.mp3")


def _extract_text_from_vtt(vtt_text: str) -> str:
    lines = []
    for line in vtt_text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("WEBVTT"):
            continue
        if "-->" in raw:
            continue
        if raw.isdigit():
            continue
        clean = re.sub(r"<[^>]+>", "", raw)
        clean = unescape(clean).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def fetch_youtube_captions_text(url: str) -> str:
    with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    caption_sets = []
    subtitles = info.get("subtitles") or {}
    automatic_captions = info.get("automatic_captions") or {}

    for lang in ("ko", "ko-KR", "en", "en-US"):
        if lang in subtitles:
            caption_sets.append(subtitles[lang])
        if lang in automatic_captions:
            caption_sets.append(automatic_captions[lang])

    if not caption_sets:
        for source in (subtitles, automatic_captions):
            for tracks in source.values():
                caption_sets.append(tracks)
                break
            if caption_sets:
                break

    if not caption_sets:
        raise ValueError("이 영상은 가져올 수 있는 자막(캡션)이 없습니다. ffmpeg 설치 후 오디오 전사를 사용해주세요.")

    preferred_exts = ("vtt", "srv3", "srv2", "ttml")
    selected_track = None
    for tracks in caption_sets:
        for ext in preferred_exts:
            selected_track = next((t for t in tracks if t.get("ext") == ext and t.get("url")), None)
            if selected_track:
                break
        if selected_track:
            break
        selected_track = next((t for t in tracks if t.get("url")), None)
        if selected_track:
            break

    if not selected_track:
        raise ValueError("자막 URL을 찾을 수 없습니다.")

    resp = requests.get(selected_track["url"], timeout=30)
    resp.raise_for_status()

    transcript = _extract_text_from_vtt(resp.text)
    if not transcript.strip():
        raise ValueError("자막 텍스트를 추출하지 못했습니다.")
    return transcript


def split_audio_to_two_min_chunks(audio_path: str, chunk_dir: str) -> List[str]:
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = 2 * 60 * 1000
    chunk_paths: List[str] = []

    for idx, start in enumerate(range(0, len(audio), chunk_ms)):
        end = min(start + chunk_ms, len(audio))
        chunk = audio[start:end]
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

    return chunk_paths


def transcribe_chunk(chunk_path: str, openai_client: OpenAI) -> Tuple[int, str]:
    chunk_idx = int(Path(chunk_path).stem.split("_")[-1])
    with open(chunk_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
    text = getattr(resp, "text", "") or ""
    return chunk_idx, text.strip()


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> List[str]:
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def pick_relevant_chunks(chunks: List[str], query: str, top_k: int = 6) -> List[str]:
    tokens = {t for t in re.findall(r"[A-Za-z0-9가-힣]+", query.lower()) if len(t) > 1}
    if not tokens:
        return chunks[:top_k]

    scored = []
    for c in chunks:
        lower = c.lower()
        score = sum(lower.count(token) for token in tokens)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for score, c in scored[:top_k] if score > 0]
    return selected if selected else chunks[:top_k]


def llm_text(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


def build_summary_prompt(transcript: str) -> str:
    return (
        "당신은 강의/뉴스/인터뷰 요약 전문 에디터입니다.\n"
        "아래 전사 텍스트만 근거로 한국어 요약을 작성하세요.\n"
        "추측이나 외부 사실 추가는 금지합니다.\n\n"
        "출력 형식(마크다운):\n"
        "## 한눈에 보기\n"
        "- 정확히 5개 bullet\n"
        "- 각 bullet은 1문장\n\n"
        "## 핵심 포인트\n"
        "- 번호 목록 5개\n"
        "- 각 항목은 '주장/사실 -> 왜 중요한지' 구조\n\n"
        "## 결론\n"
        "- 2문장 이내\n\n"
        "## 추천 후속 질문\n"
        "- 정확히 3개\n\n"
        f"[전사 텍스트]\n{transcript[:120000]}"
    )


def build_qa_prompt(question: str, context_text: str) -> str:
    return (
        "당신은 YouTube 전사 기반 Q&A 어시스턴트입니다.\n"
        "반드시 제공된 컨텍스트 안에서만 답변하세요.\n"
        "컨텍스트에 없는 내용은 '전사에서 확인되지 않습니다'라고 명시하세요.\n\n"
        "출력 형식(마크다운):\n"
        "### 답변\n"
        "- 핵심 답변을 3~6문장으로 작성\n\n"
        "### 근거 요약\n"
        "- bullet 2~4개\n"
        "- 각 bullet에 컨텍스트 표현을 짧게 인용(따옴표)\n\n"
        "### 신뢰도\n"
        "- High / Medium / Low 중 하나와 이유 1문장\n\n"
        f"[질문]\n{question}\n\n"
        f"[컨텍스트]\n{context_text[:120000]}"
    )


def format_recent_history(chat_history: List[dict], max_turns: int = 6) -> str:
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]
    lines: List[str] = []
    for msg in recent:
        role = "사용자" if msg.get("role") == "user" else "어시스턴트"
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_qa_prompt_with_history(question: str, context_text: str, recent_history: str) -> str:
    history_block = recent_history if recent_history else "이전 대화 없음"
    return (
        "당신은 YouTube 전사 기반 Q&A 어시스턴트입니다.\n"
        "이전 대화 맥락을 참고하되, 사실 근거는 반드시 제공된 컨텍스트에서만 찾으세요.\n"
        "반드시 제공된 컨텍스트 안에서만 답변하세요.\n"
        "컨텍스트에 없는 내용은 '전사에서 확인되지 않습니다'라고 명시하세요.\n\n"
        "출력 형식(마크다운):\n"
        "### 답변\n"
        "- 핵심 답변을 3~6문장으로 작성\n\n"
        "### 근거 요약\n"
        "- bullet 2~4개\n"
        "- 각 bullet에 컨텍스트 표현을 짧게 인용(따옴표)\n\n"
        "### 신뢰도\n"
        "- High / Medium / Low 중 하나와 이유 1문장\n\n"
        f"[이전 대화]\n{history_block}\n\n"
        f"[질문]\n{question}\n\n"
        f"[컨텍스트]\n{context_text[:120000]}"
    )


def process_video(url: str, selected_model: str, progress, status_text) -> Tuple[str, str, List[str]]:
    ffmpeg_ready = has_ffmpeg()
    openai_client = None
    if ffmpeg_ready:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Whisper 전사를 위해 OPENAI_API_KEY가 필요합니다.")
        openai_client = OpenAI(api_key=openai_api_key)

    work_dir = tempfile.mkdtemp(prefix="yt_qa_")
    audio_dir = os.path.join(work_dir, "audio")
    chunk_dir = os.path.join(work_dir, "chunks")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)

    try:
        progress.progress(5)
        if ffmpeg_ready:
            status_text.info("YouTube 오디오 다운로드 중...")
            audio_path = download_youtube_audio(url, audio_dir)

            progress.progress(25)
            status_text.info("2분 단위 오디오 분할 중...")
            chunk_paths = split_audio_to_two_min_chunks(audio_path, chunk_dir)
            if not chunk_paths:
                raise ValueError("오디오 분할 결과가 없습니다.")

            progress.progress(35)
            status_text.info(f"청크 {len(chunk_paths)}개 병렬 전사 중...")

            transcribed = {}
            max_workers = min(8, max(2, len(chunk_paths)))
            done_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(transcribe_chunk, p, openai_client) for p in chunk_paths]
                for future in as_completed(futures):
                    idx, text = future.result()
                    transcribed[idx] = text
                    done_count += 1
                    progress.progress(min(35 + int((done_count / len(chunk_paths)) * 45), 80))

            ordered_texts = [transcribed[i] for i in sorted(transcribed.keys())]
            transcript = "\n\n".join(t for t in ordered_texts if t)
        else:
            status_text.warning("ffmpeg 미설치 환경입니다. YouTube 자막(캡션)으로 처리합니다.")
            progress.progress(35)
            transcript = fetch_youtube_captions_text(url)
            progress.progress(80)

        if not transcript.strip():
            raise ValueError("전사 결과가 비어 있습니다.")

        status_text.info("선택한 모델로 요약 생성 중...")
        llm = get_llm(selected_model)
        summary_prompt = build_summary_prompt(transcript)
        summary = llm_text(llm, summary_prompt)
        topical_chunks = chunk_text(transcript)

        progress.progress(100)
        status_text.success("처리가 완료되었습니다.")
        return transcript, summary, topical_chunks
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def reset_all() -> None:
    for key in [
        "youtube_url",
        "processing_done",
        "processing_error",
        "summary",
        "transcript",
        "chat_history",
        "topical_chunks",
    ]:
        if key == "youtube_url":
            st.session_state[key] = ""
        elif key == "processing_done":
            st.session_state[key] = False
        elif key in ("chat_history", "topical_chunks"):
            st.session_state[key] = []
        else:
            st.session_state[key] = ""


st.set_page_config(page_title="Youtube Q&A Chatbot", page_icon="🎬", layout="wide")
init_state()
st.title("Youtube Q&A Chatbot")

with st.sidebar:
    st.subheader("설정")
    st.session_state.selected_model = st.selectbox(
        "모델 선택",
        ["gpt-5.4", "gemini-3-pro-preview", "claude-opus-4-7"],
        index=["gpt-5.4", "gemini-3-pro-preview", "claude-opus-4-7"].index(st.session_state.selected_model),
    )
    st.session_state.youtube_url = st.text_input("YouTube URL 입력", value=st.session_state.youtube_url)

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    process_clicked = st.button("동영상 처리하기", type="primary", use_container_width=True)
    reset_clicked = st.button("다시 시작하기", use_container_width=True)

    if reset_clicked:
        reset_all()
        st.rerun()

    if process_clicked:
        url = st.session_state.youtube_url.strip()
        if not url:
            st.session_state.processing_error = "YouTube URL을 입력해주세요."
            status_placeholder.error(st.session_state.processing_error)
        else:
            try:
                st.session_state.processing_error = ""
                transcript, summary, topical_chunks = process_video(
                    url, st.session_state.selected_model, progress_bar, status_placeholder
                )
                st.session_state.transcript = transcript
                st.session_state.summary = summary
                st.session_state.topical_chunks = topical_chunks
                st.session_state.chat_history = []
                st.session_state.processing_done = True
            except Exception as e:
                st.session_state.processing_done = False
                st.session_state.processing_error = f"처리 중 오류: {e}"
                status_placeholder.error(st.session_state.processing_error)

if st.session_state.processing_error:
    st.error(st.session_state.processing_error)

if st.session_state.summary:
    with st.expander("요약 결과", expanded=True):
        st.markdown(st.session_state.summary)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("질문을 입력하세요")
if user_q:
    if not st.session_state.processing_done or not st.session_state.transcript:
        st.warning("먼저 사이드바에서 동영상을 처리해주세요.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        try:
            llm = get_llm(st.session_state.selected_model)
            context_chunks = pick_relevant_chunks(st.session_state.topical_chunks, user_q, top_k=6)
            context_text = "\n\n".join(context_chunks)
            recent_history = format_recent_history(st.session_state.chat_history, max_turns=6)
            qa_prompt = build_qa_prompt_with_history(user_q, context_text, recent_history)
            answer = llm_text(llm, qa_prompt)
        except Exception as e:
            answer = f"답변 생성 중 오류: {e}"

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
