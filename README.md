# ai4ceo

AI·Streamlit 실험 및 프롬프트 모음 저장소입니다.

## 멀티유저 PDF RAG 챗봇

주요 앱: [`prompt/10.multi-users/multi-users-ref.py`](prompt/10.multi-users/multi-users-ref.py)

```bash
cd prompt/10.multi-users
pip install -r requirements.txt
streamlit run multi-users-ref.py
```

- Supabase Auth·RLS, 세션·임베딩은 사용자별로 분리됩니다.
- LLM API 키는 사이드바에서 입력합니다(저장소에 비밀 키를 넣지 마세요).
- DB 스키마는 `prompt/10.multi-users/` 아래 SQL 파일을 참고하세요.

## GitHub에 처음 올릴 때

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/bbang2abba/ai4ceo.git
git push -u origin main
```
