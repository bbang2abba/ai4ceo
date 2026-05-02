# -*- coding: utf-8 -*-
"""E2E: 두 테스트 계정 가입 → (DB에서 이메일 확인 처리 후) 로그인·세션 격리 검증."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

URL = os.getenv("SUPABASE_URL")
ANON = os.getenv("SUPABASE_ANON_KEY")
META = Path(__file__).resolve().parent / ".e2e_two_users_meta.json"
PW = "E2eTest_" + uuid.uuid4().hex[:12] + "!a"


def _client():
    if not URL or not ANON:
        print("FAIL: SUPABASE_URL / SUPABASE_ANON_KEY", file=sys.stderr)
        sys.exit(1)
    return create_client(URL, ANON)


def phase_signup() -> None:
    # Supabase Auth는 example.com 등 일부 도메인을 거부할 수 있음(형식 검증).
    suffix = uuid.uuid4().hex[:12]
    e1 = f"e2e.mu.a.{suffix}@gmail.com"
    e2 = f"e2e.mu.b.{suffix}@gmail.com"
    sb = _client()
    for label, em in (("u1", e1), ("u2", e2)):
        try:
            r = sb.auth.sign_up({"email": em, "password": PW})
            print(f"signup {label}: {em} user={getattr(r.user, 'id', None)}", file=sys.stderr)
        except Exception as ex:
            print(f"signup {label} FAIL: {ex}", file=sys.stderr)
            sys.exit(2)
        if label == "u1":
            time.sleep(8)
    META.write_text(json.dumps({"email1": e1, "email2": e2, "password": PW}, indent=2), encoding="utf-8")
    print(json.dumps({"email1": e1, "email2": e2, "password": PW, "meta_path": str(META)}))


def phase_verify() -> None:
    if not META.is_file():
        print("FAIL: run phase signup first", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(META.read_text(encoding="utf-8"))
    e1, e2 = meta["email1"], meta["email2"]
    pw = meta["password"]

    sb = _client()

    def sign_in(email: str):
        sb.auth.sign_out()
        r = sb.auth.sign_in_with_password({"email": email, "password": pw})
        if not r.session or not r.user:
            raise RuntimeError(f"sign_in failed for {email}")
        return r.user.id

    uid1 = sign_in(e1)
    sid1 = str(uuid.uuid4())
    sb.table("sessions").upsert(
        {
            "id": sid1,
            "session_id": sid1,
            "user_id": uid1,
            "title": "USER1_ONLY_MARKER",
            "chat_history": "[]",
            "conversation_memory": "[]",
            "processed_files": "[]",
        },
        on_conflict="id",
    ).execute()
    rows1 = sb.table("sessions").select("id,title").execute().data or []
    titles1 = {r.get("title") for r in rows1}

    uid2 = sign_in(e2)
    rows2 = sb.table("sessions").select("id,title").execute().data or []
    titles2 = {r.get("title") for r in rows2}

    if "USER1_ONLY_MARKER" in titles2:
        print("FAIL: user2 can see user1 session (RLS broken)", file=sys.stderr)
        sys.exit(3)

    sid2 = str(uuid.uuid4())
    sb.table("sessions").upsert(
        {
            "id": sid2,
            "session_id": sid2,
            "user_id": uid2,
            "title": "USER2_ONLY_MARKER",
            "chat_history": "[]",
            "conversation_memory": "[]",
            "processed_files": "[]",
        },
        on_conflict="id",
    ).execute()

    sign_in(e1)
    rows1b = sb.table("sessions").select("id,title").execute().data or []
    titles1b = {r.get("title") for r in rows1b}

    if "USER2_ONLY_MARKER" in titles1b:
        print("FAIL: user1 can see user2 session (RLS broken)", file=sys.stderr)
        sys.exit(4)
    if "USER1_ONLY_MARKER" not in titles1b:
        print("FAIL: user1 lost own session", file=sys.stderr)
        sys.exit(5)

    print("OK: RLS isolation - user1 sees only own marker; user2 did not see user1 marker.")
    print(f"cleanup: delete from sessions where id in ('{sid1}','{sid2}');  -- optional")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("signup", "verify"))
    args = p.parse_args()
    if args.phase == "signup":
        phase_signup()
    else:
        phase_verify()


if __name__ == "__main__":
    main()
