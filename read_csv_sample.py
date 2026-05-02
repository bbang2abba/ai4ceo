"""
구내식당train.csv를 읽어 인코딩을 자동 판별하고, SQLite DB(구내식당.db)로 저장합니다.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# 프로젝트 루트 기준 경로
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "prompt" / "5.SQL-Agent" / "구내식당train.csv"
DB_PATH = ROOT / "구내식당.db"
TABLE_NAME = "구내식당train"

# 인코딩 후보: 한글 CSV는 cp949/euc-kr이 흔함
_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

# 터미널에 한글로 설명할 때 쓰는 표기
_ENCODING_LABEL_KO = {
    "utf-8-sig": "UTF-8 (BOM 포함)",
    "utf-8": "UTF-8",
    "cp949": "CP949 (Windows 한글, 확장 완성형)",
    "euc-kr": "EUC-KR",
}


def _hangul_count(s: str) -> int:
    return sum(1 for c in s if "\uac00" <= c <= "\ud7a3")


def detect_encoding(path: Path, sample_size: int = 200_000) -> str:
    """샘플 바이트를 여러 인코딩으로 디코딩해, 한글 비율이 가장 높은 인코딩을 선택합니다."""
    raw = path.read_bytes()[:sample_size]
    best_enc: str | None = None
    best_score = -1

    for enc in _ENCODING_CANDIDATES:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        score = _hangul_count(text)
        if score > best_score:
            best_score = score
            best_enc = enc

    if best_enc is not None and best_score > 0:
        return best_enc

    # 한글이 샘플에 없거나 판별 실패 시: 오류 없이 읽히는 첫 인코딩
    for enc in _ENCODING_CANDIDATES:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue

    return "cp949"


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    # 긴 한글 메뉴 문자열이 잘리지 않도록 표시 폭 조정
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.unicode.east_asian_width", True)

    enc = detect_encoding(CSV_PATH)
    enc_ko = _ENCODING_LABEL_KO.get(enc, enc)
    print("=== 인코딩 ===")
    print(f"자동으로 선택한 인코딩: {enc_ko} (내부 식별자: {enc})")

    df = pd.read_csv(CSV_PATH, encoding=enc)

    print("\n=== CSV를 판다스로 읽은 뒤, 처음 5행 ===")
    print(df.head(5).to_string())
    print()

    if DB_PATH.is_file():
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cur.fetchall()]
        print("=== SQLite 데이터베이스 ===")
        print("테이블 이름 목록:", tables)

        for t in tables:
            cur.execute(f'PRAGMA table_info("{t}")')
            cols = [row[1] for row in cur.fetchall()]
            print(f'테이블 「{t}」의 컬럼 이름:')
            for i, c in enumerate(cols, 1):
                print(f"  {i}. {c}")

        print("\n=== SQLite에서 조회한 처음 5행 ===")
        sample = pd.read_sql_query(
            f'SELECT * FROM "{TABLE_NAME}" LIMIT 5', conn
        )
        print(sample.to_string())
    finally:
        conn.close()

    print(f"\n저장한 DB 파일 경로: {DB_PATH}")


if __name__ == "__main__":
    main()
