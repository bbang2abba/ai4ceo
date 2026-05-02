# -*- coding: utf-8 -*-
"""
Streamlit Cloud / 저장소 루트 진입점 (스펙: Main file path = multi-users-ref.py).

실제 앱: prompt/10.multi-users/multi-users-ref.py

Cloud: Secrets에 SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY 설정.
"""
from pathlib import Path
import runpy

_APP = Path(__file__).resolve().parent / "prompt" / "10.multi-users" / "multi-users-ref.py"
runpy.run_path(str(_APP), run_name="__main__")
