# -*- coding: utf-8 -*-
"""
호환용 별칭. 배포 시 메인 파일은 multi-users-ref.py 사용을 권장합니다.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "multi-users-ref.py"), run_name="__main__")
