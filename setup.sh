#!/bin/bash
# ai4ceo Environment Setup - Professor Jang
#
# 사전에 Python / pip / uv 가 없어도 됩니다.
# 이 스크립트가 uv를 설치하고, uv가 지정 버전 Python을 내려받은 뒤 가상환경을 만듭니다.
#
# 필요한 것: bash, sh, 그리고 curl 또는 wget (HTTPS 다운로드용)
#
# Windows: 이 파일(setup.sh)은 사용하지 말고 PowerShell에서 아래 중 하나를 실행하세요.
#   .\setup          (확장자 생략 — setup.ps1 과 동일)
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
#   setup.cmd / setup.bat 더블클릭(내부적으로 setup.ps1 실행)
#
# Linux 등: Usage: bash setup.sh  (or chmod +x setup.sh && ./setup.sh)

# Configuration
PYTHON_VER="3.13.1"
UV_BIN_DIR="$HOME/.local/bin"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

# --- HTTP GET to stdout: curl 우선, 없으면 wget (Python 불필요) ---
fetch_url() {
    local url="$1"
    if command -v curl &> /dev/null; then
        curl -LsSf "$url"
        return $?
    fi
    if command -v wget &> /dev/null; then
        wget -qO- "$url"
        return $?
    fi
    return 127
}

echo "============================================================"
echo "[0/6] 선행 도구 확인 (시스템 Python·pip·uv 없어도 진행 가능)"
echo "============================================================"

if ! command -v sh &> /dev/null; then
    echo "[ERROR] 'sh' 가 없습니다. POSIX 환경이 필요합니다."
    exit 1
fi

if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
    echo "[ERROR] HTTPS 다운로드를 위해 'curl' 또는 'wget' 중 하나가 필요합니다."
    echo "  - Debian/Ubuntu: sudo apt install curl"
    echo "  - Fedora: sudo dnf install curl"
    exit 1
fi

echo "✓ 다운로드 도구 확인 완료 (curl 또는 wget)"
echo ""

echo "============================================================"
echo "[1/6] Checking for 'uv' package manager..."
echo "============================================================"
export PATH="$UV_BIN_DIR:$PATH"

if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing via official script (no Python required)..."
    # pipefail: 다운로드 실패 시 파이프 끝의 sh만 성공하는 것을 막음
    if ! ( set -o pipefail; fetch_url "$UV_INSTALL_URL" | sh ); then
        echo "[ERROR] uv 설치 스크립트 실행에 실패했습니다. 네트워크·방화벽·프록시를 확인하세요."
        exit 1
    fi
    export PATH="$UV_BIN_DIR:$PATH"

    if ! command -v uv &> /dev/null; then
        if [ -f "$UV_BIN_DIR/uv" ]; then
            UV_CMD="$UV_BIN_DIR/uv"
            echo "Using uv from: $UV_BIN_DIR"
        else
            echo "[ERROR] uv installation may have failed."
            echo "  터미널을 새로 연 뒤 PATH에 $UV_BIN_DIR 가 포함되는지 확인하세요."
            exit 1
        fi
    else
        UV_CMD="uv"
        echo "'uv' installation completed and verified."
    fi
else
    UV_CMD="uv"
    echo "'uv' is already installed."
fi

echo ""
echo "============================================================"
echo "[2/6] Adding uv to PATH permanently..."
echo "============================================================"
SHELL_RC=""
if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

if [ -f "$SHELL_RC" ]; then
    if grep -q "\.local/bin" "$SHELL_RC" 2>/dev/null; then
        echo "✓ uv path already exists in user PATH"
    else
        echo "" >> "$SHELL_RC"
        echo "# uv package manager" >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "✓ Added uv path to $SHELL_RC"
        echo "Please run: source $SHELL_RC"
    fi
else
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" > "$SHELL_RC"
    echo "✓ Created $SHELL_RC with uv directory"
fi

echo ""
echo "============================================================"
echo "[3/6] Installing managed Python $PYTHON_VER (via uv, no system Python needed)..."
echo "============================================================"
$UV_CMD python install "$PYTHON_VER"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Python $PYTHON_VER"
    exit 1
fi

echo ""
echo "============================================================"
echo "[4/6] pyproject.toml 확인 및 기존 .venv 정리"
echo "============================================================"
if [ ! -f "pyproject.toml" ]; then
    echo "[ERROR] pyproject.toml not found!"
    echo "Please run this script in the project root folder."
    exit 1
fi

echo ""
echo "--- 기존 가상환경(.venv) 정리 ---"
if [ -d ".venv" ]; then
    echo "Attempting to remove old .venv directory..."
    if command -v pkill &> /dev/null; then
        pkill -f "python.*\.venv" 2>/dev/null || true
        sleep 3
    fi

    if rm -rf .venv 2>/dev/null; then
        echo "Old .venv directory removed successfully."
    else
        echo "[ERROR] Could not remove .venv directory automatically."
        echo "Please manually delete the .venv folder and run this script again."
        echo ""
        echo "You can delete it by running:"
        echo "  rm -rf .venv"
        echo ""
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "[5/6] Synchronizing environment (uv sync)..."
echo "============================================================"
echo "This may take 1-3 minutes (Torch excluded)."

$UV_CMD sync --python "$PYTHON_VER"
if [ $? -ne 0 ]; then
    echo "[ERROR] Synchronization failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "[6/6] Verification"
echo "============================================================"
$UV_CMD pip list | grep -iE "langchain|openai|streamlit|python" || true

echo ""
echo "============================================================"
echo "Environment setup completed successfully! (Python $PYTHON_VER)"
echo "============================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
