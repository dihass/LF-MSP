#!/bin/bash

echo "🚀 Setting up Sepsis App..."

# Stop on error
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 Project directory: $PROJECT_DIR"

# ─────────────────────────────────────────────
# 1. Check Python 3.11
# ─────────────────────────────────────────────
if ! command -v python3.11 &> /dev/null
then
    echo "❌ Python 3.11 is required."
    echo "👉 Install it using: brew install python@3.11"
    exit 1
fi

# ─────────────────────────────────────────────
# 2. Remove old venv
# ─────────────────────────────────────────────
echo "🧹 Removing old virtual environment..."
rm -rf venv

# ─────────────────────────────────────────────
# 3. Create new venv
# ─────────────────────────────────────────────
echo "🐍 Creating virtual environment (Python 3.11)..."
python3.11 -m venv venv

# ─────────────────────────────────────────────
# 4. Activate venv
# ─────────────────────────────────────────────
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# ─────────────────────────────────────────────
# 5. Upgrade pip
# ─────────────────────────────────────────────
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# ─────────────────────────────────────────────
# 6. Install dependencies
# ─────────────────────────────────────────────
echo "📦 Installing dependencies..."
python -m pip install -r requirements.txt

# ─────────────────────────────────────────────
# 7. Verify installation
# ─────────────────────────────────────────────
echo "🔍 Verifying environment..."

PYTHON_PATH=$(which python)
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")

echo "Python path: $PYTHON_PATH"
echo "NumPy version: $NUMPY_VERSION"

# Ensure it's using venv
if [[ "$PYTHON_PATH" != *"venv"* ]]; then
    echo "❌ ERROR: Not using virtual environment!"
    exit 1
fi

# Ensure correct NumPy
if [[ "$NUMPY_VERSION" != "1.26.4" ]]; then
    echo "❌ ERROR: NumPy version mismatch!"
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "👉 Next steps:"
echo "   source venv/bin/activate"
echo "   ./run"