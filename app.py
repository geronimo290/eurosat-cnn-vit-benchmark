"""
Punto de entrada oficial para Hugging Face Spaces.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.main import demo

if __name__ == '__main__':
    demo.launch()
