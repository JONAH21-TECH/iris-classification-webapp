"""
Local Development Server Runner
===============================
Run this file to test the web application on your local machine:
    python app.py
Open http://127.0.0.1:5000 in your browser.
"""

import sys

# Force UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from api.index import app

if __name__ == "__main__":
    print("=" * 60)
    print("  [OK] Iris Classification Web Application Running!")
    print("  --> Open http://127.0.0.1:5000 in your web browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
