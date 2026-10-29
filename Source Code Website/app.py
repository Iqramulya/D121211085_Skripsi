#!/usr/bin/env python
"""Flask TFT App - Main Entry Point"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 50)
    print("🪙 Gold Price Predictor - TFT Model")
    print("=" * 50)
    print(f"Starting server at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=True)

