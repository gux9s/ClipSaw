#!/bin/bash
# запуск ClipSaw web
cd "$(dirname "$0")"
export PATH="$HOME/.deno/bin:$HOME/.local/bin:$PATH"
pip3 install -r requirements.txt -q
python3 app.py
