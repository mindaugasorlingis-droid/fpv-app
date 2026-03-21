#!/bin/bash
# FPV Ground Station launcher
cd "$(dirname "$0")"

echo "=== FPV Ground Station ==="
echo "Starting mediamtx..."
./mediamtx mediamtx.yml &
MEDIAMTX_PID=$!

sleep 1

echo "Starting Flask app on port 8086..."
/opt/homebrew/bin/python3 app.py

# Cleanup on exit
kill $MEDIAMTX_PID 2>/dev/null
