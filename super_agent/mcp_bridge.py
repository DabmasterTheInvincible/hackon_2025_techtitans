# super_agent/some_module.py

import subprocess
import json

def run_mcp(packet_dict):
    process = subprocess.Popen(
        ["../mcp_service/venv/Scripts/python", "../mcp_service/run_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(json.dumps(packet_dict))
    if process.returncode != 0:
        raise RuntimeError(f"MCP error: {stderr}")
    return json.loads(stdout)
