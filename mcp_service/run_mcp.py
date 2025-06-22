# tscc-platform/mcp_service/run_mcp.py

from mcp import run_packet
import sys, json

if __name__ == "__main__":
    packet = json.loads(sys.stdin.read())
    result = run_packet(packet)
    print(json.dumps(result))
