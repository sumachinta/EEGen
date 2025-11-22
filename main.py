import asyncio
import json
import ssl
from pathlib import Path

import websockets

HUB_IP = "stream2.mindfulmakers.xyz"   # or stream1 if you prefer
OUTPUT_FILE = "eeg_stream.jsonl"


async def main():
    print(f"Connecting to {HUB_IP}")
    print(f"Saving incoming EEG messages to {OUTPUT_FILE}")

    # Ensure file exists (optional)
    Path(OUTPUT_FILE).touch(exist_ok=True)

    # Disable SSL certificate verification (for testing only)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Open the file with a normal 'with' (sync context manager)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # WebSocket still uses async context manager
        async with websockets.connect(f"wss://{HUB_IP}", ssl=ssl_context) as ws:
            async for msg in ws:
                try:
                    eeg = json.loads(msg)
                    print("Received EEG message:", eeg)
                except json.JSONDecodeError:
                    # If you want to just skip malformed messages, keep this simple
                    # Or you can log them:
                    # print("Received non-JSON message:", msg)
                    continue

                # Write each JSON object as one line
                f.write(json.dumps(eeg) + "\n")
                f.flush()  # make sure it hits disk

                # Optional: small debug print
                print("Saved EEG sample")


if __name__ == "__main__":
    asyncio.run(main())
