import os
import subprocess
import time
import sys

def run_uvicorn():
    """Run Uvicorn server safely with auto-restart on crash."""
    while True:
        try:
            print("\n🚀 Starting Cyberbullying Detection System...")
            print("🔄 Watching for file changes (Press CTRL+C to stop)\n")

            # Run the FastAPI app (no reload crash issue)
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--reload"
            ])

        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user.")
            break

        except Exception as e:
            print(f"\n⚠️ Server crashed with error: {e}")
            print("Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        run_uvicorn()
    except KeyboardInterrupt:
        print("\n👋 Exiting server gracefully.")
