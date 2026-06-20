# main.py
import subprocess
import signal
import sys
import time

SERVICES = [
    {
        "name": "detector_service",
        "cmd": [sys.executable, "detector_service/app.py"]
    },
    {
        "name": "verification_service",
        "cmd": [sys.executable, "verification_service/app.py"]
    },
    {
        "name": "frontend_service",
        "cmd": [sys.executable, "frontend_service/app.py"]
    }
]

processes = []


def stop_all():
    print("\nStopping all services...")
    for proc_info in processes:
        proc = proc_info["process"]
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception as e:
                print(f"Error terminating {proc_info['name']}: {e}")

    time.sleep(2)

    for proc_info in processes:
        proc = proc_info["process"]
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    print("All services stopped.")


def signal_handler(sig, frame):
    stop_all()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting services...\n")

    for service in SERVICES:
        proc = subprocess.Popen(service["cmd"])
        processes.append({
            "name": service["name"],
            "process": proc
        })
        print(f"[OK] {service['name']} (pid={proc.pid})")

    print("\nAll services started.")
    print("Press Ctrl+C to stop.")
    print("\nAccess frontend at: http://localhost:5000\n")

    try:
        while True:
            for proc_info in processes:
                proc = proc_info["process"]
                if proc.poll() is not None:
                    print(f"\nService crashed: {proc_info['name']}")
                    stop_all()
                    sys.exit(1)
            time.sleep(5)
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
