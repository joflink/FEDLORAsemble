# run_all.py

import subprocess
import time

print("Starting server...")
server = subprocess.Popen(["python", "server_app.py"])

time.sleep(3)

clients = []
for i in range(2):
    print(f"Starting client {i}...")
    client = subprocess.Popen(["python", "client_app.py", "--partition-id", str(i)])
    clients.append(client)

try:
    server.wait()
    for c in clients:
        c.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    server.terminate()
    for c in clients:
        c.terminate()
