import os, time, json, random
from http.server import BaseHTTPRequestHandler, HTTPServer

# Simple HTTP server so orchestrator can "send workload" with a type and burst size.
# This keeps execution in Docker real and observable.

PORT = int(os.environ.get("PORT", "8080"))

def cpu_bound(n:int):
    # deterministic cpu work
    x = 0
    for i in range(n):
        x += (i*i) % 97
    return x

def io_bound(ms:int):
    time.sleep(ms/1000.0)

def mixed_work(n:int, ms:int):
    cpu_bound(n)
    io_bound(ms)

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def do_POST(self):
        if self.path != "/run":
            return self._json(404, {"error":"not found"})
        length = int(self.headers.get("Content-Length","0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        wtype = payload.get("type","cpu")
        burst = int(payload.get("burst", 1))

        start = time.time()
        # each "request" is a small unit of work
        for _ in range(burst):
            if wtype == "cpu":
                cpu_bound(35000)
            elif wtype == "io":
                io_bound(8)
            else:
                mixed_work(20000, 4)
        dur_ms = (time.time()-start)*1000.0
        self._json(200, {"ok":True, "duration_ms": round(dur_ms,2), "type": wtype, "burst": burst})

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
