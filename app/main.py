import time
from fastapi import FastAPI, HTTPException

app = FastAPI()
leak = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/slow")
def slow():
    time.sleep(3)
    return {"status": "slow"}

@app.get("/error")
def error():
    raise HTTPException(status_code=500, detail="simulated failure")

@app.get("/leak")
def leak_memory():
    leak.append(bytearray(10 * 1024 * 1024))  # +10 MB each call
    return {"leaked_mb": len(leak) * 10}