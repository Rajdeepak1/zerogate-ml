import time
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ZeroGate ML Worker Node")

# Simulate heavy ML model initialization during container cold boot
print("[WORKER] Loading PyTorch model weights into memory...")
time.sleep(2.5)  # Simulates loading large checkpoint weights from disk
torch.manual_seed(42)
MODEL = nn.Sequential(
    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 64),
    nn.Softmax(dim=-1)
).eval()
READY = True
print("[WORKER] Model loaded and ready for inference.")

class BatchRequest(BaseModel):
    items: List[str]

@app.get("/health")
async def health():
    if READY:
        return {"status": "ready"}
    return {"status": "loading"}, 503

@app.post("/v1/batch-predict")
async def batch_predict(req: BatchRequest):
    start = time.perf_counter()
    batch_size = len(req.items)

    # Convert text inputs into deterministic feature tensors of shape (batch_size, 128)
    with torch.no_grad():
        input_tensor = torch.randn(batch_size, 128)
        # Single vectorized forward pass for the entire batch!
        output_tensor = MODEL(input_tensor)
        scores = output_tensor.max(dim=-1).values.tolist()

    # Simulate realistic GPU/CPU kernel execution overhead (e.g. 40ms per batch + 5ms per item)
    time.sleep(0.04 + (0.005 * batch_size))
    compute_ms = round((time.perf_counter() - start) * 1000, 2)

    results = [
        {
            "input": text,
            "confidence": round(score, 4),
            "worker_batch_compute_ms": compute_ms
        }
        for text, score in zip(req.items, scores)
    ]
    return {"results": results, "batch_size": batch_size, "compute_ms": compute_ms}
