import asyncio
import httpx
import time
from dataclasses import dataclass
from gateway.metrics import BATCH_SIZE_HISTOGRAM, TOTAL_REQUESTS

@dataclass
class ParkedRequest:
    text: str
    future: asyncio.Future
    arrived_at: float

class DynamicMicroBatcher:
    def __init__(
        self,
        orchestrator,
        worker_url: str = "http://127.0.0.1:8001/v1/batch-predict",
        max_batch_size: int = 8,
        batch_window_ms: int = 50
    ):
        self.orchestrator = orchestrator
        self.worker_url = worker_url
        self.max_batch_size = max_batch_size
        self.window_sec = batch_window_ms / 1000.0
        self.queue: asyncio.Queue[ParkedRequest] = asyncio.Queue()

    async def submit(self, text: str) -> dict:
        TOTAL_REQUESTS.inc()
        self.orchestrator.active_requests += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(ParkedRequest(text=text, future=future, arrived_at=time.perf_counter()))
        try:
            return await future
        finally:
            self.orchestrator.active_requests -= 1
            self.orchestrator.last_active_ts = time.time()

    async def run_batch_loop(self):
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                # 1. Block asynchronously until at least one request arrives
                first_req = await self.queue.get()
                batch = [first_req]

                # 2. Ensure container is awake (during a cold start, more requests will queue up!)
                wake_info = await self.orchestrator.ensure_awake()

                # 3. Collect additional requests up to max_batch_size within the 50ms window
                deadline = asyncio.get_running_loop().time() + self.window_sec
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        next_req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(next_req)
                    except asyncio.TimeoutError:
                        break

                batch_size = len(batch)
                BATCH_SIZE_HISTOGRAM.observe(batch_size)

                # 4. Dispatch single batched request to the ML worker container
                try:
                    payload = {"items": [req.text for req in batch]}
                    resp = await client.post(self.worker_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                    # 5. Fan-out individual answers back to each waiting HTTP connection
                    for req, result in zip(batch, data["results"]):
                        total_e2e_ms = round((time.perf_counter() - req.arrived_at) * 1000, 2)
                        if not req.future.done():
                            req.future.set_result({
                                "input": result["input"],
                                "confidence": result["confidence"],
                                "gateway_telemetry": {
                                    "wake_type": wake_info["wake_type"],
                                    "wake_latency_ms": wake_info["wake_ms"],
                                    "batch_size_bundled": batch_size,
                                    "worker_compute_ms": data["compute_ms"],
                                    "total_end_to_end_ms": total_e2e_ms
                                }
                            })
                except Exception as exc:
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(exc)
