import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from gateway.orchestrator import ContainerOrchestrator
from gateway.batcher import DynamicMicroBatcher
from gateway.metrics import IDLE_SECONDS_SAVED, ESTIMATED_DOLLARS_SAVED

PAUSE_AFTER_SEC = 20   # Transition RUNNING -> PAUSED after 20s of zero traffic
STOP_AFTER_SEC = 60    # Transition PAUSED  -> STOPPED after 60s of zero traffic
AWS_G4DN_HOURLY_USD = 0.526

orchestrator = ContainerOrchestrator()
batcher = DynamicMicroBatcher(orchestrator=orchestrator, max_batch_size=8, batch_window_ms=50)

async def idle_reconciler_loop():
    """Background daemon that scales down idle containers and logs FinOps savings."""
    check_interval = 2.0
    while True:
        await asyncio.sleep(check_interval)
        status = await asyncio.to_thread(orchestrator.get_status)
        idle_for = time.time() - orchestrator.last_active_ts

        if orchestrator.active_requests == 0:
            if status == "running" and idle_for >= PAUSE_AFTER_SEC:
                await orchestrator.pause_worker()
            elif status == "paused" and idle_for >= STOP_AFTER_SEC:
                await orchestrator.stop_worker()

        # FinOps accounting: if not actively burning resources in 'running' state
        if status in ("paused", "exited", "created"):
            IDLE_SECONDS_SAVED.inc(check_interval)
            ESTIMATED_DOLLARS_SAVED.inc((check_interval / 3600.0) * AWS_G4DN_HOURLY_USD)

@asynccontextmanager
async def lifespan(app: FastAPI):
    batch_task = asyncio.create_task(batcher.run_batch_loop())
    reconciler_task = asyncio.create_task(idle_reconciler_loop())
    yield
    batch_task.cancel()
    reconciler_task.cancel()

app = FastAPI(title="ZeroGate Serverless ML Inference Gateway", lifespan=lifespan)

# Mount Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class PredictRequest(BaseModel):
    text: str

@app.post("/v1/predict")
async def predict(req: PredictRequest):
    return await batcher.submit(req.text)

@app.get("/v1/status")
async def status():
    state = await asyncio.to_thread(orchestrator.get_status)
    idle_sec = round(time.time() - orchestrator.last_active_ts, 1)
    return {
        "worker_container": orchestrator.container_name,
        "current_state": state,
        "active_in_flight_requests": orchestrator.active_requests,
        "idle_seconds": idle_sec,
        "thresholds": {"pause_at_sec": PAUSE_AFTER_SEC, "stop_at_sec": STOP_AFTER_SEC}
    }
