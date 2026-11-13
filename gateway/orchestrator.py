import asyncio
import time
import docker
import httpx
from gateway.metrics import WORKER_STATE, WAKEUP_LATENCY

STATE_MAP = {"exited": 0, "created": 0, "dead": 0, "paused": 1, "running": 2}

class ContainerOrchestrator:
    def __init__(self, container_name: str = "zg-ml-worker", health_url: str = "http://127.0.0.1:8001/health"):
        self.client = docker.from_env()
        self.container_name = container_name
        self.health_url = health_url
        self.lock = asyncio.Lock()
        self.last_active_ts = time.time()
        self.active_requests = 0

    def get_status(self) -> str:
        container = self.client.containers.get(self.container_name)
        status = container.status
        WORKER_STATE.set(STATE_MAP.get(status, 0))
        return status

    async def ensure_awake(self) -> dict:
        """Ensures the container is running and healthy. Returns wake telemetry."""
        self.last_active_ts = time.time()
        status = await asyncio.to_thread(self.get_status)
        if status == "running":
            return {"wake_type": "already_hot", "wake_ms": 0.0}

        async with self.lock:
            # Re-check status inside lock in case an earlier request just woke it up!
            container = await asyncio.to_thread(self.client.containers.get, self.container_name)
            status = container.status
            if status == "running":
                return {"wake_type": "coalesced_hot", "wake_ms": 0.0}

            start_t = time.perf_counter()

            if status == "paused":
                wake_type = "warm_unpause"
                print("[ORCHESTRATOR] Unpausing container via Linux cgroups freezer...")
                await asyncio.to_thread(container.unpause)
            else:
                wake_type = "cold_start"
                print(f"[ORCHESTRATOR] Cold-starting container from '{status}' state...")
                await asyncio.to_thread(container.start)

            # Poll health endpoint until PyTorch model responds 200 OK
            async with httpx.AsyncClient() as http_client:
                for _ in range(150):  # Up to 30 seconds (150 * 0.2s)
                    try:
                        resp = await http_client.get(self.health_url, timeout=0.4)
                        if resp.status_code == 200:
                            elapsed = time.perf_counter() - start_t
                            WAKEUP_LATENCY.labels(wake_type=wake_type).observe(elapsed)
                            WORKER_STATE.set(2)
                            self.last_active_ts = time.time()
                            print(f"[ORCHESTRATOR] Worker ready via {wake_type} in {elapsed*1000:.1f}ms")
                            return {"wake_type": wake_type, "wake_ms": round(elapsed * 1000, 2)}
                    except Exception:
                        pass
                    await asyncio.sleep(0.2)

            raise RuntimeError("ML Worker failed health check after wake-up.")

    async def pause_worker(self):
        async with self.lock:
            container = await asyncio.to_thread(self.client.containers.get, self.container_name)
            if container.status == "running" and self.active_requests == 0:
                print("[RECONCILER] Idle threshold reached -> Pausing container (0% CPU, Warm RAM)...")
                await asyncio.to_thread(container.pause)
                WORKER_STATE.set(1)

    async def stop_worker(self):
        async with self.lock:
            container = await asyncio.to_thread(self.client.containers.get, self.container_name)
            if container.status in ("running", "paused") and self.active_requests == 0:
                if container.status == "paused":
                    await asyncio.to_thread(container.unpause)
                print("[RECONCILER] Deep idle threshold reached -> Stopping container (0 MB RAM)...")
                await asyncio.to_thread(container.stop, timeout=2)
                WORKER_STATE.set(0)
