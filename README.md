# ZeroGate ⚡ Serverless ML Inference Gateway

A Knative-inspired, lightweight scale-to-zero API gateway and dynamic micro-batcher for machine learning workloads. Built for single-node deployments to slash AI/ML compute costs without the massive overhead of Kubernetes.

## 🚀 The Problem & Solution
Standard cloud inference endpoints waste up to 80% of their compute budget idling. Standard serverless functions suffer from strict payload limits and massive 10+ second cold starts when loading heavy PyTorch models.

**ZeroGate** sits in front of containerized ML workers and dynamically manages their lifecycle using the Docker Engine API and Linux `cgroups`.

### Key Features
* **Async Connection Parking:** When a request hits a stopped container, the HTTP connection is held open while the container boots, preventing 502 errors.
* **Dynamic Micro-Batching:** Groups concurrent incoming requests within a 50ms window into a single vectorized PyTorch forward pass.
* **Two-Tier Idle Lifecycle:** Transitions idle containers from `RUNNING` -> `PAUSED` (0% CPU, warm RAM) -> `STOPPED` (0 MB RAM, cold state) to maximize FinOps savings.

## 📊 Benchmarks & Performance
| Lifecycle State | Wake-Up Latency | Compute Resource Usage |
| :--- | :--- | :--- |
| **Hot (Running)** | `0.0 ms` | 100% RAM, Active CPU |
| **Warm Freeze (Paused)** | **`33.29 ms`** | 100% RAM, **0.0% CPU** |
| **Cold Boot (Stopped)** | `~3,200 ms` | **0 MB RAM, 0% CPU** |

## 🚦 Quickstart (Local Development)

**1. Clone and Setup**
`git clone https://github.com/Rajdeepak1/zerogate-ml.git`
`cd zerogate-ml`
`pip install -r requirements.txt`

**2. Build the ML Worker Container**
`docker build -t zerogate-worker:v1 ./worker`
`docker create --name zg-ml-worker -p 8001:8001 zerogate-worker:v1`

**3. Run the ZeroGate Control Plane**
`uvicorn gateway.main:app --host 0.0.0.0 --port 8000`

**4. Run the Locust Load Test**
`locust -f benchmarks/locustfile.py --host http://localhost:8000`
