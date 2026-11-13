from prometheus_client import Counter, Gauge, Histogram

# 1. State Gauge: 0 = Stopped (Cold), 1 = Paused (Warm Cgroup Freeze), 2 = Running (Hot)
WORKER_STATE = Gauge(
    "zerogate_worker_state",
    "Current lifecycle state of ML worker (0=Stopped, 1=Paused, 2=Running)"
)

# 2. Cold & Warm Start Latency Histograms
WAKEUP_LATENCY = Histogram(
    "zerogate_wakeup_duration_seconds",
    "Time taken to wake worker container from Stopped or Paused state",
    ["wake_type"],  # "cold_start" vs "warm_unpause"
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0]
)

# 3. Dynamic Batching Efficiency
BATCH_SIZE_HISTOGRAM = Histogram(
    "zerogate_batch_size",
    "Number of concurrent requests bundled into a single ML forward pass",
    buckets=[1, 2, 4, 8, 16, 32]
)

TOTAL_REQUESTS = Counter(
    "zerogate_requests_total",
    "Total inference requests processed by the gateway"
)

# 4. FinOps Cost Savings Counter (Simulated AWS g4dn.xlarge @ $0.526/hour)
IDLE_SECONDS_SAVED = Counter(
    "zerogate_idle_compute_seconds_saved_total",
    "Total seconds the ML worker was kept in Paused or Stopped state"
)
ESTIMATED_DOLLARS_SAVED = Counter(
    "zerogate_estimated_aws_dollars_saved_total",
    "Estimated AWS GPU compute cost saved via scale-to-zero ($0.526/hr baseline)"
)
