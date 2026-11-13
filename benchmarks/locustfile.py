from locust import HttpUser, task, between
import random

class MLInferenceUser(HttpUser):
    # Each simulated user waits between 0.05 and 0.2 seconds between requests
    wait_time = between(0.05, 0.2)

    @task
    def predict(self):
        sample_id = random.randint(1, 100000)
        self.client.post(
            "/v1/predict",
            json={"text": f"High-concurrency inference request #{sample_id}"}
        )
