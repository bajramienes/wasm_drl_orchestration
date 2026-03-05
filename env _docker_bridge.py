import random
import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class WASMCloudEnv:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.state = self.reset()
        self.metrics = {
            "task_scheduling": [],
            "energy_efficiency": [],
            "latency": [],
            "scalability": [],
            "resource_allocation": []
        }

    def reset(self):
        # Initialize dummy state
        self.state = {
            "pending_tasks": random.randint(10, 100),
            "current_load": random.uniform(0.3, 0.9),
            "cpu": random.randint(10, 90),
            "mem": random.randint(100, 2000)  # in MB
        }
        return self.state

    def step(self, action: Dict[str, Any]):
        """
        action: dict with keys like {'scale': 2, 'routing_policy': 'round_robin'}
        """
        # Simulated metrics based on agent behavior
        reward = 0
        latency = random.uniform(20, 100) / action.get("scale", 1)
        energy = random.uniform(50, 200) * action.get("scale", 1)
        scheduled_tasks = random.randint(5, 20)
        scalability_score = 1.0 / (1 + abs(self.state['current_load'] - 0.5))
        resource_allocation_eff = random.uniform(0.6, 0.95)

        # Log metrics for analysis
        self.metrics["task_scheduling"].append(scheduled_tasks)
        self.metrics["energy_efficiency"].append(energy)
        self.metrics["latency"].append(latency)
        self.metrics["scalability"].append(scalability_score)
        self.metrics["resource_allocation"].append(resource_allocation_eff)

        # Simulate progression
        self.state = self.reset()
        done = True
        return self.state, reward, done, {}

    def get_metrics_summary(self):
        return {
            metric: sum(values) / len(values) if values else 0
            for metric, values in self.metrics.items()
        }

    def train_episode(self, episode_num: int):
        # Simulate training duration and behavior
        logging.info(f"[TRAIN] Agent {self.agent_name} - Episode {episode_num} training...")
        time.sleep(0.01)  # Simulated training delay
        action = self._generate_action(training=True)
        self.step(action)

    def eval_episode(self, episode_num: int):
        # Simulate evaluation duration and behavior
        logging.info(f"[EVAL] Agent {self.agent_name} - Episode {episode_num} evaluating...")
        time.sleep(0.01)  # Simulated eval delay
        action = self._generate_action(training=False)
        self.step(action)

    def _generate_action(self, training=True):
        if training:
            return {
                "scale": random.choice([1, 2]),
                "routing_policy": random.choice(["round_robin", "least_conn"]),
            }
        else:
            return {
                "scale": random.choice([1, 2, 3]),
                "routing_policy": random.choice(["round_robin", "least_conn", "priority"]),
            }
