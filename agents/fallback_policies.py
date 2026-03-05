import random
from typing import Dict, Any
from .base import Agent, ACTIONS

class ConservativeAgent(Agent):
    # prefers stability: fewer scaling actions
    def predict(self, obs: Dict[str, Any]) -> str:
        if obs.get("backlog", 0) > 50 and obs.get("replicas",1) < 3:
            return "scale_up"
        if obs.get("mem_percent", 0) > 85 and obs.get("replicas",1) > 1:
            return "scale_down"
        return "noop"

class AggressiveAgent(Agent):
    # more exploration
    def predict(self, obs: Dict[str, Any]) -> str:
        if random.random() < 0.15:
            return random.choice(ACTIONS)
        if obs.get("backlog", 0) > 30:
            return "scale_up"
        if obs.get("backlog", 0) < 5 and obs.get("replicas",1) > 1:
            return "scale_down"
        return "noop"

class BalancedAgent(Agent):
    def predict(self, obs: Dict[str, Any]) -> str:
        backlog = obs.get("backlog",0)
        replicas = obs.get("replicas",1)
        if backlog > 40 and replicas < 4:
            return "scale_up"
        if backlog < 10 and replicas > 1:
            return "scale_down"
        if obs.get("latency_ms") and obs["latency_ms"] > 1500 and random.random() < 0.1:
            return "migrate"
        return "noop"
