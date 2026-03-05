from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

ACTIONS = ["scale_up","scale_down","migrate","noop"]

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    def reset(self, seed: Optional[int] = None):
        return

    @abstractmethod
    def predict(self, obs: Dict[str, Any]) -> str:
        ...

    def update(self, transition: Dict[str, Any]):
        return
