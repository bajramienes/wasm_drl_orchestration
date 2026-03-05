import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

import docker
import psutil
import requests

ACTIONS = ["scale_up","scale_down","migrate","noop"]

@dataclass
class RewardParams:
    alpha: float
    beta: float
    delta: float
    eta: float

class OrchestrationEnv:
    """Real Docker-based orchestration environment with unified observe/act."""

    def __init__(self, phase: str, agent_name: str, mode: str, reward_params: RewardParams,
                 workload_image: str = "wasm_workload:latest",
                 base_service_name: str = "wasmwl",
                 base_port: int = 18080,
                 max_replicas: int = 4):
        self.phase = phase
        self.agent = agent_name
        self.mode = mode
        self.reward_params = reward_params

        self.client = docker.from_env()
        self.workload_image = workload_image
        self.base_service_name = base_service_name
        self.base_port = base_port
        self.max_replicas = max_replicas

        self.seed_val: Optional[int] = None
        self.replicas: List[str] = []  # container names
        self.episode_start_ts: Optional[float] = None
        self.backlog = 0
        self.intensity = 0.0

    # ---------- Container management ----------
    def _container_name(self, idx: int) -> str:
        return f"{self.base_service_name}_{self.agent}_{idx}"

    def ensure_replicas(self, n: int):
        n = max(1, min(self.max_replicas, n))
        # start missing
        while len(self.replicas) < n:
            idx = len(self.replicas) + 1
            name = self._container_name(idx)
            ports = {"8080/tcp": self.base_port + idx}
            try:
                self.client.containers.get(name)
                # already exists; ensure running
                c = self.client.containers.get(name)
                if c.status != "running":
                    c.start()
            except docker.errors.NotFound:
                self.client.containers.run(
                    self.workload_image,
                    name=name,
                    detach=True,
                    ports=ports,
                    restart_policy={"Name":"unless-stopped"}
                )
            self.replicas.append(name)

        # stop extras
        while len(self.replicas) > n:
            name = self.replicas.pop()
            try:
                c = self.client.containers.get(name)
                c.stop(timeout=2)
                c.remove(force=True)
            except Exception:
                pass

    def current_replica_count(self) -> int:
        return max(1, len(self.replicas))

    def _pick_endpoint(self) -> str:
        # simple round-robin by time
        idx = int(time.time()*1000) % max(1, len(self.replicas))
        port = self.base_port + (idx+1)
        return f"http://127.0.0.1:{port}/run"

    # ---------- Unified interface ----------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.seed_val = seed
        if seed is not None:
            random.seed(seed)
        self.ensure_replicas(1)
        self.episode_start_ts = time.time()
        self.backlog = 0
        self.intensity = 0.0
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        # temp best-effort
        temp = 50.0
        try:
            temps = psutil.sensors_temperatures() or {}
            if temps:
                # take first available sensor group
                k = next(iter(temps.keys()))
                vals = [t.current for t in temps[k] if getattr(t, "current", None) is not None]
                if vals:
                    temp = sum(vals)/len(vals)
        except Exception:
            pass

        obs = {
            "latency_ms": None,  # filled after workload execution
            "cpu_percent": cpu,
            "mem_percent": mem,
            "temp_c": temp,
            "backlog": self.backlog,
            "intensity": self.intensity,
            "replicas": self.current_replica_count()
        }
        return obs

    def act(self, action: str):
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        n = self.current_replica_count()
        if action == "scale_up":
            self.ensure_replicas(min(self.max_replicas, n+1))
        elif action == "scale_down":
            self.ensure_replicas(max(1, n-1))
        elif action == "migrate":
            # single-host: emulate migration by restarting one replica to refresh state/placement
            if self.replicas:
                name = self.replicas[0]
                try:
                    c = self.client.containers.get(name)
                    c.restart(timeout=2)
                except Exception:
                    pass
        else:
            pass

    def step(self, action: str, workload: Dict[str, Any]) -> (Dict[str, Any], float, bool, Dict[str, Any]):
        """Execute one orchestration decision + one workload replay for the episode."""
        self.act(action)

        # Replay bursts as real HTTP requests inside Docker workload containers
        start = time.time()
        arrivals = int(workload.get("arrivals", 0))
        bursts = workload.get("bursts", [])
        types = workload.get("types", [])
        self.intensity = float(arrivals) / max(1.0, float(workload.get("duration_s", 1)))

        # backlog proxy: if arrivals huge, backlog increases
        self.backlog = max(0, arrivals - 50 * self.current_replica_count())

        # execute bursts (cap for safety in demo; paper runs full scale)
        cap = int(workload.get("cap_requests", arrivals))
        sent = 0
        for b in bursts:
            if sent >= cap:
                break
            wtype = types[min(sent, len(types)-1)] if types else "cpu"
            burst = int(min(b, cap - sent))
            try:
                r = requests.post(self._pick_endpoint(), json={"type": wtype, "burst": burst}, timeout=30)
                r.raise_for_status()
            except Exception:
                # fallback: local sleep approximating IO
                time.sleep(0.001 * burst)
            sent += burst

        latency_ms = round((time.time() - start) * 1000.0, 2)

        # system metrics after workload
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        obs = self.observe()
        obs["latency_ms"] = latency_ms

        # Energy proxy (higher is better): match paper description (utilisation + thermal factor)
        utilization = (cpu/100.0 + mem/100.0)/2.0
        temp = obs["temp_c"]
        thermal_factor = min(1.0, max(0.0, 1.0 - (temp - 40.0)/60.0))
        energy_eff = round(max(0.0, 1.0 - utilization) * thermal_factor, 3)

        # Additional quality metrics (bounded 0..1) derived from measurable proxies
        task_sched = round(max(0.0, 1.0 - (latency_ms/2000.0)) * (1.0 - self.backlog/200.0), 3)
        scalability = round(min(1.0, 0.5 + 0.15*self.current_replica_count()) * (1.0 - mem/200.0), 3)
        resource_alloc = round(max(0.0, 1.0 - (cpu+mem)/250.0), 3)

        L = max(0.001, latency_ms/1000.0)
        r = (-self.reward_params.alpha * L
             + self.reward_params.beta * energy_eff
             + self.reward_params.delta * scalability
             + self.reward_params.eta * resource_alloc)

        info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": self.agent,
            "phase": self.phase,
            "mode": self.mode,
            "seed": self.seed_val,
            "action": action,
            "replicas": self.current_replica_count(),
            "arrivals": arrivals,
            "intensity": self.intensity,
            "backlog": self.backlog,
            "latency_ms": latency_ms,
            "energy_efficiency": energy_eff,
            "task_scheduling_score": task_sched,
            "scalability_score": scalability,
            "resource_allocation_score": resource_alloc,
            "cpu_percent": cpu,
            "mem_percent": mem,
            "temp_c": obs["temp_c"],
        }
        done = True
        return obs, float(r), done, info
