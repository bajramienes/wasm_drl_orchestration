from .fallback_policies import ConservativeAgent, AggressiveAgent, BalancedAgent

def make_agent(name: str):
    n = name.lower()
    # fallback mapping (only for smoke tests)
    if n == "costar":
        return ConservativeAgent("costar")
    if n == "tdmpc2":
        return AggressiveAgent("tdmpc2")
    if n == "dreamerv3":
        return BalancedAgent("dreamerv3")
    if n == "hydra":
        return BalancedAgent("hydra")
    if n == "baseline":
        return ConservativeAgent("baseline")
    raise ValueError(f"Unknown agent: {name}")
