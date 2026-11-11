"""PPO with parameter sharing and a centralised critic (MAPPO-style), CPU-sized.

  actor   obs (55) -> 64 -> 64 -> mean acceleration (2), learned log-std
          ONE network shared by every drone (decentralised execution: each
          drone only feeds it its own observation)
  critic  [obs, team-average obs] (110) -> 64 -> 64 -> value
          only used during training (centralised training)

Rollouts: T steps from one environment with N agents = T*N samples per
update. Advantages by GAE(lambda). Episode ends are time limits, so the value
of the last state is bootstrapped instead of treated as terminal.

Two details that matter (the first version learned nothing without them):
  * actor and critic have SEPARATE optimisers and gradient clipping. With a
    shared clip, the critic's large value-error gradients dominate the
    combined norm and the actor's gradient is scaled to almost nothing.
  * rewards are divided by a running estimate of the return's standard
    deviation, so values stay O(1) whatever the task's reward scale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from swarm.rl.observe import OBS_DIM


def _mlp(inp: int, out: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = 2):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.net(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        dist = self(torch.as_tensor(obs, dtype=torch.float32))
        a = dist.mean if deterministic else dist.sample()
        return a.clamp(-1.0, 1.0).numpy()


class Critic(nn.Module):
    def __init__(self, state_dim: int = 2 * OBS_DIM):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 4
    minibatch: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy: float = 0.001
    value_coef: float = 0.5
    max_grad: float = 0.5


class RewardScaler:
    """Divide rewards by the running std of the discounted return (per agent)."""

    def __init__(self, n: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(n)
        self.count, self.mean, self.m2 = 1e-4, 0.0, 0.0

    def __call__(self, r: np.ndarray, done: bool) -> np.ndarray:
        self.ret = self.ret * self.gamma + r
        for x in self.ret:                       # Welford update
            self.count += 1
            d = x - self.mean
            self.mean += d / self.count
            self.m2 += d * (x - self.mean)
        if done:
            self.ret[:] = 0.0
        std = np.sqrt(self.m2 / self.count) if self.count > 1 else 1.0
        return r / max(std, 1e-3)


@dataclass
class TrainLog:
    updates: list = field(default_factory=list)      # dicts per update


def train(env, total_samples: int, cfg: PPOConfig = PPOConfig(), seed: int = 0,
          actor: Actor | None = None, log_every: int = 10, callback=None) -> tuple[Actor, TrainLog]:
    """Train a shared policy on `env` for about `total_samples` agent-steps."""
    torch.manual_seed(seed)
    actor = actor or Actor(env.obs_dim)
    critic = Critic(2 * env.obs_dim)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    scaler = RewardScaler(env.n, cfg.gamma)
    log = TrainLog()
    mask = env.trainable
    n = env.n
    obs = env.reset()
    ep_return, ep_returns = np.zeros(n), []
    info_acc: dict[str, list] = {}
    per_update = cfg.rollout_steps * int(mask.sum())
    updates = max(1, total_samples // per_update)
    t_start = time.perf_counter()

    for u in range(updates):
        O, S, A, LP, R, V, D = [], [], [], [], [], [], []
        for _ in range(cfg.rollout_steps):
            state = env.global_state(obs)
            with torch.no_grad():
                dist = actor(torch.as_tensor(obs))
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
                v = critic(torch.as_tensor(state, dtype=torch.float32))
            nxt, r_raw, done, info = env.step(a.numpy())
            r = scaler(r_raw, done).astype(np.float32)
            for k, val in info.items():
                if val is not None:
                    info_acc.setdefault(k, []).append(val)
            O.append(obs); S.append(state); A.append(a.numpy()); LP.append(lp.numpy())
            R.append(r); V.append(v.numpy()); D.append(np.full(n, float(done)))
            ep_return += r_raw                    # report returns in the task's own units
            obs = nxt
            if done:
                # time limit: bootstrap from the value of the final state
                with torch.no_grad():
                    R[-1] = R[-1] + cfg.gamma * critic(
                        torch.as_tensor(env.global_state(obs), dtype=torch.float32)).numpy()
                ep_returns.append(float(ep_return[mask].mean()))
                ep_return[:] = 0.0
                obs = env.reset()
        with torch.no_grad():
            last_v = critic(torch.as_tensor(env.global_state(obs), dtype=torch.float32)).numpy()

        # ---- GAE over the rollout (done == episode boundary)
        R, V, D = np.array(R), np.array(V), np.array(D)
        adv = np.zeros_like(R)
        gae = np.zeros(n)
        for t in reversed(range(cfg.rollout_steps)):
            nv = last_v if t == cfg.rollout_steps - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + cfg.gamma * nv * nonterm - V[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + V

        sel = np.s_[:, mask]
        O_ = torch.as_tensor(np.array(O)[sel].reshape(-1, env.obs_dim))
        S_ = torch.as_tensor(np.array(S)[sel].reshape(-1, 2 * env.obs_dim), dtype=torch.float32)
        A_ = torch.as_tensor(np.array(A)[sel].reshape(-1, 2))
        LP_ = torch.as_tensor(np.array(LP)[sel].reshape(-1))
        ADV = torch.as_tensor(adv[sel].reshape(-1), dtype=torch.float32)
        RET = torch.as_tensor(ret[sel].reshape(-1), dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        m = len(O_)
        for _ in range(cfg.epochs):
            perm = torch.randperm(m)
            for i in range(0, m, cfg.minibatch):
                b = perm[i:i + cfg.minibatch]
                dist = actor(O_[b])
                lp = dist.log_prob(A_[b]).sum(-1)
                ratio = (lp - LP_[b]).exp()
                pg = -torch.min(ratio * ADV[b], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV[b]).mean()
                ent = dist.entropy().sum(-1).mean()
                opt_actor.zero_grad()
                (pg - cfg.entropy * ent).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad)
                opt_actor.step()
                vl = ((critic(S_[b]) - RET[b]) ** 2).mean()
                opt_critic.zero_grad()
                (cfg.value_coef * vl).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad)
                opt_critic.step()

        entry = {"update": u + 1, "samples": (u + 1) * per_update,
                 "return": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan"),
                 "seconds": time.perf_counter() - t_start}
        entry.update({k: float(np.mean(v[-200:])) for k, v in info_acc.items() if v})
        log.updates.append(entry)
        if callback is not None:
            callback(entry, actor)
        if log_every and (u + 1) % log_every == 0:
            extra = " ".join(f"{k}={v:.3g}" for k, v in entry.items()
                             if k not in ("update", "samples", "seconds", "return"))
            print(f"  update {u + 1:4d}/{updates}  samples {entry['samples']:8d}  "
                  f"return {entry['return']:8.2f}  {extra}  ({entry['seconds']:.0f} s)", flush=True)
    return actor, log


def save_policy(actor: Actor, path: str, meta: dict) -> None:
    torch.save({"state_dict": actor.state_dict(), "meta": meta}, path)


def load_policy(path: str) -> tuple[Actor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(data["state_dict"])
    actor.eval()
    return actor, data.get("meta", {})
