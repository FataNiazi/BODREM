"""Manipulation training pipeline.

Extracted from `bo_doraemon_paperlike_manipulation.ipynb`.

Provides three training entry points used by the ablation runner:
- `run_doraemon_only(...)`   — DORAEMON Beta-DR adaptation, no outer BO loop
- `run_bo_only(...)`         — Bayesian Optimization over the DR center, fixed-physics training
- `run_bo_doraemon_paperlike(...)` — BO + DORAEMON (BODREM)

All runs share the same SAC agent / env factory / target evaluation utilities.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import mujoco
from tinysim_mujoco.manipulation.push_env import ManipulationEnvV0
from scipy.special import betaln, digamma

from skopt import Optimizer
from skopt.space import Real

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────────────────────
# SAC components
# ────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class ContinuousReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.buffer: list = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        item = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: str):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.stack(s),  dtype=torch.float32, device=device),
            torch.tensor(np.stack(a),  dtype=torch.float32, device=device),
            torch.tensor(r,  dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.stack(ns), dtype=torch.float32, device=device),
            torch.tensor(d,  dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x_t  = dist.rsample()
        action   = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        inp = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class SACConfig:
    state_dim:       int   = 15    # 12 obs + 3 goal
    action_dim:      int   = 3     # EE delta XYZ
    hidden_dim:      int   = 256
    actor_lr:        float = 3e-4
    critic_lr:       float = 3e-4
    alpha_lr:        float = 3e-4
    gamma:           float = 0.98
    tau:             float = 0.005
    buffer_capacity: int   = 100_000
    batch_size:      int   = 256
    init_alpha:      float = 0.2
    auto_alpha:      bool  = True


class SACAgent:
    def __init__(self, cfg: SACConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        self.actor = GaussianActor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.critic = TwinQNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.critic_target = TwinQNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.tensor(
            np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy  = -float(cfg.action_dim)

        self.memory = ContinuousReplayBuffer(cfg.buffer_capacity)
        self.training_losses: list[float] = []
        self._episode_counter = 0

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if training:
                action, _ = self.actor.sample(s)
            else:
                action = self.actor.deterministic(s)
            return action.squeeze(0).cpu().numpy()

    def store(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)

    def learn_step(self) -> float | None:
        if len(self.memory) < self.cfg.batch_size:
            return None
        s, a, r, ns, d = self.memory.sample(self.cfg.batch_size, self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            q1n, q2n = self.critic_target(ns, na)
            q_target = r + self.cfg.gamma * (1.0 - d) * (torch.min(q1n, q2n) - alpha * nlp)
        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_a, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tp.data)

        loss_val = float(critic_loss.item())
        self.training_losses.append(loss_val)
        return loss_val

    def end_episode(self):
        self._episode_counter += 1

    def state_payload(self) -> dict[str, Any]:
        return {
            "actor_state_dict":         self.actor.state_dict(),
            "critic_state_dict":        self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer":          self.actor_optimizer.state_dict(),
            "critic_optimizer":         self.critic_optimizer.state_dict(),
            "log_alpha":                self.log_alpha.detach().cpu().item(),
            "alpha_optimizer":          self.alpha_optimizer.state_dict(),
            "training_losses":          self.training_losses[-1000:],
            "episode_counter":          self._episode_counter,
            "hparams":                  vars(self.cfg),
        }

    def load_payload(self, payload: dict[str, Any]):
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        self.critic_target.load_state_dict(
            payload.get("critic_target_state_dict", payload["critic_state_dict"])
        )
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if "log_alpha" in payload:
            self.log_alpha.data.fill_(float(payload["log_alpha"]))
        if "alpha_optimizer" in payload:
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.training_losses = list(payload.get("training_losses", []))
        self._episode_counter = int(payload.get("episode_counter", 0))


# ────────────────────────────────────────────────────────────────────────────
# Agent helpers
# ───────────────────────────────────────────────────────────────────────────────

def make_agent_from_hparams(hparams: dict | None, device: str = DEVICE) -> SACAgent:
    if hparams is None:
        cfg = SACConfig()
    else:
        cfg = SACConfig(**{k: v for k, v in hparams.items() if k in SACConfig.__annotations__})
    return SACAgent(cfg=cfg, device=device)


def load_warmstart_agent(checkpoint_path: str | Path, device: str = DEVICE) -> SACAgent:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    hparams = payload.get("hparams")
    agent = make_agent_from_hparams(hparams, device=device)
    agent.load_payload(payload)
    return agent


def save_agent_checkpoint(agent: SACAgent, path: str | Path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_payload()
    payload["extra"] = extra or {}
    torch.save(payload, str(path))


def clone_agent(agent: SACAgent) -> SACAgent:
    cloned = make_agent_from_hparams(vars(agent.cfg), device=agent.device)
    cloned.load_payload(agent.state_payload())
    return cloned


# ────────────────────────────────────────────────────────────────────────────
# Manipulation env factory + obs flatten
# ────────────────────────────────────────────────────────────────────────────

def make_manipulation_env(friction: float, mass: float) -> ManipulationEnvV0:
    env = ManipulationEnvV0(headless=True)
    geom_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
    env._model.geom_friction[geom_id, 0] = float(friction)
    body_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    env._model.body_mass[body_id] = float(mass)
    s = 0.03  # box half-size from table.xml
    inertia_diag = mass / 3.0 * (s**2 + s**2)
    env._model.body_inertia[body_id] = np.array([inertia_diag, inertia_diag, inertia_diag])
    mujoco.mj_forward(env._model, env._data)
    return env


def flatten_obs(obs_dict: dict) -> np.ndarray:
    return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]]).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Beta distribution helpers (DORAEMON)
# ────────────────────────────────────────────────────────────────────────────

def normalize_params(friction: float, mass: float, bounds: dict) -> np.ndarray:
    zf = (float(friction) - bounds["friction"][0]) / (bounds["friction"][1] - bounds["friction"][0])
    zm = (float(mass) - bounds["mass"][0]) / (bounds["mass"][1] - bounds["mass"][0])
    return np.array([zf, zm], dtype=np.float64)


def denormalize_params(z: np.ndarray, bounds: dict) -> tuple[float, float]:
    friction = bounds["friction"][0] + float(z[0]) * (bounds["friction"][1] - bounds["friction"][0])
    mass     = bounds["mass"][0]     + float(z[1]) * (bounds["mass"][1]     - bounds["mass"][0])
    return float(friction), float(mass)


def beta_from_mean_conc(mean: np.ndarray, conc: np.ndarray):
    m = np.clip(np.asarray(mean, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    c = np.clip(np.asarray(conc, dtype=np.float64), 2.0001, None)
    return m * c, (1.0 - m) * c


def mean_conc_from_beta(alpha: np.ndarray, beta: np.ndarray):
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    c = a + b
    return a / np.clip(c, 1e-8, None), c


def make_beta_dist_at_mu(mu: np.ndarray, conc: np.ndarray, bounds: dict) -> dict:
    mu_c = np.array([
        float(np.clip(mu[0], bounds["friction"][0], bounds["friction"][1])),
        float(np.clip(mu[1], bounds["mass"][0], bounds["mass"][1])),
    ], dtype=np.float64)
    z = normalize_params(mu_c[0], mu_c[1], bounds)
    a, b = beta_from_mean_conc(z, conc)
    return {"alpha": a, "beta": b, "bounds": bounds}


def beta_logpdf(z, alpha, beta):
    z = np.clip(np.asarray(z, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    return (a - 1.0) * np.log(z) + (b - 1.0) * np.log(1.0 - z) - betaln(a, b)


def beta_entropy_fn(alpha, beta):
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    return betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)


def dist_entropy(dist: dict) -> float:
    h = beta_entropy_fn(dist["alpha"], dist["beta"])
    s1 = np.log(dist["bounds"]["friction"][1] - dist["bounds"]["friction"][0])
    s2 = np.log(dist["bounds"]["mass"][1] - dist["bounds"]["mass"][0])
    return float(np.sum(h) + s1 + s2)


def beta_kl(a0, b0, a1, b1):
    return (betaln(a1, b1) - betaln(a0, b0)
            + (a0 - a1) * digamma(a0) + (b0 - b1) * digamma(b0)
            + (a1 + b1 - a0 - b0) * digamma(a0 + b0))


def dist_kl(old: dict, new: dict) -> float:
    return float(np.sum(beta_kl(old["alpha"], old["beta"], new["alpha"], new["beta"])))


def sample_beta_params(dist: dict, rng: np.random.Generator):
    z = rng.beta(dist["alpha"], dist["beta"])
    z = np.clip(z, 1e-6, 1.0 - 1e-6)
    friction, mass = denormalize_params(z, dist["bounds"])
    return friction, mass, z.astype(np.float64)


def is_success_estimate(z_samples, success, old_dist, new_dist, min_log_w=-20.0, max_log_w=20.0):
    if len(z_samples) == 0:
        return 0.0
    z = np.clip(np.asarray(z_samples, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    s = np.asarray(success, dtype=np.float64)
    logp_old = np.sum(beta_logpdf(z, old_dist["alpha"], old_dist["beta"]), axis=1)
    logp_new = np.sum(beta_logpdf(z, new_dist["alpha"], new_dist["beta"]), axis=1)
    logw = np.clip(logp_new - logp_old, min_log_w, max_log_w)
    w = np.exp(logw - np.max(logw))
    w = w / np.clip(np.sum(w), 1e-12, None)
    return float(np.sum(w * s))


def doraemon_update_beta(
    old_dist, z_samples, success_samples,
    alpha_target, kl_max, rng,
    n_candidates, mean_std, logc_std,
    c_min, c_max, min_log_w, max_log_w,
    freeze_mean=True,
):
    m_old, c_old = mean_conc_from_beta(old_dist["alpha"], old_dist["beta"])
    candidates = [(m_old.copy(), c_old.copy())]
    for _ in range(int(n_candidates)):
        m = m_old.copy() if freeze_mean else np.clip(m_old + rng.normal(0, mean_std, size=2), 1e-4, 1 - 1e-4)
        c = np.exp(np.log(np.clip(c_old, 1e-8, None)) + rng.normal(0, logc_std, size=2))
        c = np.clip(c, c_min, c_max)
        candidates.append((m, c))

    best_feasible = None
    backup = None
    for m, c in candidates:
        a_new, b_new = beta_from_mean_conc(m, c)
        cand = {"alpha": a_new, "beta": b_new, "bounds": old_dist["bounds"]}
        kl = dist_kl(old_dist, cand)
        if kl > kl_max:
            continue
        g_hat = is_success_estimate(z_samples, success_samples, old_dist, cand, min_log_w, max_log_w)
        ent = dist_entropy(cand)
        if backup is None or g_hat > backup["g_hat"] or (np.isclose(g_hat, backup["g_hat"]) and ent > backup["entropy"]):
            backup = {"dist": cand, "g_hat": float(g_hat), "entropy": float(ent), "kl": float(kl), "mode": "backup_success"}
        if g_hat >= alpha_target:
            if best_feasible is None or ent > best_feasible["entropy"]:
                best_feasible = {"dist": cand, "g_hat": float(g_hat), "entropy": float(ent), "kl": float(kl), "mode": "feasible_entropy"}

    if best_feasible is not None:
        return best_feasible["dist"], best_feasible
    if backup is not None:
        return backup["dist"], backup
    return old_dist, {"dist": old_dist, "g_hat": 0.0, "entropy": dist_entropy(old_dist), "kl": 0.0, "mode": "no_candidate"}


# ──────────────────────────────────────────────────────────────────────────────
# Target evaluation
# ────────────────────────────────────────────────────────────────────────────

def target_eval_manipulation(agent, n_eval, max_steps, target_params, seed):
    successes = 0
    for ep in range(n_eval):
        env = make_manipulation_env(friction=target_params["friction"], mass=target_params["mass"])
        obs_dict, _ = env.reset()
        obs = flatten_obs(obs_dict)
        solved = False
        for _ in range(max_steps):
            a = agent.act(obs, training=False)
            obs_dict, reward, terminated, truncated, _ = env.step(a)
            obs = flatten_obs(obs_dict)
            if terminated:
                solved = True
                break
            if truncated:
                break
        successes += int(solved)
    rate = float(successes / max(1, n_eval))
    return rate, int(successes), int(n_eval)


# ────────────────────────────────────────────────────────────────────────────
# Pre-training (warm-start)
# ────────────────────────────────────────────────────────────────────────────

def pretrain_manipulation_agent(
    n_episodes=500,
    friction=1.0, mass=0.1,
    max_steps=128,
    seed=42, save_path=None,
    device=DEVICE,
):
    set_global_seed(seed)
    agent = make_agent_from_hparams(None, device=device)
    rewards_log = []
    for ep in range(n_episodes):
        env = make_manipulation_env(friction=friction, mass=mass)
        obs_dict, _ = env.reset()
        obs = flatten_obs(obs_dict)
        ep_reward = 0.0
        for _ in range(max_steps):
            a = agent.act(obs, training=True)
            obs_dict, reward, terminated, truncated, _ = env.step(a)
            next_obs = flatten_obs(obs_dict)
            done = terminated or truncated
            agent.store(obs, a, reward, next_obs, float(done))
            agent.learn_step()
            obs = next_obs
            ep_reward += reward
            if done:
                break
        agent.end_episode()
        rewards_log.append(ep_reward)
        if (ep + 1) % 50 == 0:
            avg = float(np.mean(rewards_log[-50:]))
            print(f"  pretrain ep {ep+1:4d}/{n_episodes} | avg_reward(50)={avg:.2f} | alpha={agent.alpha:.4f}")
    if save_path is not None:
        save_agent_checkpoint(agent, save_path, extra={"pretrain_episodes": n_episodes})
        print(f"Saved warm-start checkpoint to {save_path}")
    return agent


# ────────────────────────────────────────────────────────────────────────────
# DORAEMON training inner-loop
# ────────────────────────────────────────────────────────────────────────────

def train_under_doraemon_beta(
    agent, mu, conc_init,
    blocks, episodes_per_block,
    bounds, max_steps,
    alpha_target, kl_max,
    n_candidates, mean_std, logc_std,
    c_min, c_max,
    min_log_w, max_log_w,
    seed, freeze_mean=True,
):
    """
    Train the agent under DORAEMON Beta-DR adaptation for the specified 
    number of blocks and episodes. Returns the final agent, history of g_hat 
    and concentration, final distribution, and block-wise summary.
    """
    rng = np.random.default_rng(seed)
    dist = make_beta_dist_at_mu(mu, conc_init, bounds)

    g_history, conc_history, block_history = [], [], []

    for b in range(int(blocks)):
        z_batch, s_batch, loss_batch = [], [], []
        for e in range(int(episodes_per_block)):
            friction_e, mass_e, z_e = sample_beta_params(dist, rng)
            env = make_manipulation_env(friction=friction_e, mass=mass_e)
            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict)
            losses = []
            solved = False
            for _ in range(max_steps):
                a = agent.act(obs, training=True)
                obs_dict, reward, terminated, truncated, _ = env.step(a)
                next_obs = flatten_obs(obs_dict)
                done = terminated or truncated
                agent.store(obs, a, reward, next_obs, float(done))
                loss = agent.learn_step()
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                if terminated:
                    solved = True
                    break
                if truncated:
                    break
            agent.end_episode()
            z_batch.append(z_e)
            s_batch.append(1.0 if solved else 0.0)
            if losses:
                loss_batch.append(float(np.mean(losses)))

        z_batch = np.asarray(z_batch, dtype=np.float64)
        s_batch = np.asarray(s_batch, dtype=np.float64)

        dist, upd = doraemon_update_beta(
            old_dist=dist, z_samples=z_batch, success_samples=s_batch,
            alpha_target=alpha_target, kl_max=kl_max, rng=rng,
            n_candidates=n_candidates, mean_std=mean_std, logc_std=logc_std,
            c_min=np.asarray(c_min, dtype=np.float64),
            c_max=np.asarray(c_max, dtype=np.float64),
            min_log_w=min_log_w, max_log_w=max_log_w,
            freeze_mean=freeze_mean,
        )

        _, conc_end = mean_conc_from_beta(dist["alpha"], dist["beta"])
        g_history.append(float(upd["g_hat"]))
        conc_history.append(conc_end.copy())

        m_end, _ = mean_conc_from_beta(dist["alpha"], dist["beta"])
        mu_end = denormalize_params(m_end, bounds)
        block_history.append({
            "block": int(b + 1),
            "mode": str(upd["mode"]),
            "emp_success": float(np.mean(s_batch)) if len(s_batch) else 0.0,
            "g_hat": float(upd["g_hat"]),
            "entropy": float(upd["entropy"]),
            "kl": float(upd["kl"]),
            "mu_friction_end": float(mu_end[0]),
            "mu_mass_end": float(mu_end[1]),
            "conc_friction_end": float(conc_end[0]),
            "conc_mass_end": float(conc_end[1]),
            "loss_mean": float(np.mean(loss_batch)) if loss_batch else float("nan"),
        })

    return agent, g_history, conc_history, dist, block_history


# ────────────────────────────────────────────────────────────────────────────
# Fixed-physics SAC inner-loop (used by BO-only)
# ────────────────────────────────────────────────────────────────────────────

def train_at_fixed_physics(
    agent, friction, mass,
    n_episodes, max_steps, seed,
):
    """Train the agent at fixed physics parameters, without any DR adaptation."""
    rng = np.random.default_rng(seed)  # noqa: F841 (kept for parity / future use)
    s_log, loss_log = [], []
    for ep in range(int(n_episodes)):
        env = make_manipulation_env(friction=float(friction), mass=float(mass))
        obs_dict, _ = env.reset()
        obs = flatten_obs(obs_dict)
        losses = []
        solved = False
        for _ in range(max_steps):
            a = agent.act(obs, training=True)
            obs_dict, reward, terminated, truncated, _ = env.step(a)
            next_obs = flatten_obs(obs_dict)
            done = terminated or truncated
            agent.store(obs, a, reward, next_obs, float(done))
            loss = agent.learn_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            if terminated:
                solved = True
                break
            if truncated:
                break
        agent.end_episode()
        s_log.append(1.0 if solved else 0.0)
        if losses:
            loss_log.append(float(np.mean(losses)))
    return agent, {
        "emp_success": float(np.mean(s_log)) if s_log else 0.0,
        "loss_mean":   float(np.mean(loss_log)) if loss_log else float("nan"),
        "n_episodes":  int(n_episodes),
    }


# ────────────────────────────────────────────────────────────────────────────
# Run-config helper + BO/DORAEMON loops
# ────────────────────────────────────────────────────────────────────────────

def project_mu_to_bounds(mu, bounds):
    """Clip the mu parameters to be within the specified bounds."""
    return np.array([
        float(np.clip(mu[0], bounds["friction"][0], bounds["friction"][1])),
        float(np.clip(mu[1], bounds["mass"][0], bounds["mass"][1])),
    ], dtype=np.float32)


def _save_run_config(run_dir: Path, **kwargs):
    config = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            config[k] = v.tolist()
        elif isinstance(v, Path):
            config[k] = str(v)
        else:
            config[k] = v
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["device"] = DEVICE
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


def _init_warmstart(checkpoint_path, device=DEVICE) -> SACAgent:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        agent = load_warmstart_agent(checkpoint_path, device=device)
        print(f"[init] warm-start loaded from {checkpoint_path}")
    else:
        agent = make_agent_from_hparams(None, device=device)
        print("[init] no checkpoint, starting from scratch")
    return agent


def run_bo_doraemon_paperlike(
    *,
    runs_root: Path,
    run_name: str,
    checkpoint_path, mu0, conc0,
    T, blocks, episodes_per_block, B_r,
    bounds, max_steps,
    alpha_target, kl_max,
    n_candidates, mean_std, logc_std,
    c_min, c_max, min_log_w, max_log_w,
    target_params, seed=42,
    freeze_doraemon_mean=True,
    conc_init_mode="reset", conc_blend=0.5,
):
    """Run the BO+DORAEMON loop with a paper-like configuration and logging."""

    runs_root = Path(runs_root)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(
        run_dir,
        method="bodrem",
        checkpoint_path=checkpoint_path, mu0=np.asarray(mu0), conc0=np.asarray(conc0),
        T=T, blocks=blocks, episodes_per_block=episodes_per_block, B_r=B_r,
        bounds=bounds, max_steps=max_steps,
        alpha_target=alpha_target, kl_max=kl_max,
        n_candidates=n_candidates, mean_std=mean_std, logc_std=logc_std,
        c_min=np.asarray(c_min), c_max=np.asarray(c_max),
        min_log_w=min_log_w, max_log_w=max_log_w,
        target_params=target_params, seed=seed,
        freeze_doraemon_mean=freeze_doraemon_mean,
        conc_init_mode=conc_init_mode, conc_blend=conc_blend,
    )

    hist_csv   = run_dir / "history.csv"
    hist_json  = run_dir / "history.json"
    block_json = run_dir / "block_history.json"
    best_ckpt  = run_dir / "best_policy.pt"
    last_ckpt  = run_dir / "last_policy.pt"

    agent_prev = _init_warmstart(checkpoint_path)
    conc_prev = np.asarray(conc0, dtype=np.float32).copy()

    opt = Optimizer(
        dimensions=[
            Real(bounds["friction"][0], bounds["friction"][1], name="friction_center"),
            Real(bounds["mass"][0],     bounds["mass"][1],     name="mass_center"),
        ],
        base_estimator="GP", acq_func="EI", random_state=seed,
    )

    history, block_rows = [], []

    mu0      = np.array(mu0, dtype=np.float32)
    mu0_proj = project_mu_to_bounds(mu0, bounds)

    f0, succ0, eps0 = target_eval_manipulation(agent_prev, B_r, max_steps, target_params, seed + 7)
    opt.tell(mu0_proj.tolist(), -f0)

    best_score   = float(f0)
    best_center  = mu0_proj.copy()
    save_agent_checkpoint(agent_prev, best_ckpt,
                          extra={"center": mu0_proj.tolist(), "score": float(f0), "iter": 0})

    history.append({
        "iter": 0,
        "mu_friction": float(mu0_proj[0]), "mu_mass": float(mu0_proj[1]),
        "conc_friction_start": float(conc_prev[0]), "conc_mass_start": float(conc_prev[1]),
        "conc_friction_end":   float(conc_prev[0]), "conc_mass_end":   float(conc_prev[1]),
        "g_mean": float("nan"), "g_last": float("nan"),
        "entropy_last": float("nan"), "kl_last": float("nan"),
        "target_solve_rate": float(f0),
        "target_successes": int(succ0), "target_episodes": int(eps0),
        "is_best": True,
    })
    print(f"[BODREM init] target={f0:.3f} ({succ0}/{eps0})")

    for t in range(1, T + 1):
        mu_t = np.array(opt.ask(), dtype=np.float64)
        agent_t = clone_agent(agent_prev)

        if conc_init_mode == "carry":
            conc_start = conc_prev.copy()
        elif conc_init_mode == "blend":
            w = float(np.clip(conc_blend, 0.0, 1.0))
            conc_start = (1.0 - w) * np.asarray(conc0, dtype=np.float32) + w * conc_prev
        else:
            conc_start = np.asarray(conc0, dtype=np.float32).copy()

        agent_t, g_hist, conc_hist, dist_end, block_hist = train_under_doraemon_beta(
            agent=agent_t, mu=mu_t, conc_init=conc_start,
            blocks=blocks, episodes_per_block=episodes_per_block,
            bounds=bounds, max_steps=max_steps,
            alpha_target=alpha_target, kl_max=kl_max,
            n_candidates=n_candidates, mean_std=mean_std, logc_std=logc_std,
            c_min=c_min, c_max=c_max,
            min_log_w=min_log_w, max_log_w=max_log_w,
            seed=seed + 10000 * t, freeze_mean=freeze_doraemon_mean,
        )

        for r in block_hist:
            block_rows.append({"iter": int(t), **r})

        f_t, succ_t, eps_t = target_eval_manipulation(
            agent_t, B_r, max_steps, target_params, seed + 333 + t
        )
        opt.tell(mu_t.tolist(), -float(f_t))

        conc_end = conc_hist[-1] if conc_hist else conc_start

        is_best = False
        if float(f_t) > best_score:
            best_score  = float(f_t)
            best_center = mu_t.copy()
            save_agent_checkpoint(agent_t, best_ckpt,
                                  extra={"center": mu_t.tolist(), "score": float(f_t), "iter": t})
            is_best = True

        history.append({
            "iter": t,
            "mu_friction": float(mu_t[0]), "mu_mass": float(mu_t[1]),
            "conc_friction_start": float(conc_start[0]), "conc_mass_start": float(conc_start[1]),
            "conc_friction_end":   float(conc_end[0]),   "conc_mass_end":   float(conc_end[1]),
            "g_mean": float(np.mean(g_hist)) if g_hist else float("nan"),
            "g_last": float(g_hist[-1]) if g_hist else float("nan"),
            "entropy_last": float(block_hist[-1]["entropy"]) if block_hist else float("nan"),
            "kl_last":      float(block_hist[-1]["kl"])      if block_hist else float("nan"),
            "target_solve_rate": float(f_t),
            "target_successes": int(succ_t), "target_episodes": int(eps_t),
            "is_best": bool(is_best),
        })

        _flush_history(hist_json, hist_csv, history, block_json, block_rows)

        agent_prev = agent_t
        conc_prev  = np.asarray(conc_end, dtype=np.float32).copy()

        print(
            f"[BODREM step {t:02d}/{T}] mu=({mu_t[0]:.4f}, {mu_t[1]:.4f}) "
            f"g_last={history[-1]['g_last']:.3f} target={f_t:.3f} "
            f"({succ_t}/{eps_t}) conc=({conc_end[0]:.2f}, {conc_end[1]:.2f})"
        )

    save_agent_checkpoint(agent_prev, last_ckpt, extra={"iter": T, "best_score": best_score})

    plot_path = save_progress_plot(run_dir, history, method="BODREM", alpha_target=alpha_target)

    return {
        "method": "bodrem",
        "run_dir": str(run_dir),
        "history": history,
        "block_history": block_rows,
        "best_center": best_center.tolist(),
        "best_score": float(best_score),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "progress_plot": plot_path,
    }

def run_doraemon_only(
    *,
    runs_root: Path,
    run_name: str,
    checkpoint_path, mu0, conc0,
    blocks, episodes_per_block, B_r,
    bounds, max_steps,
    alpha_target, kl_max,
    n_candidates, mean_std, logc_std,
    c_min, c_max, min_log_w, max_log_w,
    target_params, seed=42,
    freeze_doraemon_mean=True,
    eval_every_blocks: int = 1,
):
    """DORAEMON without an outer BO loop. Center is fixed at `mu0`."""
    runs_root = Path(runs_root)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(
        run_dir,
        method="doraemon_only",
        checkpoint_path=checkpoint_path, mu0=np.asarray(mu0), conc0=np.asarray(conc0),
        blocks=blocks, episodes_per_block=episodes_per_block, B_r=B_r,
        bounds=bounds, max_steps=max_steps,
        alpha_target=alpha_target, kl_max=kl_max,
        n_candidates=n_candidates, mean_std=mean_std, logc_std=logc_std,
        c_min=np.asarray(c_min), c_max=np.asarray(c_max),
        min_log_w=min_log_w, max_log_w=max_log_w,
        target_params=target_params, seed=seed,
        freeze_doraemon_mean=freeze_doraemon_mean,
        eval_every_blocks=eval_every_blocks,
    )

    hist_csv   = run_dir / "history.csv"
    hist_json  = run_dir / "history.json"
    block_json = run_dir / "block_history.json"
    best_ckpt  = run_dir / "best_policy.pt"
    last_ckpt  = run_dir / "last_policy.pt"

    agent = _init_warmstart(checkpoint_path)
    mu0      = np.array(mu0, dtype=np.float32)
    mu0_proj = project_mu_to_bounds(mu0, bounds)
    rng = np.random.default_rng(seed)
    dist = make_beta_dist_at_mu(mu0_proj, np.asarray(conc0, dtype=np.float64), bounds)

    history, block_rows = [], []

    f0, succ0, eps0 = target_eval_manipulation(agent, B_r, max_steps, target_params, seed + 7)
    best_score = float(f0)
    save_agent_checkpoint(agent, best_ckpt, extra={"score": float(f0), "block": 0})
    history.append({
        "block": 0,
        "mu_friction": float(mu0_proj[0]), "mu_mass": float(mu0_proj[1]),
        "g_hat": float("nan"), "entropy": float("nan"), "kl": float("nan"),
        "emp_success": float("nan"),
        "target_solve_rate": float(f0),
        "target_successes": int(succ0), "target_episodes": int(eps0),
        "is_best": True,
    })
    print(f"[DORA init] target={f0:.3f} ({succ0}/{eps0})")

    for b in range(1, int(blocks) + 1):
        z_batch, s_batch, loss_batch = [], [], []
        for _ in range(int(episodes_per_block)):
            friction_e, mass_e, z_e = sample_beta_params(dist, rng)
            env = make_manipulation_env(friction=friction_e, mass=mass_e)
            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict)
            losses = []
            solved = False
            for _ in range(max_steps):
                a = agent.act(obs, training=True)
                obs_dict, reward, terminated, truncated, _ = env.step(a)
                next_obs = flatten_obs(obs_dict)
                done = terminated or truncated
                agent.store(obs, a, reward, next_obs, float(done))
                loss = agent.learn_step()
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                if terminated:
                    solved = True
                    break
                if truncated:
                    break
            agent.end_episode()
            z_batch.append(z_e)
            s_batch.append(1.0 if solved else 0.0)
            if losses:
                loss_batch.append(float(np.mean(losses)))

        z_batch = np.asarray(z_batch, dtype=np.float64)
        s_batch = np.asarray(s_batch, dtype=np.float64)

        dist, upd = doraemon_update_beta(
            old_dist=dist, z_samples=z_batch, success_samples=s_batch,
            alpha_target=alpha_target, kl_max=kl_max, rng=rng,
            n_candidates=n_candidates, mean_std=mean_std, logc_std=logc_std,
            c_min=np.asarray(c_min, dtype=np.float64),
            c_max=np.asarray(c_max, dtype=np.float64),
            min_log_w=min_log_w, max_log_w=max_log_w,
            freeze_mean=freeze_doraemon_mean,
        )

        m_end, conc_end = mean_conc_from_beta(dist["alpha"], dist["beta"])
        mu_end = denormalize_params(m_end, bounds)
        block_rows.append({
            "block": int(b),
            "mode": str(upd["mode"]),
            "emp_success": float(np.mean(s_batch)) if len(s_batch) else 0.0,
            "g_hat": float(upd["g_hat"]),
            "entropy": float(upd["entropy"]),
            "kl": float(upd["kl"]),
            "mu_friction_end": float(mu_end[0]),
            "mu_mass_end":     float(mu_end[1]),
            "conc_friction_end": float(conc_end[0]),
            "conc_mass_end":     float(conc_end[1]),
            "loss_mean": float(np.mean(loss_batch)) if loss_batch else float("nan"),
        })

        do_eval = (b % max(1, int(eval_every_blocks)) == 0) or (b == blocks)
        if do_eval:
            f_b, succ_b, eps_b = target_eval_manipulation(
                agent, B_r, max_steps, target_params, seed + 333 + b
            )
            is_best = float(f_b) > best_score
            if is_best:
                best_score = float(f_b)
                save_agent_checkpoint(agent, best_ckpt, extra={"score": float(f_b), "block": b})
            history.append({
                "block": int(b),
                "mu_friction": float(mu_end[0]), "mu_mass": float(mu_end[1]),
                "conc_friction_end": float(conc_end[0]), "conc_mass_end": float(conc_end[1]),
                "g_hat": float(upd["g_hat"]),
                "entropy": float(upd["entropy"]),
                "kl": float(upd["kl"]),
                "emp_success": float(np.mean(s_batch)) if len(s_batch) else 0.0,
                "target_solve_rate": float(f_b),
                "target_successes": int(succ_b), "target_episodes": int(eps_b),
                "is_best": bool(is_best),
            })
            print(
                f"[DORA block {b:02d}/{blocks}] g_hat={upd['g_hat']:.3f} "
                f"emp_succ={np.mean(s_batch):.2f} target={f_b:.3f} ({succ_b}/{eps_b}) "
                f"conc=({conc_end[0]:.2f}, {conc_end[1]:.2f})"
            )

        _flush_history(hist_json, hist_csv, history, block_json, block_rows)

    save_agent_checkpoint(agent, last_ckpt, extra={"blocks": int(blocks), "best_score": best_score})

    plot_path = save_progress_plot(run_dir, history, method="DORAEMON-only",
                                   alpha_target=alpha_target)

    return {
        "method": "doraemon_only",
        "run_dir": str(run_dir),
        "history": history,
        "block_history": block_rows,
        "best_score": float(best_score),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "progress_plot": plot_path,
    }


# ── BO only (no DR adaptation, fixed-physics inner training) ───────────────

def run_bo_only(
    *,
    runs_root: Path,
    run_name: str,
    checkpoint_path, mu0,
    T, episodes_per_iter, B_r,
    bounds, max_steps,
    target_params, seed=42,
):
    """BO over (friction, mass) center. Each iter trains SAC at the suggested
    fixed physics — no DR adaptation, no DORAEMON."""
    runs_root = Path(runs_root)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(
        run_dir,
        method="bo_only",
        checkpoint_path=checkpoint_path, mu0=np.asarray(mu0),
        T=T, episodes_per_iter=episodes_per_iter, B_r=B_r,
        bounds=bounds, max_steps=max_steps,
        target_params=target_params, seed=seed,
    )

    hist_csv  = run_dir / "history.csv"
    hist_json = run_dir / "history.json"
    best_ckpt = run_dir / "best_policy.pt"
    last_ckpt = run_dir / "last_policy.pt"

    agent_prev = _init_warmstart(checkpoint_path)

    opt = Optimizer(
        dimensions=[
            Real(bounds["friction"][0], bounds["friction"][1], name="friction_center"),
            Real(bounds["mass"][0],     bounds["mass"][1],     name="mass_center"),
        ],
        base_estimator="GP", acq_func="EI", random_state=seed,
    )

    history = []

    mu0      = np.array(mu0, dtype=np.float32)
    mu0_proj = project_mu_to_bounds(mu0, bounds)

    f0, succ0, eps0 = target_eval_manipulation(agent_prev, B_r, max_steps, target_params, seed + 7)
    opt.tell(mu0_proj.tolist(), -f0)

    best_score  = float(f0)
    best_center = mu0_proj.copy()
    save_agent_checkpoint(agent_prev, best_ckpt,
                          extra={"center": mu0_proj.tolist(), "score": float(f0), "iter": 0})

    history.append({
        "iter": 0,
        "mu_friction": float(mu0_proj[0]), "mu_mass": float(mu0_proj[1]),
        "emp_success": float("nan"),
        "target_solve_rate": float(f0),
        "target_successes": int(succ0), "target_episodes": int(eps0),
        "is_best": True,
    })
    print(f"[BO init] target={f0:.3f} ({succ0}/{eps0})")

    for t in range(1, T + 1):
        mu_t = np.array(opt.ask(), dtype=np.float64)
        agent_t = clone_agent(agent_prev)

        agent_t, train_stats = train_at_fixed_physics(
            agent=agent_t,
            friction=float(mu_t[0]), mass=float(mu_t[1]),
            n_episodes=episodes_per_iter, max_steps=max_steps,
            seed=seed + 10000 * t,
        )

        f_t, succ_t, eps_t = target_eval_manipulation(
            agent_t, B_r, max_steps, target_params, seed + 333 + t
        )
        opt.tell(mu_t.tolist(), -float(f_t))

        is_best = float(f_t) > best_score
        if is_best:
            best_score  = float(f_t)
            best_center = mu_t.copy()
            save_agent_checkpoint(agent_t, best_ckpt,
                                  extra={"center": mu_t.tolist(), "score": float(f_t), "iter": t})

        history.append({
            "iter": t,
            "mu_friction": float(mu_t[0]), "mu_mass": float(mu_t[1]),
            "emp_success": float(train_stats["emp_success"]),
            "loss_mean":   float(train_stats["loss_mean"]),
            "target_solve_rate": float(f_t),
            "target_successes": int(succ_t), "target_episodes": int(eps_t),
            "is_best": bool(is_best),
        })

        _flush_history(hist_json, hist_csv, history, None, None)

        agent_prev = agent_t
        print(
            f"[BO step {t:02d}/{T}] mu=({mu_t[0]:.4f}, {mu_t[1]:.4f}) "
            f"emp_succ={train_stats['emp_success']:.2f} "
            f"target={f_t:.3f} ({succ_t}/{eps_t})"
        )

    save_agent_checkpoint(agent_prev, last_ckpt, extra={"iter": T, "best_score": best_score})

    plot_path = save_progress_plot(run_dir, history, method="BO-only", alpha_target=None)

    return {
        "method": "bo_only",
        "run_dir": str(run_dir),
        "history": history,
        "best_center": best_center.tolist(),
        "best_score": float(best_score),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "progress_plot": plot_path,
    }


# ────────────────────────────────────────────────────────────────────────────
# History I/O helper
# AI Generated !
# ────────────────────────────────────────────────────────────────────────────

def save_progress_plot(
    run_dir: Path,
    history: list[dict],
    method: str,
    alpha_target: float | None = None,
    filename: str | None = None,
) -> str | None:
    """Save the 4-panel progress plot used by the original notebook.

    Adapts to all three methods. The x-axis is `iter` for BODREM/BO-only and
    `block` for DORAEMON-only. Panels missing data (e.g. `g_last` for BO-only)
    are skipped gracefully.
    """
    if not history:
        return None
    run_dir = Path(run_dir)

    # Generate filename from method if not provided
    if filename is None:
        method_name = method.lower().replace("-", "_")
        filename = f"{method_name}_progress.pdf"

    x_key = "iter" if "iter" in history[0] else "block"
    xs    = [r[x_key]            for r in history]
    fvals = [r.get("target_solve_rate", float("nan")) for r in history]
    mu_f  = [r.get("mu_friction", float("nan"))       for r in history]
    mu_m  = [r.get("mu_mass",     float("nan"))       for r in history]

    # DORAEMON metrics may live under different keys depending on the runner
    glast = [r.get("g_last", r.get("g_hat", float("nan"))) for r in history]
    cf    = [r.get("conc_friction_end", float("nan"))      for r in history]
    cm    = [r.get("conc_mass_end",     float("nan"))      for r in history]

    has_g    = any(not (isinstance(v, float) and np.isnan(v)) for v in glast)
    has_conc = any(not (isinstance(v, float) and np.isnan(v)) for v in cf)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(xs, fvals, marker="o")
    axes[0].set_title(f"Target solve-rate vs {x_key}")
    axes[0].set_xlabel(x_key)
    axes[0].set_ylabel("Target solve-rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)

    if has_g:
        axes[1].plot(xs, glast, marker="o", label="g (IS estimate)")
        if alpha_target is not None:
            axes[1].axhline(alpha_target, color="r", linestyle="--", label="alpha_target")
        axes[1].set_title("DORAEMON Success Constraint")
        axes[1].set_xlabel(x_key)
        axes[1].set_ylabel("Success estimate")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].set_title("DORAEMON metric N/A")

    if has_conc:
        axes[2].plot(xs, cf, marker="o", label="conc_friction")
        axes[2].plot(xs, cm, marker="o", label="conc_mass")
        axes[2].set_title("Beta concentration")
        axes[2].set_xlabel(x_key)
        axes[2].set_ylabel("Concentration")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    else:
        axes[2].axis("off")
        axes[2].set_title("Beta concentration N/A")

    sc = axes[3].scatter(mu_f, mu_m, c=xs, cmap="viridis", s=60)
    axes[3].plot(mu_f, mu_m, alpha=0.5)
    axes[3].set_title("Centre trajectory in (friction, mass)")
    axes[3].set_xlabel("friction centre")
    axes[3].set_ylabel("mass centre")
    axes[3].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[3], label=x_key)

    fig.suptitle(f"{method}  ({run_dir.name})", fontsize=12)
    plt.tight_layout()
    out_path = run_dir / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return str(out_path)


def _flush_history(hist_json, hist_csv, history, block_json, block_rows):
    with open(hist_json, "w") as f:
        json.dump(history, f, indent=2)
    if history:
        fieldnames = list(dict.fromkeys(k for r in history for k in r.keys()))
        with open(hist_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
    if block_json is not None and block_rows is not None:
        with open(block_json, "w") as f:
            json.dump(block_rows, f, indent=2)
