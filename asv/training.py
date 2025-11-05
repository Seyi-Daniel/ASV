"""Training loop orchestration."""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import replace
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

from .agent import ActResult, Agent
from .env import MultiBoatSectorsEnv
from .hyperparams import GlobalHyperParameters
from .replay import Replay


def decode_action_idx(action: int) -> Tuple[int, int]:
    """Human readable decode of an action index."""
    return action // 3, action % 3


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train(params: GlobalHyperParameters) -> str:
    """Run Double-DQN training and return the output directory."""
    params.synchronize()
    hp = params.training

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = replace(params.env)
    boat_cfg = replace(params.boat)
    turn_cfg = replace(params.turn)
    spawn_cfg = replace(params.spawn)
    env = MultiBoatSectorsEnv(env_cfg, boat_cfg, turn_cfg, spawn_cfg)

    replay = Replay(hp.replay_size)
    agent = Agent(in_dim=100, n_actions=9, hp=hp, device=device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", timestamp)
    _ensure_dir(out_dir)

    writer = None
    log_file = None
    if hp.log_actions:
        log_path = os.path.join(out_dir, hp.actions_filename)
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "episode",
                "step",
                "boat_id",
                "action",
                "steer",
                "throttle",
                "was_random",
                "epsilon",
                "reward",
                "ship_x",
                "ship_y",
                "ship_speed",
                "ship_heading_deg",
                "goal_x",
                "goal_y",
                "goal_dist",
                "reached",
                "reason",
            ]
        )

    returns: List[float] = []
    ema_loss: float | None = None
    ema_ret: float | None = None
    best_metric = float("-inf")

    for episode in range(1, hp.episodes + 1):
        observations = env.reset(seed=hp.seed + episode)
        episode_return = 0.0
        reason = ""
        steps = 0

        for step_idx in range(hp.steps_per_episode):
            if hp.render:
                env.set_overlay(
                    {
                        "episode": episode,
                        "epsilon": agent.epsilon(),
                        "global_steps": agent.global_step,
                        "loss_ema": ema_loss,
                    }
                )
                env.render()

            act_results: List[ActResult] = []
            actions: List[int] = []
            for obs in observations:
                result = agent.act(obs)
                act_results.append(result)
                actions.append(result.action)

            next_obs, rewards, done, info = env.step(actions)
            reason = info.get("reason", "")

            for idx in range(len(env.ships)):
                agent.global_step += 1
                replay.push(observations[idx], actions[idx], rewards[idx], next_obs[idx], float(done))
                if len(replay) >= hp.min_replay:
                    loss = agent.update(replay, hp.batch_size)
                    if loss is not None:
                        ema_loss = loss if ema_loss is None else (0.99 * ema_loss + 0.01 * loss)

            episode_return += float(sum(rewards))
            steps += 1

            if agent.global_step % hp.target_update == 0:
                agent.hard_update()

            if writer is not None:
                for idx in range(len(env.ships)):
                    ship = env.ships[idx]
                    goal_x, goal_y = env.goals[idx]
                    distance = math.hypot(goal_x - ship.x, goal_y - ship.y)
                    steer, throttle = decode_action_idx(actions[idx])
                    act_result = act_results[idx]
                    writer.writerow(
                        [
                            episode,
                            steps,
                            idx,
                            actions[idx],
                            steer,
                            throttle,
                            bool(act_result.was_random),
                            float(act_result.epsilon),
                            float(rewards[idx]),
                            float(ship.x),
                            float(ship.y),
                            float(ship.u),
                            float(ship.h * 180.0 / math.pi),
                            float(goal_x),
                            float(goal_y),
                            float(distance),
                            bool(env.reached[idx]),
                            reason,
                        ]
                    )
                    if hp.print_action_log:
                        print(
                            f"EP{episode} STEP{steps} B{idx} A={actions[idx]} "
                            f"rnd={act_result.was_random} eps={act_result.epsilon:.3f} r={rewards[idx]:+.3f}"
                        )

            observations = next_obs

            if done:
                if hp.render:
                    env.render()
                break

            if log_file is not None and (steps % 200 == 0):
                log_file.flush()

        returns.append(episode_return)
        if ema_ret is None:
            ema_ret = episode_return
        else:
            ema_ret = hp.ema_beta * ema_ret + (1.0 - hp.ema_beta) * episode_return

        loss_str = f"{ema_loss:.5f}" if isinstance(ema_loss, (int, float)) else "-"
        print(
            f"Ep {episode:04d} | steps {steps:4d} | return {episode_return:8.3f} | "
            f"ema_ret {ema_ret:8.3f} | eps {agent.epsilon():.3f} | replay {len(replay):6d} | "
            f"loss_ema {loss_str} | reason={reason}"
        )

        if episode % hp.save_every == 0 or episode == hp.episodes:
            torch.save(agent.q.state_dict(), os.path.join(out_dir, f"q_ep{episode}.pth"))
            np.save(os.path.join(out_dir, "returns.npy"), np.array(returns, dtype=np.float32))
            if log_file is not None:
                log_file.flush()

        agent.decay_epsilon_episode()

        if hp.save_best and ema_ret > best_metric:
            best_metric = ema_ret
            torch.save(agent.q.state_dict(), os.path.join(out_dir, hp.best_filename))

    if log_file is not None:
        log_file.close()
    print("Training finished. Results in:", out_dir)
    if hp.log_actions:
        print("Action log:", os.path.join(out_dir, hp.actions_filename))
    return out_dir
