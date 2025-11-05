"""Multi-boat sectors environment implementation."""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .boat import Boat
from .config import BoatParams, EnvConfig, SpawnConfig, TurnSessionConfig
from .utils import angle_deg, clamp, tcpa_dcpa, wrap_pi

try:  # pragma: no cover - optional dependency
    import pygame

    HAS_PYGAME = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PYGAME = False


class MultiBoatSectorsEnv:
    """Shared-policy, multi-boat sectors environment."""

    def __init__(
        self,
        cfg: EnvConfig = EnvConfig(),
        kin: BoatParams = BoatParams(),
        tcfg: TurnSessionConfig = TurnSessionConfig(),
        spawn: SpawnConfig = SpawnConfig(),
    ) -> None:
        self.cfg = cfg
        self.kin = kin
        self.tcfg = tcfg
        self.spawn = spawn
        self.rng = random.Random(cfg.seed)

        self.world_w = float(cfg.world_w)
        self.world_h = float(cfg.world_h)

        self.ships: List[Boat] = []
        self.goals: List[Tuple[float, float]] = []
        self.prev_goal_d: List[float] = []
        self.reached: List[bool] = []
        self.time = 0.0
        self.step_index = 0

        self._screen = None
        self._font = None
        self._clock = None
        self.ppm = float(cfg.pixels_per_meter)
        self.hud_on = bool(cfg.show_hud)
        self._overlay_extra: Dict[str, float] = {}
        self._traces: List[deque] = []
        self._last_info: Dict = {}

        if self.cfg.render and HAS_PYGAME:
            self._setup_render()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _setup_render(self) -> None:
        if not HAS_PYGAME:
            return
        pygame.init()
        width_px = max(200, int(round(self.world_w * self.ppm)))
        height_px = max(200, int(round(self.world_h * self.ppm)))
        self._screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Multi-Boat v3 — Chunked Sessions + CPA")
        self._font = pygame.font.Font(None, 18)
        self._clock = pygame.time.Clock()

    def enable_render(self) -> None:
        if not self._screen and HAS_PYGAME:
            self.cfg.render = True
            self._setup_render()

    def set_overlay(self, values: Optional[Dict]) -> None:
        self._overlay_extra = values or {}

    def sx(self, x_m: float) -> int:
        return int(round(x_m * self.ppm))

    def sy(self, y_m: float) -> int:
        return int(round(y_m * self.ppm))

    def _handle_events(self) -> None:
        if not HAS_PYGAME:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.hud_on = not self.hud_on
                elif event.key == pygame.K_g:
                    self.cfg.show_grid = not self.cfg.show_grid
                elif event.key == pygame.K_s:
                    self.cfg.show_sectors = not self.cfg.show_sectors
                elif event.key == pygame.K_t:
                    self.cfg.show_trails = not self.cfg.show_trails
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    @property
    def n_boats(self) -> int:
        return len(self.ships) if self.ships else int(self.spawn.n_boats)

    def set_n_boats(self, n: int) -> List[np.ndarray]:
        self.spawn.n_boats = int(n)
        return self.reset()

    # Rendering ---------------------------------------------------------
    def _draw_ship(self, surf, sh: Boat) -> None:
        Lm, Wm = self.kin.length, self.kin.width
        verts_local_m = [(+0.5 * Lm, 0.0), (-0.5 * Lm, -0.5 * Wm), (-0.5 * Lm, +0.5 * Wm)]
        ch, shn = math.cos(sh.h), math.sin(sh.h)
        pts = []
        for vx_m, vy_m in verts_local_m:
            wx_m = sh.x + vx_m * ch - vy_m * shn
            wy_m = sh.y + vx_m * shn + vy_m * ch
            pts.append((self.sx(wx_m), self.sy(wy_m)))
        color = (90, 160, 255) if sh.id == 0 else (70, 200, 120)
        pygame.draw.polygon(surf, color, pts)
        rad_px = max(1, int(round(self.collision_radius() * self.ppm)))
        pygame.draw.circle(surf, (255, 255, 255), (self.sx(sh.x), self.sy(sh.y)), rad_px, 1)
        label = self._font.render(f"{sh.id}", True, (255, 255, 255))
        surf.blit(label, (self.sx(sh.x) + 8, self.sy(sh.y) - 8))

    def _draw_sector_rays(self, surf, sh: Boat, n: int = 12, ray_len_m: Optional[float] = None) -> None:
        Lm = ray_len_m or 320.0
        for k in range(n):
            ang = sh.h + 2.0 * math.pi * k / n
            x2 = self.sx(sh.x + Lm * math.cos(ang))
            y2 = self.sy(sh.y + Lm * math.sin(ang))
            pygame.draw.line(surf, (220, 220, 220), (self.sx(sh.x), self.sy(sh.y)), (x2, y2), 1)

    def _draw_hud(self, surf) -> None:
        if not self.hud_on:
            return
        font = pygame.font.Font(None, 18)
        pad, line = 8, 20
        fps = self._clock.get_fps() if self._clock else 0.0

        def fmt_reward(value: float) -> str:
            return f"{value:+.3f}" if isinstance(value, (int, float)) else "-"

        lines = [f"FPS {fps:5.1f}   step {self.step_index}   t {self.time:6.2f}s"]
        if self._overlay_extra:
            ep = self._overlay_extra.get("episode")
            eps = self._overlay_extra.get("epsilon")
            gsteps = self._overlay_extra.get("global_steps")
            loss_ema = self._overlay_extra.get("loss_ema")
            if ep is not None:
                if eps is not None:
                    lines.append(f"ep {ep}   ε {eps:.3f}")
                else:
                    lines.append(f"ep {ep}")
            if gsteps is not None:
                lines.append(f"gsteps {gsteps}")
            if isinstance(loss_ema, (int, float)):
                lines.append(f"loss_ema {loss_ema:.5f}")

        if self.ships:
            if self._last_info:
                rewards = self._last_info.get("rewards")
            else:
                rewards = None
            for i, sh in enumerate(self.ships):
                lines.append(
                    f"B{i}: spd {sh.u:4.1f} m/s  hdg {math.degrees(sh.h):6.1f}°  "
                    f"turn={'ON' if sh.session_active else 'OFF'}"
                )
                gx, gy = self.goals[i]
                gd = math.hypot(gx - sh.x, gy - sh.y)
                reward = rewards[i] if isinstance(rewards, (list, tuple)) and i < len(rewards) else None
                lines.append(
                    f"  goal_d {gd:6.1f} m   reached={self.reached[i]}   r={fmt_reward(reward)}"
                )

        reason = self._last_info.get("reason", "")
        if reason:
            lines.append(f"reason: {reason}")

        w = max(font.size(line)[0] for line in lines) + 2 * pad
        h = len(lines) * line + 2 * pad
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        y = pad
        for line in lines:
            img = font.render(line, True, (240, 240, 240))
            panel.blit(img, (pad, y))
            y += line
        surf.blit(panel, (10, 10))

    def render(self) -> None:
        if not self._screen:
            return
        self._handle_events()
        surf = self._screen
        surf.fill((18, 52, 86))

        pygame.draw.rect(
            surf,
            (180, 180, 180),
            (0, 0, self.sx(self.world_w), self.sy(self.world_h)),
            2,
        )

        if self.cfg.show_grid:
            step = 20.0
            for gx in np.arange(0.0, self.world_w + 1e-6, step):
                pygame.draw.line(surf, (35, 65, 92), (self.sx(gx), 0), (self.sx(gx), self.sy(self.world_h)))
            for gy in np.arange(0.0, self.world_h + 1e-6, step):
                pygame.draw.line(surf, (35, 65, 92), (0, self.sy(gy)), (self.sx(self.world_w), self.sy(gy)))

        for idx, sh in enumerate(self.ships):
            color = (255, 200, 80) if self.reached[idx] else (160, 70, 200)
            pygame.draw.circle(surf, color, (self.sx(self.goals[idx][0]), self.sy(self.goals[idx][1])), 6)

        if self.cfg.show_trails:
            for trace in self._traces:
                if len(trace) < 2:
                    continue
                pygame.draw.lines(surf, (220, 220, 220), False, trace, 1)

        for trace, sh in zip(self._traces, self.ships):
            trace.append((self.sx(sh.x), self.sy(sh.y)))

        for sh in self.ships:
            self._draw_ship(surf, sh)
            if self.cfg.show_sectors:
                self._draw_sector_rays(surf, sh)

        self._draw_hud(surf)
        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)

    # ------------------------------------------------------------------
    # Core mechanics
    # ------------------------------------------------------------------
    def collision_radius(self) -> float:
        return 0.5 * math.hypot(self.kin.length, self.kin.width)

    def _outside(self, sh: Boat) -> bool:
        return not (0.0 <= sh.x <= self.world_w and 0.0 <= sh.y <= self.world_h)

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        if seed is not None:
            self.rng.seed(seed)

        self.ships.clear()
        self.goals.clear()
        self.prev_goal_d.clear()
        self.reached.clear()
        self.time = 0.0
        self.step_index = 0
        self._traces = []

        margin = self.spawn.margin
        attempts = int(self.spawn.max_spawn_attempts)
        for i in range(int(self.spawn.n_boats)):
            placed = False
            for _ in range(attempts):
                x = self.rng.uniform(margin, self.world_w - margin)
                y = self.rng.uniform(margin, self.world_h - margin)
                h = self.rng.uniform(-math.pi, math.pi)

                ok = True
                for other in self.ships:
                    dx = other.x - x
                    dy = other.y - y
                    if dx * dx + dy * dy < (self.spawn.min_sep_factor * self.collision_radius()) ** 2:
                        ok = False
                        break

                if not ok:
                    continue

                boat = Boat(i, x, y, h, self.spawn.start_speed, self.kin, self.tcfg)
                self.ships.append(boat)

                gx = x + self.spawn.goal_ahead_distance * math.cos(h)
                gy = y + self.spawn.goal_ahead_distance * math.sin(h)
                clearance = self.spawn.goal_edge_clearance
                gx = clamp(gx, clearance, self.world_w - clearance)
                gy = clamp(gy, clearance, self.world_h - clearance)

                self.goals.append((gx, gy))
                self.prev_goal_d.append(math.hypot(gx - x, gy - y))
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to place boat after max attempts; consider relaxing constraints or enlarging the world."
                )

            self.reached.append(False)

        self._traces = [deque(maxlen=600) for _ in self.ships]
        return self.get_obs_all()

    def get_obs(self, i: int) -> np.ndarray:
        ego = self.ships[i]
        others = [s for s in self.ships if s is not ego]

        n_sectors = 12
        buckets: Dict[int, List[Tuple[float, dict]]] = {k: [] for k in range(n_sectors)}
        diag = math.hypot(self.world_w, self.world_h)
        rng = self.cfg.sensor_range or diag

        for tgt in others:
            dx, dy = tgt.x - ego.x, tgt.y - ego.y
            dist = math.hypot(dx, dy)
            if self.cfg.sensor_range and dist > self.cfg.sensor_range:
                continue

            ch, shn = math.cos(ego.h), math.sin(ego.h)
            x_rel = ch * dx + shn * dy
            y_rel = -shn * dx + ch * dy
            rel_brg = math.atan2(y_rel, x_rel)

            rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
            sector = int(rel_deg // (360.0 / n_sectors))

            tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
            w_tcpa = math.exp(-tcpa / max(1e-6, self.cfg.tcpa_decay))
            w_dcpa = math.exp(-dcpa / max(1e-6, self.cfg.dcpa_scale))
            score = -(w_tcpa * w_dcpa)

            buckets[sector].append(
                (
                    score,
                    dict(
                        x_rel=x_rel,
                        y_rel=y_rel,
                        rel_brg=rel_brg,
                        dist=dist,
                        tcpa=tcpa,
                        dcpa=dcpa,
                        tgt_speed=tgt.u,
                        tgt_rel_h=wrap_pi(tgt.h - ego.h),
                    ),
                )
            )

        feats: List[float] = []
        for k in range(n_sectors):
            if buckets[k]:
                buckets[k].sort(key=lambda t: t[0])
                data = buckets[k][0][1]
                feats.extend(
                    [
                        data["x_rel"] / rng,
                        data["y_rel"] / rng,
                        data["rel_brg"] / math.pi,
                        data["dist"] / rng,
                        clamp(data["tcpa"], 0.0, self.cfg.cpa_horizon) / self.cfg.cpa_horizon,
                        clamp(data["dcpa"], 0.0, rng) / rng,
                        data["tgt_speed"] / self.kin.max_speed,
                        data["tgt_rel_h"] / math.pi,
                    ]
                )
            else:
                feats.extend([0.0] * 8)

        feats.extend(
            [
                ego.x / self.world_w,
                ego.y / self.world_h,
                ego.u / self.kin.max_speed,
                ego.h / math.pi,
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def get_obs_all(self) -> List[np.ndarray]:
        return [self.get_obs(i) for i in range(len(self.ships))]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, object]]:
        if isinstance(actions, int):
            actions = [actions]
        if len(actions) != len(self.ships):
            raise AssertionError("actions length must equal number of boats")

        self.step_index += 1

        d_prev = [self.prev_goal_d[i] for i in range(len(self.ships))]

        for sh, act in zip(self.ships, actions):
            sh.apply_action(act)

        dt_sub = self.cfg.dt / max(1, self.cfg.substeps)
        for _ in range(self.cfg.substeps):
            for sh in self.ships:
                sh.integrate(dt_sub)

        self.time += self.cfg.dt

        done = False
        reason = ""

        radius = self.collision_radius()
        r2 = (2.0 * radius) ** 2
        for i in range(len(self.ships)):
            for j in range(i + 1, len(self.ships)):
                dx = self.ships[i].x - self.ships[j].x
                dy = self.ships[i].y - self.ships[j].y
                if dx * dx + dy * dy <= r2:
                    done, reason = True, "collision"
                    break
            if done:
                break

        if not done:
            for sh in self.ships:
                if self._outside(sh):
                    done, reason = True, "out_of_bounds"
                    break

        d_now: List[float] = []
        just_arrived = [False] * len(self.ships)
        for i, sh in enumerate(self.ships):
            gx, gy = self.goals[i]
            dist = math.hypot(gx - sh.x, gy - sh.y)
            d_now.append(dist)
            if not self.reached[i] and dist <= self.spawn.goal_radius:
                self.reached[i] = True
                if d_prev[i] > self.spawn.goal_radius:
                    just_arrived[i] = True

        if not done and all(self.reached):
            done, reason = True, "goals_reached"

        if not done and self.step_index >= self.cfg.max_steps:
            done, reason = True, "max_steps"

        rewards: List[float] = []
        for i in range(len(self.ships)):
            reward = 0.0
            reward += self.cfg.progress_weight * (d_prev[i] - d_now[i])
            reward += self.cfg.living_penalty

            if just_arrived[i]:
                reward += self.cfg.goal_bonus

            if self.cfg.risk_weight > 0.0:
                ego = self.ships[i]
                w_max = 0.0

                for j, tgt in enumerate(self.ships):
                    if j == i:
                        continue
                    tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
                    if 0.0 <= tcpa <= self.cfg.cpa_horizon:
                        w = math.exp(-tcpa / self.cfg.tcpa_decay) * math.exp(-dcpa / self.cfg.dcpa_scale)
                        w_max = max(w_max, w)

                        if getattr(ego, "session_active", False) and ego.session_dir > 0:
                            rel_brg = math.atan2(
                                -(math.sin(ego.h) * (tgt.x - ego.x) - math.cos(ego.h) * (tgt.y - ego.y)),
                                (math.cos(ego.h) * (tgt.x - ego.x) + math.sin(ego.h) * (tgt.y - ego.y)),
                            )
                            rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
                            if 0.0 <= rel_deg <= 112.5:
                                reward -= self.cfg.colregs_penalty * w

                reward -= self.cfg.risk_weight * w_max

            if reason == "collision":
                reward += self.cfg.collision_penalty
            elif reason == "out_of_bounds":
                reward += self.cfg.oob_penalty
            elif reason == "max_steps":
                reward += self.cfg.max_steps_penalty

            rewards.append(float(reward))

        for i in range(len(self.ships)):
            self.prev_goal_d[i] = d_now[i]

        info = dict(rewards=tuple(rewards), reason=reason, reached=tuple(self.reached))
        self._last_info = info
        return self.get_obs_all(), rewards, done, info
