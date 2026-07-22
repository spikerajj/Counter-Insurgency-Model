from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

# Data models

@dataclass
class Civilian:
    id: int
    x: int
    y: int
    anger: float
    fear: float
    violence_threshold: float
    state: str = "neutral"  # neutral, discontent, latent, active
    active: bool = False
    exposed: bool = False

    def clamp(self) -> None:
        self.anger = float(np.clip(self.anger, 0.0, 1.0))
        self.fear = float(np.clip(self.fear, 0.0, 1.0))
        self.violence_threshold = float(np.clip(self.violence_threshold, 0.0, 1.0))


@dataclass
class Soldier:
    id: int
    x: int
    y: int
    attacked_this_tick: bool = False


# Core model

class CoinModel:
    """
    Counterinsurgency ABM inspired by Pechenkina & Bennett (2017).

    Defaults:
    - 50x50 flat grid (non-torus)
    - 500 civilians
    - 100 soldiers
    - Moore neighborhood radius 3
    - No movement
    - One agent per cell
    """

    STATE_COLORS = {
        "neutral": "#2ca02c",     # green
        "discontent": "#ffd92f",  # yellow
        "latent": "#ff7f0e",      # orange
        "active": "#d62728",      # red
    }

    def __init__(
        self,
        grid_width: int = 50,
        grid_height: int = 50,
        grid_size: Optional[int] = None,
        num_civilians: int = 500,
        num_soldiers: int = 100,
        interaction_range: int = 3,
        effectiveness: float = 0.5,
        accuracy: float = 0.5,
        p_gr: float = 0.5,
        p_ir: float = 0.5,
        p_iewr: float = 0.5,
        soldier_recruit_n: int = 5,
        soldier_anger_delta: float = -0.15,
        insurgent_recruit_n: int = 5,
        insurgent_anger_delta: float = 0.15,
        max_ticks: int = 5000,
        seed: Optional[int] = 1,
        latent_rule: str = "formal",  # formal = anger>fear and anger>threshold
        parallel_actions: bool = True,
    ) -> None:
        if grid_size is not None:
            grid_width = grid_size
            grid_height = grid_size
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.num_civilians_initial = int(num_civilians)
        self.num_soldiers = int(num_soldiers)
        self.interaction_range = int(interaction_range)

        self.effectiveness = float(effectiveness)
        self.accuracy = float(accuracy)
        self.p_gr = float(p_gr)
        self.p_ir = float(p_ir)
        self.p_iewr = float(p_iewr)

        self.soldier_recruit_n = int(soldier_recruit_n)
        self.soldier_anger_delta = float(soldier_anger_delta)
        self.insurgent_recruit_n = int(insurgent_recruit_n)
        self.insurgent_anger_delta = float(insurgent_anger_delta)

        self.max_ticks = int(max_ticks)
        self.latent_rule = latent_rule
        self.parallel_actions = bool(parallel_actions)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.civilians: Dict[int, Civilian] = {}
        self.soldiers: Dict[int, Soldier] = {}
        self.occupancy: Dict[Tuple[int, int], Tuple[str, int]] = {}

        self._next_civilian_id = 0
        self._next_soldier_id = 0

        self.tick_count = 0
        self.cumulative_insurgents_killed = 0
        self.recent_attacks = 0
        self.termination_reason: Optional[str] = None

        self.history = {
            "tick": [],
            "total_civilians": [],
            "active_insurgents": [],
            "latent_insurgents": [],
            "cumulative_killed": [],
            "recent_attacks": [],
            "avg_anger": [],
            "avg_fear": [],
        }

        self._initialize_population()
        self.recompute_states()
        self._record_history()

  
    # Initialization
 

    def _sample_truncated_normal(self, mean: float, sd: float, low: float = 0.0, high: float = 1.0) -> float:
        while True:
            value = float(self.np_rng.normal(mean, sd))
            if low <= value <= high:
                return value

    def _random_empty_cell(self) -> Tuple[int, int]:
        if len(self.occupancy) >= self.grid_width * self.grid_height:
            raise RuntimeError("No empty cells left on the grid.")
        while True:
            x = self.rng.randrange(self.grid_width)
            y = self.rng.randrange(self.grid_height)
            if (x, y) not in self.occupancy:
                return x, y

    def _place_civilian(self, civilian: Civilian) -> None:
        self.civilians[civilian.id] = civilian
        self.occupancy[(civilian.x, civilian.y)] = ("civilian", civilian.id)

    def _place_soldier(self, soldier: Soldier) -> None:
        self.soldiers[soldier.id] = soldier
        self.occupancy[(soldier.x, soldier.y)] = ("soldier", soldier.id)

    def _initialize_population(self) -> None:
        if self.num_soldiers == 100 and self.grid_width == 50 and self.grid_height == 50:
            for i in range(10):
                for j in range(10):
                    x = i * 5 + 2
                    y = j * 5 + 2
                    soldier = Soldier(id=self._next_soldier_id, x=x, y=y)
                    self._next_soldier_id += 1
                    self._place_soldier(soldier)
        else:
            for _ in range(self.num_soldiers):
                x, y = self._random_empty_cell()
                soldier = Soldier(id=self._next_soldier_id, x=x, y=y)
                self._next_soldier_id += 1
                self._place_soldier(soldier)

        for _ in range(self.num_civilians_initial):
            x, y = self._random_empty_cell()
            civ = Civilian(
                id=self._next_civilian_id,
                x=x,
                y=y,
                anger=self._sample_truncated_normal(0.25, 0.125),
                fear=self._sample_truncated_normal(0.50, 0.25),
                violence_threshold=self._sample_truncated_normal(0.50, 0.25),
            )
            self._next_civilian_id += 1
            self._place_civilian(civ)

    # Spatial helpers

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def moore_cells(self, x: int, y: int, radius: Optional[int] = None) -> List[Tuple[int, int]]:
        r = self.interaction_range if radius is None else int(radius)
        cells: List[Tuple[int, int]] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    cells.append((nx, ny))
        return cells

    def civilians_in_range(
        self,
        x: int,
        y: int,
        radius: Optional[int] = None,
        exclude_id: Optional[int] = None,
    ) -> List[Civilian]:
        cells = set(self.moore_cells(x, y, radius))
        out: List[Civilian] = []
        for civ in self.civilians.values():
            if exclude_id is not None and civ.id == exclude_id:
                continue
            if (civ.x, civ.y) in cells:
                out.append(civ)
        return out

    def soldiers_in_range(self, x: int, y: int, radius: Optional[int] = None) -> List[Soldier]:
        cells = set(self.moore_cells(x, y, radius))
        return [s for s in self.soldiers.values() if (s.x, s.y) in cells]

    # State logic

    def _is_latent(self, civ: Civilian) -> bool:
        # Recommended formal interpretation from the model definition.
        return civ.anger > civ.fear and civ.anger > civ.violence_threshold

    def _update_civilian_state(self, civ: Civilian) -> None:
        """
        State ordering used for display and termination:
        - active: latent insurgent that is active
        - latent: willing to engage in violence
        - discontent: anger > fear
        - neutral: everything else
        """
        is_latent = self._is_latent(civ)
        if not is_latent:
            civ.active = False
            civ.exposed = False

        if civ.active:
            civ.state = "active"
        elif is_latent:
            civ.state = "latent"
        elif civ.anger > civ.fear:
            civ.state = "discontent"
        else:
            civ.state = "neutral"
        civ.clamp()

    def recompute_states(self) -> None:
        for civ in self.civilians.values():
            self._update_civilian_state(civ)

    def _latent_insurgents(self) -> List[Civilian]:
        return [c for c in self.civilians.values() if c.state == "latent"]

    def _active_insurgents(self) -> List[Civilian]:
        return [c for c in self.civilians.values() if c.state == "active"]

    def _all_insurgents(self) -> List[Civilian]:
        return [c for c in self.civilians.values() if c.state in {"latent", "active"}]

    def _choose_random_insurgent(self) -> Optional[Civilian]:
        pool = self._all_insurgents()
        return self.rng.choice(pool) if pool else None

    def _choose_random_soldier(self) -> Optional[Soldier]:
        pool = list(self.soldiers.values())
        return self.rng.choice(pool) if pool else None

    def _active_exposed_insurgents_in_range(self, x: int, y: int, radius: Optional[int] = None) -> List[Civilian]:
        cells = set(self.moore_cells(x, y, radius))
        return [
            civ for civ in self.civilians.values()
            if civ.active and civ.exposed and (civ.x, civ.y) in cells
        ]

    # -----------------------------------------------------------------
    # Recruitment / attack mechanics
    # -----------------------------------------------------------------

    def _apply_recruitment_effect(self, civ: Civilian, delta: float) -> None:
        civ.anger = civ.anger + delta * (1.0 - civ.anger)
        civ.clamp()

    def _apply_fear_increase(self, civ: Civilian) -> None:
        civ.fear = civ.fear + 0.1 * (1.0 - civ.fear)
        civ.clamp()

    def _spawn_replacement_civilian(self) -> None:
        x, y = self._random_empty_cell()
        while True:
            anger = self._sample_truncated_normal(0.25, 0.125)
            fear = self._sample_truncated_normal(0.50, 0.25)
            violence_threshold = self._sample_truncated_normal(0.50, 0.25)
            if not (anger > fear and anger > violence_threshold):
                break

        civ = Civilian(
            id=self._next_civilian_id,
            x=x,
            y=y,
            anger=anger,
            fear=fear,
            violence_threshold=violence_threshold,
        )
        self._next_civilian_id += 1
        self._place_civilian(civ)

    def _eliminate_insurgent(self, insurgent: Civilian) -> None:
        removed = self.civilians.pop(insurgent.id, None)
        if removed is None:
            return
        self.occupancy.pop((removed.x, removed.y), None)
        self.cumulative_insurgents_killed += 1
        self._spawn_replacement_civilian()

    def _resolve_counterattack(
        self,
        center_x: int,
        center_y: int,
        target_insurgent: Civilian,
        provoked: bool,
    ) -> int:
        """
        Soldier strike / counterattack around the target insurgent.
        - effectiveness: probability the insurgent is removed
        - accuracy: probability civilians are NOT injured
        - anger and fear update is applied to injured civilians
        """
        injured_candidates = self.civilians_in_range(center_x, center_y, self.interaction_range, exclude_id=target_insurgent.id)
        injured: List[Civilian] = []

        for civ in injured_candidates:
            if self.rng.random() < (1.0 - self.accuracy):
                injured.append(civ)

        injured_count = len(injured)

        if injured_count > 0:
            anger_increment = 0.05 if provoked else 0.10
            anger_boost = min(0.3, anger_increment * injured_count)
            for civ in injured:
                self._apply_fear_increase(civ)
                civ.anger = civ.anger + anger_boost * (1.0 - civ.anger)
                civ.clamp()

        if self.rng.random() < self.effectiveness:
            self._eliminate_insurgent(target_insurgent)

        return injured_count

    def _perform_insurgent_attack(self, insurgent: Civilian) -> None:
        nearby_soldiers = self.soldiers_in_range(insurgent.x, insurgent.y, self.interaction_range)
        if not nearby_soldiers:
            return

        insurgent.active = True
        insurgent.exposed = True
        self.recent_attacks += 1

        target_soldier = self.rng.choice(nearby_soldiers)
        target_soldier.attacked_this_tick = True

        # Counterattack immediately
        self._resolve_counterattack(
            center_x=insurgent.x,
            center_y=insurgent.y,
            target_insurgent=insurgent,
            provoked=True,
        )

    def _perform_insurgent_recruit(self, insurgent: Civilian) -> None:
        civilians = self.civilians_in_range(
            insurgent.x,
            insurgent.y,
            self.interaction_range,
            exclude_id=insurgent.id,
        )
        if civilians:
            k = min(self.insurgent_recruit_n, len(civilians))
            recruited = self.rng.sample(civilians, k=k)
            for civ in recruited:
                self._apply_recruitment_effect(civ, self.insurgent_anger_delta)

        # Probability of being exposed while recruiting
        if self.rng.random() < self.p_iewr:
            insurgent.active = True
            insurgent.exposed = True
            nearby_soldiers = self.soldiers_in_range(insurgent.x, insurgent.y, self.interaction_range)
            if nearby_soldiers:
                self.recent_attacks += 1
                target_soldier = self.rng.choice(nearby_soldiers)
                target_soldier.attacked_this_tick = True
                self._resolve_counterattack(
                    center_x=insurgent.x,
                    center_y=insurgent.y,
                    target_insurgent=insurgent,
                    provoked=True,
                )

    def _perform_soldier_recruit(self, soldier: Soldier) -> None:
        civilians = self.civilians_in_range(soldier.x, soldier.y, self.interaction_range)
        if civilians:
            k = min(self.soldier_recruit_n, len(civilians))
            recruited = self.rng.sample(civilians, k=k)
            for civ in recruited:
                self._apply_recruitment_effect(civ, self.soldier_anger_delta)

    def _perform_soldier_attack(self, soldier: Soldier) -> None:
        targets = self._active_exposed_insurgents_in_range(soldier.x, soldier.y, self.interaction_range)
        if not targets:
            return

        target = self.rng.choice(targets)
        soldier.attacked_this_tick = True
        self.recent_attacks += 1

        self._resolve_counterattack(
            center_x=target.x,
            center_y=target.y,
            target_insurgent=target,
            provoked=False,
        )

    def _insurgent_action(self, insurgent: Civilian) -> None:
        if self.rng.random() < self.p_ir:
            self._perform_insurgent_recruit(insurgent)
        else:
            self._perform_insurgent_attack(insurgent)

    def _soldier_action(self, soldier: Soldier) -> None:
        if self.rng.random() < self.p_gr:
            self._perform_soldier_recruit(soldier)
        else:
            self._perform_soldier_attack(soldier)

    # Tick loop / history
 

    def _reset_tick_flags(self) -> None:
        self.recent_attacks = 0
        for soldier in self.soldiers.values():
            soldier.attacked_this_tick = False

    def _record_history(self) -> None:
        avg_anger = float(np.mean([c.anger for c in self.civilians.values()])) if self.civilians else 0.0
        avg_fear = float(np.mean([c.fear for c in self.civilians.values()])) if self.civilians else 0.0

        self.history["tick"].append(self.tick_count)
        self.history["total_civilians"].append(len(self.civilians))
        self.history["active_insurgents"].append(len(self._active_insurgents()))
        self.history["latent_insurgents"].append(len(self._latent_insurgents()))
        self.history["cumulative_killed"].append(self.cumulative_insurgents_killed)
        self.history["recent_attacks"].append(self.recent_attacks)
        self.history["avg_anger"].append(avg_anger)
        self.history["avg_fear"].append(avg_fear)

    def step(self) -> bool:
        """
        One tick:
        1) recompute civilian states
        2) either one insurgent / one soldier act (legacy mode) or all eligible agents act (parallel mode)
        3) states are re-evaluated
        """
        self._reset_tick_flags()
        self.recompute_states()

        if self.parallel_actions:
            insurgents = self._all_insurgents()
            self.rng.shuffle(insurgents)
            for insurgent in insurgents:
                if insurgent.id in self.civilians and self.civilians[insurgent.id] is insurgent:
                    self._insurgent_action(insurgent)

            soldiers = list(self.soldiers.values())
            self.rng.shuffle(soldiers)
            for soldier in soldiers:
                self._soldier_action(soldier)
        else:
            insurgent = self._choose_random_insurgent()
            if insurgent is not None:
                self._insurgent_action(insurgent)

            soldier = self._choose_random_soldier()
            if soldier is not None:
                self._soldier_action(soldier)

        self.recompute_states()
        self.tick_count += 1
        self._record_history()

        if self.tick_count % 100 == 0:
            print(
                f"Tick {self.tick_count} | "
                f"Latent={len(self._latent_insurgents())} | "
                f"Active={len(self._active_insurgents())} | "
                f"Killed={self.cumulative_insurgents_killed}"
            )

        if len(self._all_insurgents()) == 0:
            self.termination_reason = "state_victory"
            return False

        if self.tick_count >= self.max_ticks:
            self.termination_reason = "sustained_insurgency"
            return False

        return True

    def run(self, max_ticks: Optional[int] = None) -> None:
        if max_ticks is not None:
            self.max_ticks = int(max_ticks)

        while self.termination_reason is None and self.tick_count < self.max_ticks:
            if not self.step():
                break

   
    # Outputs / plotting
   

    def plot_grid(self, ax: plt.Axes) -> None:
        ax.clear()
        ax.set_title("World Map")
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(-0.5, self.grid_height - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for state in ("neutral", "discontent", "latent", "active"):
            pts = [c for c in self.civilians.values() if c.state == state]
            if not pts:
                continue
            ax.scatter(
                [c.x for c in pts],
                [c.y for c in pts],
                s=18,
                c=self.STATE_COLORS[state],
                marker="o",
                edgecolors="none",
                alpha=0.95,
                label=state,
            )

        normal_soldiers = [s for s in self.soldiers.values() if not s.attacked_this_tick]
        attacked_soldiers = [s for s in self.soldiers.values() if s.attacked_this_tick]

        if normal_soldiers:
            ax.scatter(
                [s.x for s in normal_soldiers],
                [s.y for s in normal_soldiers],
                s=42,
                c="#1f77b4",
                marker="s",
                edgecolors="black",
                linewidths=0.25,
                label="soldier",
            )

        if attacked_soldiers:
            ax.scatter(
                [s.x for s in attacked_soldiers],
                [s.y for s in attacked_soldiers],
                s=58,
                c="#1f77b4",
                marker="s",
                edgecolors="red",
                linewidths=1.8,
                label="attacked soldier",
            )

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8, frameon=True)

    def plot_histories(self, ax: plt.Axes) -> None:
        ax.clear()
        ax.set_title("Actors")
        ax.set_xlabel("time")
        ax.set_ylabel("number")

        ax.plot(self.history["tick"], self.history["active_insurgents"], color="red", label="Active Insurgents")
        ax.plot(self.history["tick"], self.history["total_civilians"], color="blue", label="Civilians")
        ax.plot(self.history["tick"], self.history["latent_insurgents"], color="cyan", label="Latent Insurgents")
        ax.plot(self.history["tick"], self.history["cumulative_killed"], color="black", label="Civilian Deaths")

        # Smooth recent attacks with a 100-tick rolling sum
        attacks = self.history["recent_attacks"]
        window_size = 100
        smoothed_attacks = []
        current_sum = 0
        for i, val in enumerate(attacks):
            current_sum += val
            if i >= window_size:
                current_sum -= attacks[i - window_size]
            smoothed_attacks.append(current_sum)

        ax.plot(self.history["tick"], smoothed_attacks, color="orange", label="Recent Attacks")

        ax.set_xlim(0, max(1, self.tick_count))
        ax.legend(loc="upper right", fontsize=8)

    def plot_anger_fear(self, ax: plt.Axes) -> None:
        ax.clear()
        ax.set_title("Anger / Fear")
        ax.set_xlabel("Ticks")
        ax.set_ylabel("Average level")
        ax.set_ylim(0, 1)

        ax.plot(self.history["tick"], self.history["avg_anger"], label="Average anger")
        ax.plot(self.history["tick"], self.history["avg_fear"], label="Average fear")

        ax.set_xlim(0, max(1, self.tick_count))
        ax.legend(loc="upper left", fontsize=8)

    def summary_figure(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        self.plot_histories(axes[0])
        self.plot_anger_fear(axes[1])
        fig.suptitle(
            f"COIN simulation | {self.termination_reason or 'running'} | "
            f"ticks={self.tick_count} | killed={self.cumulative_insurgents_killed}",
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def animate(self, interval_ms: int = 60) -> FuncAnimation:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        def update(_frame: int):
            if self.termination_reason is None:
                self.step()
            self.plot_histories(axes[0])
            self.plot_anger_fear(axes[1])
            fig.suptitle(
                f"COIN simulation | {self.termination_reason or 'running'} | "
                f"ticks={self.tick_count} | killed={self.cumulative_insurgents_killed}",
                fontsize=12,
            )
            return axes

        return FuncAnimation(fig, update, frames=self.max_ticks, interval=interval_ms, blit=False, repeat=False)

    def to_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for i in range(len(self.history["tick"])):
            rows.append(
                {
                    "tick": self.history["tick"][i],
                    "total_civilians": self.history["total_civilians"][i],
                    "active_insurgents": self.history["active_insurgents"][i],
                    "latent_insurgents": self.history["latent_insurgents"][i],
                    "cumulative_killed": self.history["cumulative_killed"][i],
                    "recent_attacks": self.history["recent_attacks"][i],
                    "avg_anger": self.history["avg_anger"][i],
                    "avg_fear": self.history["avg_fear"][i],
                }
            )
        return rows


# Utility / export functions


def run_one_simulation(
    grid_width: int = 50,
    grid_height: int = 50,
    num_civilians: int = 500,
    num_soldiers: int = 100,
    interaction_range: int = 3,
    effectiveness: float = 0.5,
    accuracy: float = 0.5,
    p_gr: float = 0.5,
    p_ir: float = 0.5,
    p_iewr: float = 0.5,
    soldier_recruit_n: int = 5,
    soldier_anger_delta: float = -0.15,
    insurgent_recruit_n: int = 5,
    insurgent_anger_delta: float = 0.15,
    seed: int = 1,
    max_ticks: int = 5000,
    latent_rule: str = "formal",
    animate: bool = False,
    interval_ms: int = 60,
    parallel_actions: bool = True,
) -> CoinModel:
    model = CoinModel(
        grid_width=grid_width,
        grid_height=grid_height,
        num_civilians=num_civilians,
        num_soldiers=num_soldiers,
        interaction_range=interaction_range,
        effectiveness=effectiveness,
        accuracy=accuracy,
        p_gr=p_gr,
        p_ir=p_ir,
        p_iewr=p_iewr,
        soldier_recruit_n=soldier_recruit_n,
        soldier_anger_delta=soldier_anger_delta,
        insurgent_recruit_n=insurgent_recruit_n,
        insurgent_anger_delta=insurgent_anger_delta,
        seed=seed,
        max_ticks=max_ticks,
        latent_rule=latent_rule,
        parallel_actions=parallel_actions,
    )
    if animate:
        anim = model.animate(interval_ms=interval_ms)
        plt.show()
        _ = anim
    else:
        model.run()
    return model


def save_summary_figure(model: CoinModel, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = model.summary_figure()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_individual_figures(model: CoinModel, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    model.plot_grid(ax)
    fig.tight_layout()
    fig.savefig(outdir / "grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    model.plot_histories(ax)
    fig.tight_layout()
    fig.savefig(outdir / "actors_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    model.plot_anger_fear(ax)
    fig.tight_layout()
    fig.savefig(outdir / "anger_fear.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = model.summary_figure()
    fig.savefig(outdir / "summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rows = model.to_rows()
    with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "tick_count": model.tick_count,
        "termination_reason": model.termination_reason,
        "cumulative_insurgents_killed": model.cumulative_insurgents_killed,
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def plot_duration_surface(
    accuracy_values: Sequence[float],
    effectiveness_values: Sequence[float],
    p_gr: float = 0.5,
    p_ir: float = 0.5,
    p_iewr: float = 0.5,
    soldier_recruit_n: int = 5,
    soldier_anger_delta: float = -0.15,
    insurgent_recruit_n: int = 5,
    insurgent_anger_delta: float = 0.15,
    seed: int = 1,
    max_ticks: int = 5000,
    n_replicates: int = 1,
    parallel_actions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = np.array(list(accuracy_values), dtype=float)
    eff = np.array(list(effectiveness_values), dtype=float)
    X, Y = np.meshgrid(acc, eff)
    Z = np.zeros_like(X, dtype=float)

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            durations = []
            for r in range(n_replicates):
                model = CoinModel(
                    effectiveness=float(Y[i, j]),
                    accuracy=float(X[i, j]),
                    p_gr=p_gr,
                    p_ir=p_ir,
                    p_iewr=p_iewr,
                    soldier_recruit_n=soldier_recruit_n,
                    soldier_anger_delta=soldier_anger_delta,
                    insurgent_recruit_n=insurgent_recruit_n,
                    insurgent_anger_delta=insurgent_anger_delta,
                    seed=seed + r,
                    max_ticks=max_ticks,
                    parallel_actions=parallel_actions,
                )
                model.run()
                durations.append(model.tick_count)
            Z[i, j] = float(np.mean(durations))
    return X, Y, Z


def show_duration_surface(
    accuracy_values: Sequence[float],
    effectiveness_values: Sequence[float],
    **kwargs,
) -> plt.Figure:
    X, Y, Z = plot_duration_surface(accuracy_values, effectiveness_values, **kwargs)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Effectiveness")
    ax.set_zlabel("Duration of insurgency")
    ax.set_title("Duration Surface (Accuracy vs Effectiveness)")
    ax.set_zlim(0, 5000)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    fig.tight_layout()
    return fig


def save_duration_surface_figure(
    output_path: str | Path,
    accuracy_values: Sequence[float],
    effectiveness_values: Sequence[float],
    **kwargs,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = show_duration_surface(accuracy_values, effectiveness_values, **kwargs)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_paper_experiments(output_dir: str | Path = "results", seed: int = 1, parallel_actions: bool = True) -> None:
    """
    Generate the nine paper-style scenarios:
    - Fig3 A/B/C: effectiveness=0.2, accuracy=0.2
    - Fig4 A/B/C: effectiveness=0.5, accuracy=0.5
    - Fig5 A/B/C: effectiveness=0.8, accuracy=0.8
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("Fig3_A", 0.2, 0.2, 0.0, 0.0, 0.0, 5, -0.15, 5, 0.15),
        ("Fig3_B", 0.2, 0.2, 0.5, 0.0, 0.0, 9, -0.25, 5, 0.15),
        ("Fig3_C", 0.2, 0.2, 0.5, 0.5, 1.0, 9, -0.25, 9, 0.25),

        ("Fig4_A", 0.5, 0.5, 0.0, 0.0, 0.0, 5, -0.15, 5, 0.15),
        ("Fig4_B", 0.5, 0.5, 0.5, 0.0, 0.0, 9, -0.25, 5, 0.15),
        ("Fig4_C", 0.5, 0.5, 0.5, 0.5, 1.0, 9, -0.25, 9, 0.25),

        ("Fig5_A", 0.8, 0.8, 0.0, 0.0, 0.0, 5, -0.15, 5, 0.15),
        ("Fig5_B", 0.8, 0.8, 0.5, 0.0, 0.0, 9, -0.25, 5, 0.15),
        ("Fig5_C", 0.8, 0.8, 0.5, 0.5, 1.0, 9, -0.25, 9, 0.25),
    ]

    for name, eff, acc, pgr, pir, piewr, s_n, s_delta, i_n, i_delta in cases:
        run_dir = output_dir / name
        model = CoinModel(
            effectiveness=eff,
            accuracy=acc,
            p_gr=pgr,
            p_ir=pir,
            p_iewr=piewr,
            soldier_recruit_n=s_n,
            soldier_anger_delta=s_delta,
            insurgent_recruit_n=i_n,
            insurgent_anger_delta=i_delta,
            seed=seed,
            max_ticks=5000,
            parallel_actions=parallel_actions,
        )
        model.run()
        save_individual_figures(model, run_dir)
        print(
            f"{name}: ticks={model.tick_count}, "
            f"termination={model.termination_reason}, "
            f"killed={model.cumulative_insurgents_killed}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COIN agent-based simulation")
    sub = p.add_subparsers(dest="mode", required=False)

    sim = sub.add_parser("simulate", help="Run one simulation")
    sim.add_argument("--grid-width", type=int, default=50)
    sim.add_argument("--grid-height", type=int, default=50)
    sim.add_argument("--num-civilians", type=int, default=500)
    sim.add_argument("--num-soldiers", type=int, default=100)
    sim.add_argument("--interaction-range", type=int, default=3)
    sim.add_argument("--effectiveness", type=float, default=0.5)
    sim.add_argument("--accuracy", type=float, default=0.5)
    sim.add_argument("--p-gr", type=float, default=0.5)
    sim.add_argument("--p-ir", type=float, default=0.5)
    sim.add_argument("--p-iewr", type=float, default=0.5)
    sim.add_argument("--soldier-recruit-n", type=int, default=5)
    sim.add_argument("--soldier-anger-delta", type=float, default=-0.15)
    sim.add_argument("--insurgent-recruit-n", type=int, default=5)
    sim.add_argument("--insurgent-anger-delta", type=float, default=0.15)
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--max-ticks", type=int, default=5000)
    sim.add_argument("--latent-rule", type=str, choices=["formal"], default="formal")
    sim.add_argument("--animate", action="store_true")
    sim.add_argument("--interval-ms", type=int, default=60)
    sim.add_argument("--output", type=str, default="coin_summary.png")

    surf = sub.add_parser("surface", help="Compute and save a duration surface")
    surf.add_argument("--accuracy-values", type=str, default="0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    surf.add_argument("--effectiveness-values", type=str, default="0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    surf.add_argument("--p-gr", type=float, default=0.5)
    surf.add_argument("--p-ir", type=float, default=0.5)
    surf.add_argument("--p-iewr", type=float, default=0.5)
    surf.add_argument("--soldier-recruit-n", type=int, default=5)
    surf.add_argument("--soldier-anger-delta", type=float, default=-0.15)
    surf.add_argument("--insurgent-recruit-n", type=int, default=5)
    surf.add_argument("--insurgent-anger-delta", type=float, default=0.15)
    surf.add_argument("--seed", type=int, default=1)
    surf.add_argument("--max-ticks", type=int, default=5000)
    surf.add_argument("--replicates", type=int, default=1)
    surf.add_argument("--output", type=str, default="coin_duration_surface.png")

    paper = sub.add_parser("paper", help="Run the nine paper-style case studies")
    paper.add_argument("--output-dir", type=str, default="results")
    paper.add_argument("--seed", type=int, default=1)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode in (None, "simulate"):
        model = run_one_simulation(
            grid_width=getattr(args, "grid_width", 50),
            grid_height=getattr(args, "grid_height", 50),
            num_civilians=getattr(args, "num_civilians", 500),
            num_soldiers=getattr(args, "num_soldiers", 100),
            interaction_range=getattr(args, "interaction_range", 3),
            effectiveness=getattr(args, "effectiveness", 0.5),
            accuracy=getattr(args, "accuracy", 0.5),
            p_gr=getattr(args, "p_gr", 0.5),
            p_ir=getattr(args, "p_ir", 0.5),
            p_iewr=getattr(args, "p_iewr", 0.5),
            soldier_recruit_n=getattr(args, "soldier_recruit_n", 5),
            soldier_anger_delta=getattr(args, "soldier_anger_delta", -0.15),
            insurgent_recruit_n=getattr(args, "insurgent_recruit_n", 5),
            insurgent_anger_delta=getattr(args, "insurgent_anger_delta", 0.15),
            seed=getattr(args, "seed", 1),
            max_ticks=getattr(args, "max_ticks", 5000),
            latent_rule=getattr(args, "latent_rule", "formal"),
            animate=getattr(args, "animate", False),
            interval_ms=getattr(args, "interval_ms", 60),
        )
        if not getattr(args, "animate", False):
            save_summary_figure(model, getattr(args, "output", "coin_summary.png"))
            print(f"Saved summary figure to {getattr(args, 'output', 'coin_summary.png')}")
            print(f"Termination reason: {model.termination_reason}")
            print(f"Ticks: {model.tick_count}")
        return

    if args.mode == "surface":
        acc_vals = parse_float_list(args.accuracy_values)
        eff_vals = parse_float_list(args.effectiveness_values)
        save_duration_surface_figure(
            getattr(args, "output", "coin_duration_surface.png"),
            accuracy_values=acc_vals,
            effectiveness_values=eff_vals,
            p_gr=args.p_gr,
            p_ir=args.p_ir,
            p_iewr=args.p_iewr,
            soldier_recruit_n=args.soldier_recruit_n,
            soldier_anger_delta=args.soldier_anger_delta,
            insurgent_recruit_n=args.insurgent_recruit_n,
            insurgent_anger_delta=args.insurgent_anger_delta,
            seed=args.seed,
            max_ticks=args.max_ticks,
            n_replicates=args.replicates,
        )
        print(f"Saved duration surface to {getattr(args, 'output', 'coin_duration_surface.png')}")
        return

    if args.mode == "paper":
        run_paper_experiments(output_dir=args.output_dir, seed=args.seed)
        return


if __name__ == "__main__":
    main()
