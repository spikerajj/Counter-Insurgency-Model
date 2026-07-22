
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from coin_abm import (
    run_one_simulation,
    run_paper_experiments,
    save_summary_figure,
)


# Interactive prompts


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return int(raw) if raw else default


def _prompt_bool(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{label} ({suffix}): ").strip().lower()

    if not raw:
        return default

    return raw in {"y", "yes", "true", "1"}


# CLI


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Counterinsurgency Agent-Based Model"
    )

    sub = parser.add_subparsers(dest="mode")

    # Single simulation mode
    sim = sub.add_parser(
        "simulate",
        help="Run one simulation"
    )

    sim.add_argument("--grid-width", type=int)
    sim.add_argument("--grid-height", type=int)
    sim.add_argument("--num-civilians", type=int)
    sim.add_argument("--num-soldiers", type=int)
    sim.add_argument("--interaction-range", type=int)

    sim.add_argument("--effectiveness", type=float)
    sim.add_argument("--accuracy", type=float)

    sim.add_argument("--p-gr", dest="p_gr", type=float)
    sim.add_argument("--p-ir", dest="p_ir", type=float)
    sim.add_argument("--p-iewr", dest="p_iewr", type=float)

    sim.add_argument("--soldier-recruit-n", type=int)
    sim.add_argument("--soldier-anger-delta", type=float)

    sim.add_argument("--insurgent-recruit-n", type=int)
    sim.add_argument("--insurgent-anger-delta", type=float)

    sim.add_argument("--seed", type=int)
    sim.add_argument("--max-ticks", type=int)

    sim.add_argument(
        "--output",
        default="coin_summary.png"
    )

    # Paper mode
    paper = sub.add_parser(
        "paper",
        help="Run paper experiments"
    )

    paper.add_argument(
        "--output-dir",
        default="results"
    )

    paper.add_argument(
        "--seed",
        type=int,
        default=1
    )

    return parser

# Interactive parameter collection


def interactive_parameters() -> dict:

    print(
        "\nEnter simulation parameters "
        "(press Enter for defaults)\n"
    )

    num_civilians = _prompt_int("Number of civilians", 500)
    num_soldiers = _prompt_int("Number of soldiers", 100)
    grid_width = _prompt_int("Grid width", 50)
    grid_height = _prompt_int("Grid height", 50)
    vision_radius = _prompt_int("Vision radius (interaction range)", 3)
    max_ticks = _prompt_int("Maximum ticks", 5000)
    soldier_recruit_n = _prompt_int("Number of civilians recruited by soldier", 5)
    insurgent_recruit_n = _prompt_int("Number of civilians recruited by insurgent", 5)

    effectiveness = _prompt_float("Effectiveness", 0.5)
    accuracy = _prompt_float("Accuracy", 0.5)
    p_gr = _prompt_float("P_GR (probability soldier recruits)", 0.5)
    p_ir = _prompt_float("P_IR (probability insurgent recruits)", 0.5)
    p_iewr = _prompt_float("P_IEWR (insurgent exposed while recruiting)", 0.5)
    soldier_anger_delta = _prompt_float("Soldier anger change", -0.15)
    insurgent_anger_delta = _prompt_float("Insurgent anger change", 0.15)
    seed = _prompt_int("Random seed", 1)

    return {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "num_civilians": num_civilians,
        "num_soldiers": num_soldiers,
        "interaction_range": vision_radius,
        "effectiveness": effectiveness,
        "accuracy": accuracy,
        "p_gr": p_gr,
        "p_ir": p_ir,
        "p_iewr": p_iewr,
        "soldier_recruit_n": soldier_recruit_n,
        "soldier_anger_delta": soldier_anger_delta,
        "insurgent_recruit_n": insurgent_recruit_n,
        "insurgent_anger_delta": insurgent_anger_delta,
        "seed": seed,
        "max_ticks": max_ticks,
        "animate": False,
        "interval_ms": 60,
        "parallel_actions": True,
    }


def _print_final_summary(model, output_path: Path, animate: bool) -> None:
    print("\n" + "="*50)
    print("             COIN SIMULATION FINAL RESULTS")
    print("="*50)
    print(f"Termination Reason:   {model.termination_reason}")
    print(f"Ticks Simulated:      {model.tick_count}")
    print(f"Insurgents Killed:    {model.cumulative_insurgents_killed}")
    
    # Calculate final averages
    final_anger = float(np.mean([c.anger for c in model.civilians.values()])) if model.civilians else 0.0
    final_fear = float(np.mean([c.fear for c in model.civilians.values()])) if model.civilians else 0.0
    print(f"Final Average Anger:  {final_anger:.4f}")
    print(f"Final Average Fear:   {final_fear:.4f}")
    
    # Calculate counts
    active_count = len(model._active_insurgents())
    latent_count = len(model._latent_insurgents())
    neutral_count = len([c for c in model.civilians.values() if c.state == "neutral"])
    discontent_count = len([c for c in model.civilians.values() if c.state == "discontent"])
    
    print(f"Active Insurgents:    {active_count}")
    print(f"Latent Insurgents:    {latent_count}")
    print(f"Discontent Civilians: {discontent_count}")
    print(f"Neutral Civilians:    {neutral_count}")
    print("="*50)
    if not animate:
        print(f"Saved Summary Figure (excl. grid map) to:\n{output_path.resolve()}")
    print("="*50 + "\n")


# Main


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    # Paper batch mode

    if args.mode == "paper":
        run_paper_experiments(
            output_dir=args.output_dir,
            seed=args.seed
        )
        return

    # CLI simulation mode

    if args.mode == "simulate":
        params = {
            "grid_width": args.grid_width,
            "grid_height": args.grid_height,
            "num_civilians": args.num_civilians,
            "num_soldiers": args.num_soldiers,
            "interaction_range": args.interaction_range,
            "effectiveness": args.effectiveness,
            "accuracy": args.accuracy,
            "p_gr": args.p_gr,
            "p_ir": args.p_ir,
            "p_iewr": args.p_iewr,
            "soldier_recruit_n": args.soldier_recruit_n,
            "soldier_anger_delta": args.soldier_anger_delta,
            "insurgent_recruit_n": args.insurgent_recruit_n,
            "insurgent_anger_delta": args.insurgent_anger_delta,
            "seed": args.seed,
            "max_ticks": args.max_ticks,
            "animate": False,
            "interval_ms": 60,
            "parallel_actions": True,
        }

        # If user omitted key parameters, switch to interactive mode.
        # We check both effectiveness and num_civilians to decide.
        if params["effectiveness"] is None and params["num_civilians"] is None:
            params = interactive_parameters()
        else:
            # Fill missing CLI params with defaults
            if params["grid_width"] is None: params["grid_width"] = 50
            if params["grid_height"] is None: params["grid_height"] = 50
            if params["num_civilians"] is None: params["num_civilians"] = 500
            if params["num_soldiers"] is None: params["num_soldiers"] = 100
            if params["interaction_range"] is None: params["interaction_range"] = 3
            if params["effectiveness"] is None: params["effectiveness"] = 0.5
            if params["accuracy"] is None: params["accuracy"] = 0.5
            if params["p_gr"] is None: params["p_gr"] = 0.5
            if params["p_ir"] is None: params["p_ir"] = 0.5
            if params["p_iewr"] is None: params["p_iewr"] = 0.5
            if params["soldier_recruit_n"] is None: params["soldier_recruit_n"] = 5
            if params["soldier_anger_delta"] is None: params["soldier_anger_delta"] = -0.15
            if params["insurgent_recruit_n"] is None: params["insurgent_recruit_n"] = 5
            if params["insurgent_anger_delta"] is None: params["insurgent_anger_delta"] = 0.15
            if params["seed"] is None: params["seed"] = 1
            if params["max_ticks"] is None: params["max_ticks"] = 5000

        model = run_one_simulation(**params)

        output_path = Path(
            args.output
            if args.output
            else "coin_summary.png"
        )

        if not params["animate"]:
            save_summary_figure(
                model,
                output_path
            )

        _print_final_summary(model, output_path, params["animate"])
        return

    # Default interactive mode

    params = interactive_parameters()

    model = run_one_simulation(**params)

    output_path = Path("coin_summary.png")

    if not params["animate"]:
        save_summary_figure(
            model,
            output_path
        )

    _print_final_summary(model, output_path, params["animate"])


if __name__ == "__main__":
    main()
