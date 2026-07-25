from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st

from coin_abm import CoinModel


st.set_page_config(
    page_title="Counterinsurgency Agent-Based Simulation",
    page_icon="🧪",
    layout="wide",
)


def get_default_params() -> Dict[str, Any]:
    return {
        "grid_width": 50,
        "grid_height": 50,
        "num_civilians": 500,
        "num_soldiers": 100,
        "interaction_range": 3,
        "effectiveness": 0.5,
        "accuracy": 0.5,
        "p_gr": 0.5,
        "p_ir": 0.5,
        "p_iewr": 0.5,
        "soldier_recruit_n": 5,
        "soldier_anger_delta": -0.15,
        "insurgent_recruit_n": 5,
        "insurgent_anger_delta": 0.15,
        "seed": 1,
        "max_ticks": 5000,
        "parallel_actions": True,
    }


def collect_sidebar_params() -> Dict[str, Any]:
    params = get_default_params()

    with st.sidebar:
        st.title("Simulation Controls")
        st.caption("Adjust the model parameters below and run the simulation.")

        st.subheader("Population & Grid")
        params["grid_width"] = st.number_input(
            "Grid Width (W)",
            value=params["grid_width"],
            step=1,
            help="Number of columns in the simulation grid.",
        )
        params["grid_height"] = st.number_input(
            "Grid Height (H)",
            value=params["grid_height"],
            step=1,
            help="Number of rows in the simulation grid.",
        )
        params["num_civilians"] = st.number_input(
            "Civilians (N_c)",
            value=params["num_civilians"],
            step=1,
            help="Total number of civilians in the model.",
        )
        params["num_soldiers"] = st.number_input(
            "Soldiers (N_s)",
            value=params["num_soldiers"],
            step=1,
            help="Total number of government soldiers in the model.",
        )
        params["interaction_range"] = st.number_input(
            "Interaction Range (r)",
            value=params["interaction_range"],
            step=1,
            help="Radius within which agents can observe and interact.",
        )

        st.subheader("Behavioral Parameters")
        params["effectiveness"] = st.number_input(
            "Effectiveness (E)",
            min_value=0.0,
            max_value=1.0,
            value=params["effectiveness"],
            step=0.01,
            help="Probability that a soldier successfully neutralizes an identified insurgent. Represents the operational capability of government forces.",
        )
        params["accuracy"] = st.number_input(
            "Accuracy (A)",
            min_value=0.0,
            max_value=1.0,
            value=params["accuracy"],
            step=0.01,
            help="Probability that intelligence correctly identifies the actual insurgent target. Determines the quality of targeting and collateral damage.",
        )
        params["p_gr"] = st.number_input(
            "P_GR",
            min_value=0.0,
            max_value=1.0,
            value=params["p_gr"],
            step=0.01,
            help="Probability that insurgents successfully recruit a nearby civilian. Controls the growth rate of insurgency through recruitment.",
        )
        params["p_ir"] = st.number_input(
            "P_IR",
            min_value=0.0,
            max_value=1.0,
            value=params["p_ir"],
            step=0.01,
            help="Probability that a latent insurgent becomes an active insurgent. Determines how quickly passive support transforms into active violence.",
        )
        params["p_iewr"] = st.number_input(
            "P_IEWR",
            min_value=0.0,
            max_value=1.0,
            value=params["p_iewr"],
            step=0.01,
            help="Probability that an insurgent attack event succeeds. Represents insurgent operational effectiveness and ability to inflict damage.",
        )

        st.subheader("Recruitment & Anger")
        params["soldier_recruit_n"] = st.number_input(
            "Soldier Recruitment Count",
            value=params["soldier_recruit_n"],
            step=1,
            help="Number of civilians a soldier can recruit during one action.",
        )
        params["soldier_anger_delta"] = st.number_input(
            "Soldier Anger Delta",
            value=params["soldier_anger_delta"],
            step=0.01,
            help="Change in civilian anger caused by soldier recruitment.",
        )
        params["insurgent_recruit_n"] = st.number_input(
            "Insurgent Recruitment Count",
            value=params["insurgent_recruit_n"],
            step=1,
            help="Number of civilians an insurgent can recruit during one action.",
        )
        params["insurgent_anger_delta"] = st.number_input(
            "Insurgent Anger Delta",
            value=params["insurgent_anger_delta"],
            step=0.01,
            help="Change in civilian anger caused by insurgent recruitment.",
        )

        st.subheader("Simulation Controls")
        params["seed"] = st.number_input(
            "Random Seed (S)",
            value=params["seed"],
            step=1,
            help="Random seed used for pseudo-random number generation and reproducibility.",
        )
        params["max_ticks"] = st.number_input(
            "Maximum Ticks (T_max)",
            value=params["max_ticks"],
            step=1,
            help="Maximum number of simulation iterations before the run stops.",
        )
        params["parallel_actions"] = st.checkbox(
            "Parallel Actions",
            value=params["parallel_actions"],
            help="Run actions for all insurgents and soldiers during each tick.",
        )

        st.divider()
        run_button = st.button("Run Simulation", type="primary", use_container_width=True)

    return params, run_button


def build_model(params: Dict[str, Any]) -> CoinModel:
    return CoinModel(
        grid_width=int(params["grid_width"]),
        grid_height=int(params["grid_height"]),
        num_civilians=int(params["num_civilians"]),
        num_soldiers=int(params["num_soldiers"]),
        interaction_range=int(params["interaction_range"]),
        effectiveness=float(params["effectiveness"]),
        accuracy=float(params["accuracy"]),
        p_gr=float(params["p_gr"]),
        p_ir=float(params["p_ir"]),
        p_iewr=float(params["p_iewr"]),
        soldier_recruit_n=int(params["soldier_recruit_n"]),
        soldier_anger_delta=float(params["soldier_anger_delta"]),
        insurgent_recruit_n=int(params["insurgent_recruit_n"]),
        insurgent_anger_delta=float(params["insurgent_anger_delta"]),
        seed=int(params["seed"]),
        max_ticks=int(params["max_ticks"]),
        parallel_actions=bool(params["parallel_actions"]),
    )


def run_model_with_progress(model: CoinModel, max_ticks: int) -> None:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    while model.termination_reason is None and model.tick_count < max_ticks:
        model.step()
        percent = min(100, int((model.tick_count / max_ticks) * 100)) if max_ticks else 100
        progress_text.info(f"Tick {model.tick_count}/{max_ticks} completed")
        progress_bar.progress(percent)


def compute_outcome(model: CoinModel) -> str:
    if model.termination_reason == "state_victory":
        return "Government victory"
    if model.termination_reason == "sustained_insurgency":
        return "Sustained insurgency"
    if len(model._active_insurgents()) > len(model._latent_insurgents()):
        return "Insurgent victory"
    return "Simulation completed"


def collect_metrics(model: CoinModel) -> Dict[str, Any]:
    final_anger = float(sum(c.anger for c in model.civilians.values()) / len(model.civilians)) if model.civilians else 0.0
    final_fear = float(sum(c.fear for c in model.civilians.values()) / len(model.civilians)) if model.civilians else 0.0

    return {
        "ticks": model.tick_count,
        "active_insurgents": len(model._active_insurgents()),
        "latent_insurgents": len(model._latent_insurgents()),
        "civilian_deaths": model.cumulative_insurgents_killed,
        "total_killed_insurgents": model.cumulative_insurgents_killed,
        "avg_anger": final_anger,
        "avg_fear": final_fear,
    }


def build_summary_text(model: CoinModel, outcome: str, metrics: Dict[str, Any]) -> str:
    lines = [
        "COIN Simulation Summary",
        "=" * 28,
        f"Outcome: {outcome}",
        f"Termination Reason: {model.termination_reason}",
        f"Total Ticks: {metrics['ticks']}",
        f"Active Insurgents: {metrics['active_insurgents']}",
        f"Latent Insurgents: {metrics['latent_insurgents']}",
        f"Civilian Deaths: {metrics['civilian_deaths']}",
        f"Total Killed Insurgents: {metrics['total_killed_insurgents']}",
        f"Average Anger: {metrics['avg_anger']:.4f}",
        f"Average Fear: {metrics['avg_fear']:.4f}",
    ]
    return "\n".join(lines)


def build_graphs(model: CoinModel) -> Dict[str, Any]:
    figures: Dict[str, Any] = {}

    fig_history, ax_history = plt.subplots(figsize=(8, 5))
    model.plot_histories(ax_history)
    fig_history.tight_layout()
    figures["actor_history"] = fig_history

    fig_af, ax_af = plt.subplots(figsize=(8, 5))
    model.plot_anger_fear(ax_af)
    fig_af.tight_layout()
    figures["anger_fear"] = fig_af

    return figures


def prepare_downloads(model: CoinModel, outcome: str, metrics: Dict[str, Any], figures: Dict[str, Any]) -> Dict[str, bytes]:
    csv_buffer = io.StringIO()
    rows = model.to_rows()
    headers = list(rows[0].keys()) if rows else [
        "tick",
        "total_civilians",
        "active_insurgents",
        "latent_insurgents",
        "cumulative_killed",
        "recent_attacks",
        "avg_anger",
        "avg_fear",
    ]
    csv_buffer.write(",".join(headers) + "\n")
    for row in rows:
        csv_buffer.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    png_bytes = io.BytesIO()
    with zipfile.ZipFile(png_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, fig in figures.items():
            fig_buffer = io.BytesIO()
            fig.savefig(fig_buffer, format="png", dpi=200, bbox_inches="tight")
            fig_buffer.seek(0)
            archive.writestr(f"{name}.png", fig_buffer.getvalue())
    png_bytes.seek(0)

    summary_text = build_summary_text(model, outcome, metrics).encode("utf-8")

    return {
        "csv": csv_buffer.getvalue().encode("utf-8"),
        "pngs": png_bytes.getvalue(),
        "summary": summary_text,
    }


def render_app() -> None:
    st.title("Counterinsurgency Agent-Based Simulation")
    st.caption("Interactive implementation of a COIN model")

    params, run_button = collect_sidebar_params()

    tabs = st.tabs(["Simulation Results", "Graphs", "Raw Data"])

    with tabs[0]:
        if run_button:
            try:
                with st.spinner("Running simulation..."):
                    model = build_model(params)
                    run_model_with_progress(model, int(params["max_ticks"]))

                outcome = compute_outcome(model)
                metrics = collect_metrics(model)
                summary_text = build_summary_text(model, outcome, metrics)

                st.success("Simulation completed successfully.")
                st.subheader("Simulation Outcome")
                st.write(outcome)

                metric_cols = st.columns(4)
                metric_cols[0].metric("Total ticks", metrics["ticks"])
                metric_cols[1].metric("Active insurgents", metrics["active_insurgents"])
                metric_cols[2].metric("Latent insurgents", metrics["latent_insurgents"])
                metric_cols[3].metric("Civilian deaths", metrics["civilian_deaths"])

                metric_cols_2 = st.columns(4)
                metric_cols_2[0].metric("Total killed insurgents", metrics["total_killed_insurgents"])
                metric_cols_2[1].metric("Average anger", f"{metrics['avg_anger']:.4f}")
                metric_cols_2[2].metric("Average fear", f"{metrics['avg_fear']:.4f}")
                metric_cols_2[3].metric("Termination", model.termination_reason or "running")

                downloads = prepare_downloads(model, outcome, metrics, build_graphs(model))
                st.download_button(
                    "Download CSV results",
                    data=downloads["csv"],
                    file_name="coin_results.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download PNG graphs",
                    data=downloads["pngs"],
                    file_name="coin_graphs.zip",
                    mime="application/zip",
                )
                st.download_button(
                    "Download summary text",
                    data=downloads["summary"],
                    file_name="coin_summary.txt",
                    mime="text/plain",
                )

                st.session_state["last_model"] = model
                st.session_state["last_metrics"] = metrics
                st.session_state["last_outcome"] = outcome
                st.session_state["last_summary"] = summary_text
                st.session_state["last_graphs"] = build_graphs(model)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Simulation failed: {exc}")
        elif "last_model" in st.session_state:
            model = st.session_state["last_model"]
            metrics = st.session_state["last_metrics"]
            outcome = st.session_state["last_outcome"]
            st.success("Loaded the most recent simulation results.")
            st.subheader("Simulation Outcome")
            st.write(outcome)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total ticks", metrics["ticks"])
            metric_cols[1].metric("Active insurgents", metrics["active_insurgents"])
            metric_cols[2].metric("Latent insurgents", metrics["latent_insurgents"])
            metric_cols[3].metric("Civilian deaths", metrics["civilian_deaths"])

            metric_cols_2 = st.columns(4)
            metric_cols_2[0].metric("Total killed insurgents", metrics["total_killed_insurgents"])
            metric_cols_2[1].metric("Average anger", f"{metrics['avg_anger']:.4f}")
            metric_cols_2[2].metric("Average fear", f"{metrics['avg_fear']:.4f}")
            metric_cols_2[3].metric("Termination", model.termination_reason or "running")
        else:
            st.info("Run the simulation to view the results.")

    with tabs[1]:
        if "last_graphs" in st.session_state:
            graphs = st.session_state["last_graphs"]
            st.subheader("Actor History")
            st.pyplot(graphs["actor_history"])
            st.subheader("Anger / Fear")
            st.pyplot(graphs["anger_fear"])
        else:
            st.info("Run the simulation to display the generated figures.")

    with tabs[2]:
        if "last_model" in st.session_state:
            model = st.session_state["last_model"]
            rows = model.to_rows()
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Run the simulation to inspect the raw data series.")


if __name__ == "__main__":
    render_app()
