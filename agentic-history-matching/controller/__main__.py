"""Run preparation, live LLM search, and protected evaluation from one YAML config."""

import argparse
from pathlib import Path
from time import perf_counter

from cases.pressure import PressureCase
from controller.config import load_config
from controller.experiment import evaluate_selection, prepare_case
from controller.progress import ProgressSection, logger, terminal_progress
from controller.proposer import run_agent_search
from controller.search import SearchController
from numerical.geology import BandGeometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer flow barriers from synthetic pressure observations "
            "with a live LLM through a Pydantic AI model provider"
        ),
        epilog=(
            "LLM proposes model structures; ES-MDA fits permeability parameters. "
            "Truth and final observations are withheld during search. "
            "Output must not already exist; choose another path or delete it manually. "
            "Live requests may incur provider charges. "
            "Progress and a short final summary are written to stderr; "
            "the full JSON report is saved under output/protected/evaluation/report.json."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config",
    )
    args = parser.parse_args()

    with terminal_progress():
        started = perf_counter()
        logger.info("Loading config: %s", args.config)
        config = load_config(args.config)
        output = config.output
        if output.exists():
            raise FileExistsError(
                f"output already exists: {output}; choose a different output path "
                "or delete it manually before running"
            )
        logger.info(
            "Run started: model=%s; realizations=%d; inference seed=%d; output=%s",
            config.model,
            config.realizations,
            config.seed,
            output,
        )
        logger.info("Preparing synthetic observations; truth and final readings stay protected")
        case = PressureCase()
        prepare_case(
            output,
            case,
            truth_geometry=BandGeometry(0.65, 0.1),
        )
        evidence = output / "evidence"
        protected = output / "protected"
        search = output / "search"
        logger.info("Preparation complete: evidence=%s; protected=%s", evidence, protected)
        logger.info("Search started: fit the homogeneous baseline, then request LLM proposals")
        search_controller = SearchController(
            evidence,
            search,
            realizations=config.realizations,
            seed=config.seed,
            max_evaluations=config.max_evaluations,
            max_model_calls=config.max_model_calls,
        )
        logger.info("Starting agent-guided history-matching search")
        run_agent_search(search_controller, config.model)
        logger.info(
            "Agent search complete: transcript=%s; summary=%s",
            search / "proposer" / "messages.jsonl",
            search / "proposer" / "summary.json",
        )
        logger.info("Search complete: frozen selection=%s", search / "selection.json")
        logger.info("Protected evaluation started: verify the frozen selection before grading")
        evaluation = protected / "evaluation"
        report = evaluate_selection(evidence, protected, search, evaluation)
        logger.info("Protected evaluation complete")
        _log_summary(report, evaluation, perf_counter() - started)


def _log_summary(report: dict[str, object], evaluation: Path, elapsed_seconds: float) -> None:
    """Summarize the frozen selection without dumping the saved machine-readable report."""
    selected = next(
        candidate
        for candidate in report["candidates"]
        if candidate["candidate_id"] == report["selected_candidate_id"]
    )
    hypothesis = selected["hypothesis"]
    structure = str(report["selected_family"]).replace("_", " ").capitalize()
    if hypothesis["kind"] == "band":
        structure += (
            f", center {hypothesis['center_fraction']:.1%}"
            f", width {hypothesis['width_fraction']:.1%}"
        )
    scores = selected["final"]
    logger.info(
        ProgressSection(
            "EXPERIMENT COMPLETE",
            (
                f"Search outcome: {str(report['search_outcome']).capitalize()} "
                "(conditioning/validation checks)",
                f"Selected structure: {structure}",
                "Structural truth recovered: "
                f"{'Yes' if report['structural_truth_recovered'] else 'No'}",
                f"Protected final pressure RMSE: {scores['rmse_pa']:,.0f} Pa",
                f"Protected final 90% interval coverage: {scores['coverage_90']:.1%} "
                "of observations",
                f"Resources: {report['model_calls']} LLM requests; "
                f"{report['simulator_calls']} simulator calls",
                f"Run complete in {elapsed_seconds:.2f}s",
                f"Report: {evaluation / 'report.json'}",
                *(
                    f"Diagnostics: {evaluation / filename}"
                    for filename in (
                        "diagnostics.png",
                        *report["additional_diagnostic_pngs_sha256"],
                    )
                ),
            ),
        )
    )


if __name__ == "__main__":
    main()
