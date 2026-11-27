"""Prepare truth/observations and grade frozen selections separately from search."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from controller.artifacts import read_arrays, read_json, write_arrays, write_json
from controller.hypotheses import parse_hypothesis
from controller.progress import logger
from numerical.evaluation import (
    parameter_summary,
    predictive_scores,
)
from numerical.geology import (
    BandGeometry,
    Family,
    band_cells,
    encode_parameters,
    permeability_field,
)
from numerical.pressure import PressureModel
from numerical.report import CandidateDiagnostics, diagnostic_pngs
from pressure_case import Phase, PressureCase


def prepare_case(output: Path, case: PressureCase) -> tuple[Path, Path]:
    """Generate fixed truth and observations, separating evidence/ from protected/.

    Return the evidence and protected directories.
    """
    truth_family = Family.VERTICAL_BAND
    truth_geometry = BandGeometry(center_fraction=0.65, width_fraction=0.1)
    background_md = 100.0
    band_ratio = 0.01
    noise_seed = 2026
    parameters = encode_parameters(truth_family, background_md, band_ratio)
    model = PressureModel(case)
    field = permeability_field(truth_family, parameters, case, truth_geometry)
    logger.info("Simulating synthetic pressure history: %d time steps", case.total_steps)
    truth = model.simulate(field, case.total_steps)
    rng = np.random.default_rng(noise_seed)
    observed = truth.responses_pa + rng.normal(
        0, case.observation_std_pa, size=truth.responses_pa.shape
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    protected = output / "protected"
    for phase in Phase:
        mask = case.phase_mask(phase)
        count = int(mask.sum())
        directory = protected if phase == Phase.FINAL else evidence
        write_arrays(
            directory / f"{phase.value}.npz",
            values_pa=observed[mask],
            observation_ids=case.observation_ids[mask],
        )
        logger.info(
            "Saved %s observations: %d readings",
            "protected final" if phase == Phase.FINAL else phase.value,
            count,
        )
    write_arrays(
        protected / "truth.npz",
        permeability_m2=field,
        clean_responses_pa=truth.responses_pa,
    )
    write_json(
        protected / "truth.json",
        {
            "schema_version": 4,
            "family": truth_family.value,
            "background_md": background_md,
            "band_ratio": band_ratio,
            "geometry": {
                "center_fraction": truth_geometry.center_fraction,
                "width_fraction": truth_geometry.width_fraction,
            },
            "noise_seed": noise_seed,
            "noise_rng_state_after": rng.bit_generator.state,
            "truth_simulator_calls": 1,
            "max_balance_error_m3": truth.max_balance_error_m3,
        },
    )
    logger.info("Saved hidden truth; protected values are not sent to the proposer")
    return evidence, protected


def evaluate_selection(
    evidence: Path, protected: Path, run: Path, output: Path, case: PressureCase
) -> dict[str, object]:
    """Grade a frozen selection against protected truth/final observations without refitting."""
    selection = read_json(run / "selection.json")
    logger.info("Grading frozen selection against protected truth and final observations")
    truth_manifest = read_json(protected / "truth.json")
    truth = read_arrays(protected / "truth.npz")
    final = load_observations(protected, case, Phase.FINAL)
    reports: list[dict[str, object]] = []
    diagnostics: list[CandidateDiagnostics] = []
    selected = Family(selection["selected_family"])
    selected_id = selection["selected_candidate_id"]
    selected_geometry = None
    for candidate in selection["candidates"]:
        family = Family(candidate["family"])
        hypothesis = parse_hypothesis(candidate["hypothesis"], case)
        posterior = load_posterior(run, case, candidate)
        fields = np.stack(
            [
                permeability_field(family, values, case, hypothesis.geometry)
                for values in posterior["parameters"].T
            ]
        )
        mean_log_field = np.log(fields).mean(axis=0)
        fitted_field = np.exp(mean_log_field)
        diagnostics.append(
            CandidateDiagnostics(
                candidate["candidate_id"],
                family,
                hypothesis.geometry,
                fitted_field,
                posterior["predictions_pa"],
                candidate["validation"]["crps_pa"],
            )
        )
        if candidate["candidate_id"] == selected_id:
            selected_geometry = hypothesis.geometry
        final_scores = predictive_scores(
            posterior["predictions_pa"][case.phase_mask(Phase.FINAL)], final
        )
        reports.append(
            {
                "family": family.value,
                "candidate_id": candidate["candidate_id"],
                "hypothesis": hypothesis.specification(),
                "final": final_scores,
                "log_permeability_rmse": float(
                    np.sqrt(np.mean((mean_log_field - np.log(truth["permeability_m2"])) ** 2))
                ),
                "parameters": parameter_summary(family, posterior["parameters"]),
            }
        )
        logger.info(
            "Graded candidate %s (%s): final RMSE=%.2f Pa; final CRPS=%.2f Pa",
            candidate["candidate_id"],
            family.value,
            final_scores["rmse_pa"],
            final_scores["crps_pa"],
        )
    selected_report = next(report for report in reports if report["candidate_id"] == selected_id)
    family_recovered = selected.value == truth_manifest["family"]
    geometry_recovered = family_recovered
    if geometry_recovered and selected_geometry is not None:
        geometry_recovered = np.array_equal(
            band_cells(case, selected_geometry),
            band_cells(case, BandGeometry(**truth_manifest["geometry"])),
        )
    parameter_recovery = None
    if geometry_recovered:
        targets = {"background_md": truth_manifest["background_md"]}
        if selected != Family.HOMOGENEOUS:
            targets["band_ratio"] = truth_manifest["band_ratio"]
        parameter_recovery = {
            name: {
                "truth": target,
                "relative_median_error": abs(summary["median"] - target) / target,
                "covered_by_90_interval": summary["q05"] <= target <= summary["q95"],
            }
            for name, target in targets.items()
            for summary in [selected_report["parameters"][name]]
        }
    report: dict[str, object] = {
        "schema_version": 2,
        "selected_family": selected.value,
        "truth_family": truth_manifest["family"],
        "family_recovered": family_recovered,
        "structural_truth_recovered": geometry_recovered,
        "selected_candidate_id": selected_id,
        "search_outcome": selection["outcome"],
        "proposer_mode": selection["proposer_mode"],
        "model_calls": selection["model_calls"],
        "proposer_models": selection["proposer_models"],
        "parameter_recovery": parameter_recovery,
        "candidates": reports,
        "simulator_calls": selection["simulator_calls"],
        "truth_simulator_calls": truth_manifest["truth_simulator_calls"],
        "wall_seconds": selection["wall_seconds"],
        "interpretation": (
            "selection only ranks the evaluated hypotheses; observed interval coverage "
            "does not establish 90% coverage across other experiments"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    observed = np.empty(case.observation_ids.size)
    for phase in Phase:
        phase_data = final if phase == Phase.FINAL else load_observations(evidence, case, phase)
        observed[case.phase_mask(phase)] = phase_data
    logger.info("Writing final report and search-evolution diagnostics: %s", output)
    images = diagnostic_pngs(
        case,
        truth["permeability_m2"],
        truth["clean_responses_pa"],
        observed,
        diagnostics,
        selected_id,
    )
    diagnostic_files: list[str] = []
    for page, image in enumerate(images, start=1):
        filename = "diagnostics.png" if page == 1 else f"diagnostics_{page:02d}.png"
        (output / filename).write_bytes(image)
        diagnostic_files.append(filename)
        logger.info("Saved diagnostic page %d: %s", page, output / filename)
    report["diagnostic_files"] = diagnostic_files
    write_json(output / "report.json", report)
    logger.info(
        "Final assessment: selected=%s; family recovered=%s; structural truth recovered=%s",
        selected_id,
        report["family_recovered"],
        report["structural_truth_recovered"],
    )
    return report


def load_observations(directory: Path, case: PressureCase, phase: Phase) -> NDArray[np.float64]:
    """Load pressure values for a phase, checking consistency with the case."""
    with np.load(directory / f"{phase.value}.npz", allow_pickle=False) as data:
        observation_ids = data["observation_ids"]
        values = data["values_pa"]
    expected_ids = case.observation_ids[case.phase_mask(phase)]
    if not np.array_equal(observation_ids, expected_ids):
        raise ValueError(
            f"{phase.value} observation IDs/order do not match the case; "
            "predictions and observations are paired by row"
        )
    if values.shape != expected_ids.shape:
        raise ValueError(f"{phase.value} values_pa must be a vector with one value per observation")
    if not np.isfinite(values).all():
        raise ValueError(f"{phase.value} values_pa contains NaN or infinity")
    return values


def load_posterior(
    run: Path, case: PressureCase, candidate: dict[str, object]
) -> dict[str, NDArray[np.generic]]:
    """Load fitted parameters and check that predictions line up with observations."""
    arrays = read_arrays(run / candidate["candidate_id"] / "posterior.npz")
    if not np.array_equal(arrays["observation_ids"], case.observation_ids):
        raise ValueError(
            "posterior.npz fitted pressure readings are not in the experiment's reading order"
        )
    realizations = arrays["parameters"].shape[1]
    if arrays["predictions_pa"].shape != (case.observation_ids.size, realizations):
        raise ValueError(
            "posterior.npz predictions_pa must have one row per reading "
            "and one column per realization"
        )
    return arrays
