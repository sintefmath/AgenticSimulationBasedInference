"""Own sequential hypothesis evaluations, budget accounting, and frozen search decisions."""

from copy import deepcopy
from pathlib import Path

from cases.pressure import Phase
from controller.artifacts import checksum, write_json
from controller.candidate import NUMBER_OF_ITERATIONS, condition_candidate
from controller.experiment import load_case, load_observations
from controller.hypotheses import band_width_bounds, parse_hypothesis
from controller.local import LocalExecutor
from controller.progress import ProgressSection, logger
from numerical.pressure import PressureModel


class ProposalRejected(ValueError):
    """Reject a hypothesis before numerical evaluation starts."""


class SearchController:
    """Run a bounded sequential search, storing only completed candidate evaluations."""

    def __init__(
        self,
        evidence: Path,
        output: Path,
        realizations: int,
        seed: int,
        max_evaluations: int,
        max_model_calls: int,
    ) -> None:
        self.case, self.evidence_manifest = load_case(evidence)
        self.conditioning = load_observations(
            evidence, self.case, self.evidence_manifest, Phase.CONDITIONING
        )
        self.validation = load_observations(
            evidence, self.case, self.evidence_manifest, Phase.VALIDATION
        )
        self.evidence = evidence
        self.output = output
        self.realizations = realizations
        self.seed = seed
        self.max_model_calls = max_model_calls
        self.planned_calls = (NUMBER_OF_ITERATIONS + 1) * realizations
        self.max_simulations = self.planned_calls * max_evaluations
        self.max_evaluations = max_evaluations
        self.adequacy_threshold_pa = 2 * self.case.observation_std_pa
        self.model = PressureModel(self.case)
        self._evaluations: dict[str, dict[str, object]] = {}
        self._model_calls: list[dict[str, str]] = []
        self._outcome: str | None = None
        self._reason: str | None = None
        output.mkdir(parents=True, exist_ok=False)
        logger.info(
            "Search budgets: candidates<=%d; simulator calls<=%d (%d per candidate); "
            "model requests<=%d",
            self.max_evaluations,
            self.max_simulations,
            self.planned_calls,
            self.max_model_calls,
        )
        logger.info(
            "Scores in Pa: RMSE=root mean squared error; "
            "CRPS=continuous ranked probability score (lower is better)"
        )
        logger.info(
            "Pressure-error check: conditioning and validation RMSE<=%.2f Pa; "
            "passing is not proof of geological truth",
            self.adequacy_threshold_pa,
        )

    def evaluate(self, specification: object, reason: str) -> dict[str, object]:
        """Validate and evaluate one hypothesis inline, reusing equivalent evaluations.

        Proposal rejections are correctable; numerical and host failures propagate unchanged.
        """
        try:
            hypothesis = parse_hypothesis(specification, self.case)
            identifier = hypothesis.identifier(self.case)
            existing = self._evaluations.get(identifier)
            if existing is None and len(self._evaluations) >= self.max_evaluations:
                raise ValueError("hypothesis-evaluation budget exhausted")
        except ValueError as error:
            raise ProposalRejected(str(error)) from error
        details = (f"ID: {identifier}",)
        if hypothesis.geometry is None:
            details += ("Structure: homogeneous permeability",)
        else:
            geometry = hypothesis.geometry
            orientation = hypothesis.specification()["orientation"]
            axis = "x" if orientation == "vertical" else "y"
            details += (
                f"Structure: {orientation} low-permeability band",
                f"Center: {geometry.center_fraction}; width: {geometry.width_fraction} "
                "(domain fractions)",
                f"Extent: {axis}={geometry.center_fraction - geometry.width_fraction / 2:.6g}"
                f"..{geometry.center_fraction + geometry.width_fraction / 2:.6g}",
            )
        if existing is not None:
            logger.info(
                ProgressSection(
                    "PROPOSAL REUSED",
                    (
                        *details,
                        "Reusing completed evaluation",
                    ),
                    rationale=reason,
                )
            )
            return self.public_evaluation(existing)
        count = len(self._evaluations)
        title = "NEW HYPOTHESIS"
        if count == 0:
            title = "PREDECLARED BASELINE"
            details += ("Source: predeclared; not an LLM proposal",)
        logger.info(
            ProgressSection(
                f"{title} - CANDIDATE {count + 1}/{self.max_evaluations}",
                (
                    *details,
                    f"Evaluating inline; at most {self.planned_calls} simulator calls",
                ),
                rationale=reason,
            )
        )
        executor = LocalExecutor(self.model, self.planned_calls)
        logger.info(
            "Candidate started: %s (%s); artifacts=%s",
            identifier,
            hypothesis.family.value,
            self.output / identifier,
        )
        result = condition_candidate(
            hypothesis,
            executor,
            self.conditioning,
            self.validation,
            self.output / identifier,
            self.realizations,
            self.seed,
        )
        self._evaluations[identifier] = {
            "run_id": identifier,
            "hypothesis": hypothesis.specification(),
            "reason": reason,
            "result": result,
        }
        pressure_check = "PASSED" if self.adequate(result) else "FAILED"
        logger.info(
            ProgressSection(
                "CANDIDATE RESULT",
                (
                    f"ID: {identifier}",
                    f"Conditioning RMSE: {result['conditioning_rmse_final_pa']:.2f} Pa",
                    f"Validation RMSE: {result['validation']['rmse_pa']:.2f} Pa",
                    f"Validation CRPS: {result['validation']['crps_pa']:.2f} Pa (lower is better)",
                    f"Pressure-error check: {pressure_check}",
                    f"Simulator calls: {executor.simulations}/{self.planned_calls}; "
                    f"elapsed time: {result['wall_seconds']:.2f}s",
                ),
            )
        )
        return self.public_evaluation(self._evaluations[identifier])

    def public_evaluation(self, evaluation: dict[str, object]) -> dict[str, object]:
        """Expose conditioning/validation scores and residuals without protected data."""
        result = evaluation["result"]
        public_result = {
            key: result[key]
            for key in (
                "conditioning_rmse_history_pa",
                "conditioning_rmse_final_pa",
                "validation",
                "parameters",
                "diagnostics",
                "simulator_calls",
                "wall_seconds",
            )
        }
        public_result["adequacy_gate_passed"] = self.adequate(result)
        return deepcopy(
            {
                "run_id": evaluation["run_id"],
                "hypothesis": evaluation["hypothesis"],
                "reason": evaluation["reason"],
                "result": public_result,
            }
        )

    def adequate(self, result: dict[str, object]) -> bool:
        """Require conditioning and validation RMSE <= twice the observation-error std.

        The 2x cutoff is a heuristic allowance for noise and model mismatch,
        not a statistically calibrated acceptance test.
        """
        return (
            result["conditioning_rmse_final_pa"] <= self.adequacy_threshold_pa
            and result["validation"]["rmse_pa"] <= self.adequacy_threshold_pa
        )

    def snapshot(self) -> dict[str, object]:
        """Return completed evaluations and spent/remaining budgets."""
        calls = len(self._model_calls)
        simulator_calls = sum(
            evaluation["result"]["simulator_calls"] for evaluation in self._evaluations.values()
        )
        return {
            "schema_version": 3,
            "closed": self._outcome is not None,
            "outcome": self._outcome,
            "reason": self._reason,
            "evaluations": [
                self.public_evaluation(evaluation) for evaluation in self._evaluations.values()
            ],
            "remaining_evaluations": self.max_evaluations - len(self._evaluations),
            "remaining_simulator_calls": self.max_simulations - simulator_calls,
            "charged_simulator_calls": simulator_calls,
            "model_calls": calls,
            "remaining_model_calls": self.max_model_calls - calls,
        }

    def record_model_call(self, call_id: str, provider: str, model: str) -> None:
        """Record and count a model request before host dispatch."""
        self._model_calls.append({"provider": provider, "model": model})
        logger.info(
            ProgressSection(
                f"LLM REQUEST {len(self._model_calls)}/{self.max_model_calls}",
                (
                    f"Model: {provider} / {model}",
                    "Awaiting decision: propose a hypothesis or finish search",
                    f"Call: {call_id}",
                ),
            )
        )

    def public_evidence(self) -> dict[str, object]:
        """Expose controls and conditioning observations, not raw validation/final data or truth.

        Public means agent-visible. Validation scores arrive through public_evaluation.
        Gauges are colocated with wells, and observations follow well order.
        """
        values = self.conditioning["values_pa"].reshape(-1, len(self.case.gauge_ids))
        return {
            "schema_version": 2,
            "case_id": self.case.case_id,
            "physics": self.case.model_dump(mode="json"),
            "boundary_condition": self.case.boundary_condition,
            "gauge_ids": list(self.case.gauge_ids),
            "well_positions_fraction": self.case.well_positions.tolist(),
            "well_order": list(self.case.gauge_ids),
            "observation_order": "time, then well",
            "rates_m3_s": self.case.rates.tolist(),
            "times_hours": (self.case.observation_times_s[: values.shape[0]] / 3600).tolist(),
            "observed_pressure_change_bar": self.case.pressure_change_bar(values).tolist(),
            "hypothesis_grammar": {
                "kinds": ["homogeneous", "band"],
                "orientations": ["vertical", "horizontal"],
                "band_width_bounds": list(band_width_bounds(self.case)),
                "geometry_rule": (
                    "center_fraction and width_fraction are fractions of the domain length; "
                    "vertical bands span all y, horizontal bands all x; "
                    "band inside domain, at least two grid rows/columns, background on both sides"
                ),
                "conditioned_parameters": [
                    "background permeability (mD)",
                    "band/background permeability ratio, when a band is present",
                ],
                "fixed_inference": (
                    f"{self.realizations} ensemble realizations; "
                    f"{NUMBER_OF_ITERATIONS} equal-inflation ES-MDA updates; "
                    "fixed priors and independent Gaussian observation error"
                ),
            },
            "adequacy_rule": (
                "conditioning and validation RMSE <= 2 observation standard deviations; "
                "prototype error check, not proof of geological truth"
            ),
        }

    def finish(self, outcome: str, reason: str) -> dict[str, object]:
        """Freeze search before protected grading and validate adequate claims."""
        records = [
            {**evaluation["result"], "candidate_id": identifier}
            for identifier, evaluation in self._evaluations.items()
        ]
        ranked = sorted(records, key=lambda record: record["validation"]["crps_pa"])
        if outcome == "adequate" and not self.adequate(ranked[0]):
            raise ValueError(
                "best evaluated hypothesis fails the pressure-error check: "
                "fitting and validation RMSE must both be <= "
                "twice the measurement-noise standard deviation"
            )
        models = dict.fromkeys((call["provider"], call["model"]) for call in self._model_calls)
        self._outcome, self._reason = outcome, reason
        state = self.snapshot()
        selection_path = self.output / "selection.json"
        selection = {
            "schema_version": 2,
            "status": "succeeded",
            "outcome": outcome,
            "reason": reason,
            "candidates": records,
            "selected_candidate_id": ranked[0]["candidate_id"],
            "selected_family": ranked[0]["family"],
            "validation_ranking": [record["candidate_id"] for record in ranked],
            "selection_rule": (
                "lowest validation CRPS among tested hypotheses; adequacy checked separately"
            ),
            "simulator_calls": sum(record["simulator_calls"] for record in records),
            "charged_simulator_calls": state["charged_simulator_calls"],
            "wall_seconds": sum(record.get("wall_seconds", 0) for record in records),
            "final_evaluation_accessed": False,
            "model_calls": state["model_calls"],
            "proposer_models": [
                {"provider": provider, "model": model} for provider, model in models
            ],
            "proposer_mode": "live_llm",
        }
        write_json(selection_path, selection)
        write_json(
            self.output / "run.json",
            {
                "schema_version": 2,
                "evidence_sha256": checksum(self.evidence / "evidence.json"),
                "selection_sha256": checksum(selection_path),
                "realizations": self.realizations,
                "seed": self.seed,
                "max_simulations": self.max_simulations,
                "max_evaluations": self.max_evaluations,
                "max_model_calls": self.max_model_calls,
                "adequacy_rule": (
                    "conditioning and validation RMSE <= 2 observation standard deviations"
                ),
                "mode": "hypothesis_search",
            },
        )
        logger.info(
            ProgressSection(
                "SEARCH DECISION",
                (
                    f"Search frozen; outcome: {outcome}",
                    f"Selected: {ranked[0]['candidate_id']}",
                    f"Completed candidates: {len(records)}",
                ),
                rationale=reason,
            )
        )
        for position, record in enumerate(ranked, start=1):
            logger.info(
                "Validation rank %d: %s (%s); CRPS=%.2f Pa (lower is better); "
                "pressure-error check=%s",
                position,
                record["candidate_id"],
                record["family"],
                record["validation"]["crps_pa"],
                "passed" if self.adequate(record) else "failed",
            )
        logger.info(
            "Search usage: model requests=%d/%d; charged simulator calls=%d/%d",
            state["model_calls"],
            self.max_model_calls,
            state["charged_simulator_calls"],
            self.max_simulations,
        )
        logger.info("Frozen ranking saved: %s", selection_path)
        return state
