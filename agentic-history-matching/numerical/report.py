"""Render protected post-selection diagnostics with a headless Matplotlib backend."""

from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from cases.pressure import PressureCase
from numerical.geology import MILLIDARCY_M2, BandGeometry, Family


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Describe one completed evaluation and its fitted posterior."""

    candidate_id: str
    family: Family
    geometry: BandGeometry | None
    permeability_m2: NDArray[np.float64]
    predictions_pa: NDArray[np.float64]
    validation_crps_pa: float


@dataclass(frozen=True)
class _PressureSummary:
    mean_bar: NDArray[np.float64]
    lower_bar: NDArray[np.float64]
    upper_bar: NDArray[np.float64]


# Bound page width without dropping candidates or shrinking their panels.
CANDIDATES_PER_PAGE = 4


def diagnostic_pngs(
    case: PressureCase,
    truth_field_m2: NDArray[np.float64],
    clean_responses_pa: NDArray[np.float64],
    observations_pa: NDArray[np.float64],
    candidates: list[CandidateDiagnostics],
    selected_candidate_id: str,
) -> Iterator[bytes]:
    """Render every evaluation in search order, sharing field and per-gauge scales across pages."""
    truth_field = np.log10(truth_field_m2 / MILLIDARCY_M2)
    fields: dict[str, NDArray[np.float64]] = {}
    summaries: dict[str, _PressureSummary] = {}
    times_h = case.observation_times_s / 3600
    shape = (times_h.size, len(case.gauge_ids))
    truth = (clean_responses_pa.reshape(shape) - case.initial_pressure_pa) / 1e5
    observed = (observations_pa.reshape(shape) - case.initial_pressure_pa) / 1e5
    vmin, vmax = float(truth_field.min()), float(truth_field.max())
    pressure_min = np.minimum(truth.min(axis=0), observed.min(axis=0))
    pressure_max = np.maximum(truth.max(axis=0), observed.max(axis=0))
    for candidate in candidates:
        field = np.log10(candidate.permeability_m2 / MILLIDARCY_M2)
        fields[candidate.candidate_id] = field
        vmin, vmax = min(vmin, float(field.min())), max(vmax, float(field.max()))
        predictions = (
            candidate.predictions_pa.reshape(*shape, candidate.predictions_pa.shape[1])
            - case.initial_pressure_pa
        ) / 1e5
        lower, upper = np.quantile(predictions, [0.05, 0.95], axis=2)
        mean = predictions.mean(axis=2)
        summaries[candidate.candidate_id] = _PressureSummary(mean, lower, upper)
        pressure_min = np.minimum(pressure_min, np.minimum(lower.min(axis=0), mean.min(axis=0)))
        pressure_max = np.maximum(pressure_max, np.maximum(upper.max(axis=0), mean.max(axis=0)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    margin = np.maximum((pressure_max - pressure_min) * 0.05, case.observation_std_pa / 1e5)
    page_count = (len(candidates) + CANDIDATES_PER_PAGE - 1) // CANDIDATES_PER_PAGE
    for start in range(0, len(candidates), CANDIDATES_PER_PAGE):
        page = candidates[start : start + CANDIDATES_PER_PAGE]
        columns = len(page) + 1
        figure = Figure(figsize=(4.4 * columns, 15), layout="constrained")
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots(len(case.gauge_ids) + 1, columns, squeeze=False, sharey="row")
        image = _plot_field(axes[0, 0], case, truth_field, norm)
        axes[0, 0].set_title("Hidden truth\nReference observations", fontsize=10)
        for column, candidate in enumerate(page, start=1):
            axis = axes[0, column]
            axis.set_title(
                _candidate_title(candidate, start + column, selected_candidate_id),
                fontsize=10,
                color="tab:green" if candidate.candidate_id == selected_candidate_id else "black",
            )
            _plot_field(axis, case, fields[candidate.candidate_id], norm)
        figure.colorbar(
            image,
            ax=list(axes[0]),
            location="bottom",
            shrink=0.8,
            label="log10(permeability / mD)",
        )
        legend: dict[str, Artist] = {}
        for gauge, gauge_id in enumerate(case.gauge_ids):
            for column, candidate in enumerate((None, *page)):
                axis = axes[gauge + 1, column]
                axis.set(
                    title=gauge_id.replace("_", " "),
                    xlim=(0, case.final_end_s / 3600),
                    ylim=(pressure_min[gauge] - margin[gauge], pressure_max[gauge] + margin[gauge]),
                    xlabel="time (hours)" if gauge == len(case.gauge_ids) - 1 else "",
                    ylabel="pressure change (bar)" if column == 0 else "",
                )
                summary = summaries[candidate.candidate_id] if candidate is not None else None
                _plot_pressure(axis, case, times_h, truth, observed, gauge, summary)
                if gauge == 0:
                    handles, labels = axis.get_legend_handles_labels()
                    legend.update(zip(labels, handles, strict=True))
        figure.legend(
            legend.values(),
            legend.keys(),
            loc="outside lower center",
            ncol=3,
            fontsize="small",
        )
        page_number = start // CANDIDATES_PER_PAGE + 1
        figure.suptitle(
            "Model-search evolution: permeability and pressure forecasts "
            f"(page {page_number}/{page_count})\n"
            "Geometric-mean fitted fields; validation CRPS in Pa (lower is better)\n"
            "Conditioning | validation | protected final"
        )
        output = BytesIO()
        canvas.print_png(output)
        yield output.getvalue()


def _plot_field(
    axis: Axes, case: PressureCase, field: NDArray[np.float64], norm: Normalize
) -> AxesImage:
    image = axis.imshow(
        field,
        origin="lower",
        extent=(0, case.length_m, 0, case.length_m),
        norm=norm,
        cmap="viridis",
    )
    gauges = case.well_positions * case.length_m
    axis.scatter(gauges[:, 0], gauges[:, 1], marker="x", color="white", label="gauges")
    axis.set(xlabel="x (m)", ylabel="y (m)")
    return image


def _candidate_title(candidate: CandidateDiagnostics, number: int, selected_id: str) -> str:
    title = f"{number}: {candidate.family.value.replace('_', ' ')}"
    if number == 1:
        title += " (baseline)"
    if candidate.candidate_id == selected_id:
        title += " [selected]"
    title += f"\n{candidate.candidate_id}"
    if candidate.geometry is not None:
        title += (
            f"\nCenter {candidate.geometry.center_fraction:.1%}, "
            f"width {candidate.geometry.width_fraction:.1%}"
        )
    title += f"\nValidation CRPS: {candidate.validation_crps_pa:,.2f} Pa"
    return title


def _plot_pressure(
    axis: Axes,
    case: PressureCase,
    times_h: NDArray[np.float64],
    truth: NDArray[np.float64],
    observed: NDArray[np.float64],
    gauge: int,
    summary: _PressureSummary | None,
) -> None:
    if summary is not None:
        axis.fill_between(
            times_h,
            summary.lower_bar[:, gauge],
            summary.upper_bar[:, gauge],
            alpha=0.25,
            color="tab:blue",
            label="90% predictive interval",
        )
        axis.plot(times_h, summary.mean_bar[:, gauge], color="tab:blue", label="predictive mean")
    axis.plot(times_h, truth[:, gauge], color="black", linestyle="--", label="noise-free truth")
    axis.scatter(times_h, observed[:, gauge], s=10, color="tab:orange", label="observations")
    for time_s, label, style in (
        (case.conditioning_end_s, "conditioning ends", ":"),
        (case.validation_end_s, "validation ends", "-."),
    ):
        axis.axvline(time_s / 3600, color="gray", linestyle=style, label=label)
