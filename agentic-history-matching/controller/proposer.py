"""Initialize the baseline and host a bounded Pydantic AI search through SearchController."""

import json
from dataclasses import dataclass
from typing import Annotated, Literal, TextIO
from uuid import uuid4

from pydantic import AfterValidator, Field, StrictStr
from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
    TextOutput,
    Tool,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from controller.hypotheses import BandProposal, HomogeneousProposal
from controller.progress import logger
from controller.search import ProposalRejected, SearchController

SYSTEM_PROMPT = """You propose geological hypotheses for a pressure-diffusion inverse problem.
Permeability measures how easily fluid flows; the initial hypothesis is homogeneous, with the same permeability in every cell.
It may be adequate, inadequate, or uncertain.
"Public evidence" means information released to you, not information published online.
You receive conditioning observations with sample times, well locations, and known well controls.
Pressure gauges are colocated with wells; observations follow the listed well order. Positive rates inject fluid; negative rates produce it.
Conditioning observations fit parameters; validation observations only score forecasts.
You receive validation scores, not their raw readings. Protected final observations and generating truth stay hidden.
A signed residual is (mean simulated pressure - measured pressure) / measurement-noise standard deviation.
Positive residuals indicate overprediction; negative residuals indicate underprediction. Pressure changes use bar; permeability summaries use mD.
Use observations, known controls, and signed residuals to explain mismatches before proposing a revision.
The reviewed hypothesis grammar permits homogeneous permeability or one low-permeability band.
Propose orientation, center_fraction, and width_fraction. Vertical bands span all y; horizontal bands span all x.
Center and width are domain fractions along x for vertical bands or y for horizontal bands. Follow width bounds and geometry_rule.
Geometry stays fixed during conditioning. The controller samples an ensemble from fixed priors and runs equal-inflation
ES-MDA updates (Ensemble Smoother with Multiple Data Assimilation), fitting background permeability and contrast (band/background ratio).
Do not change solver, priors, likelihood, observation error, conditioning cutoffs, or scoring.
Validation ranks predictive ensembles by CRPS (continuous ranked probability score); lower is better.
The adequacy gate requires both conditioning and validation RMSE <= 2 declared observation standard deviations.
Adequacy is not proof of geological truth; retain uncertainty when evidence is insufficient.
You have no filesystem, shell, Python, protected observations, truth labels, or truth-generation code.
Request exactly one action per turn: evaluate_hypothesis or finish_search. Explain the evidence for it.
Avoid equivalent repeats with the same orientation and covered cells. Use remaining budgets and adequacy diagnostics to decide when to stop.
If no supported hypothesis is adequate, finish as inadequate or uncertain rather than declaring the least-bad model true."""  # noqa: E501

MAX_OUTPUT_TOKENS = 1024


def _nonempty_reason(value: str) -> str:
    if not value.strip():
        raise ValueError("reason must be a non-empty string")
    return value


type Reason = Annotated[StrictStr, Field(min_length=1), AfterValidator(_nonempty_reason)]


@dataclass
class _ProposerDependencies:
    search: SearchController
    transcript: TextIO
    turns: int = 0
    model_call_id: str = ""
    action: ToolCallPart | None = None


def run_agent_search(search: SearchController, model: str) -> None:
    """Evaluate the baseline, then run bounded model turns and evaluations sequentially."""
    search.evaluate({"kind": "homogeneous"}, "Predeclared initial homogeneous hypothesis")
    directory = search.output / "proposer"
    directory.mkdir(exist_ok=False)
    stop_reason, stop_detail = "finished", "Model froze the search decision"
    with (
        (directory / "messages.jsonl").open("x", encoding="utf-8") as transcript,
        capture_run_messages() as history,
    ):
        deps = _ProposerDependencies(search, transcript)
        try:
            agent = Agent[_ProposerDependencies, dict[str, object]](
                model,
                # Let Pydantic AI create the HTTP client during the run and close it afterward.
                defer_model_check=True,
                deps_type=_ProposerDependencies,
                instructions=SYSTEM_PROMPT,
                capabilities=[
                    Hooks(
                        before_run=_record_session,
                        before_model_request=_prepare_turn,
                        prepare_model_request=_record_request,
                        after_model_request=_record_response,
                    )
                ],
                tools=[
                    Tool(
                        evaluate_hypothesis,
                        sequential=True,
                        strict=True,
                        description=(
                            "Validate and evaluate one reviewed geological hypothesis with "
                            "controller-owned conditioning and validation. "
                            "Return completed scores before the next turn."
                        ),
                    )
                ],
                output_type=[
                    ToolOutput(
                        finish_search,
                        name="finish_search",
                        strict=True,
                    ),
                    # Keep tool_choice="auto", but retry text-only replies so that
                    # only a successful finish_search call can provide the final result.
                    TextOutput(_require_action),
                ],
                model_settings=_model_settings,
                # Validation corrections remain possible, but every request is counted first.
                retries={"tools": search.max_model_calls, "output": search.max_model_calls},
                end_strategy="early",
            )
            agent.instrument = False
            logger.info("Configuring model provider for %s", model)
            agent.run_sync(
                "Public evidence and completed evaluations:\n"
                + json.dumps(
                    {
                        "evidence": search.public_evidence(),
                        "evaluations": search.snapshot()["evaluations"],
                    },
                    allow_nan=False,
                ),
                deps=deps,
                usage_limits=UsageLimits(request_limit=search.max_model_calls),
            )
        except UsageLimitExceeded:
            stop_reason, stop_detail = "request_limit", "Proposer reached its model-request limit"
        except ModelAPIError as error:
            raise RuntimeError(_provider_error(error)) from None
        except UnexpectedModelBehavior:
            raise RuntimeError(
                "Model returned invalid output; raw provider error omitted"
            ) from None
        finally:
            _record(transcript, "messages", messages=_audit_messages(history))
        snapshot = search.snapshot()
        if not snapshot["closed"]:
            snapshot = search.finish("uncertain", "Host stopped at its model-request limit")
        _record(transcript, "stop", reason=stop_reason, detail=stop_detail)
        _record(transcript, "search", search=snapshot)
    if stop_reason != "finished":
        logger.warning("Proposer stopped: %s", stop_detail)


def _model_settings(ctx: RunContext[_ProposerDependencies]) -> ModelSettings:
    settings: ModelSettings = {
        "max_tokens": MAX_OUTPUT_TOKENS,
        "max_retries": 0,
        "timeout": 60.0,
        "request_timeout": 60.0,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    if isinstance(ctx.model, OpenAIResponsesModel):
        return OpenAIResponsesModelSettings(
            **settings, openai_store=False, openai_truncation="disabled"
        )
    return settings


async def _record_session(ctx: RunContext[_ProposerDependencies]) -> None:
    model = ctx.model
    _record(
        ctx.deps.transcript,
        "session",
        provider=model.system,
        model=model.model_name,
        session_id=str(uuid4()),
        model_settings=_model_settings(ctx),
        model_metadata={
            "configured_model": model.model_id,
            "context_limit": model.context_window,
        },
        instructions=SYSTEM_PROMPT,
    )
    logger.info("Model provider ready")


async def _prepare_turn(
    ctx: RunContext[_ProposerDependencies], request: ModelRequestContext
) -> ModelRequestContext:
    snapshot = ctx.deps.search.snapshot()
    # Evidence and completed results are already in history; append only current status.
    status = {key: value for key, value in snapshot.items() if key != "evaluations"}
    ctx.enqueue(
        "Current search status:\n"
        + json.dumps(status, allow_nan=False)
        + "\nChoose one tool action and explain why. "
        "Truth and protected final evaluation remain unavailable."
    )
    return request


async def _record_request(
    ctx: RunContext[_ProposerDependencies], request: ModelRequestContext
) -> ModelRequestContext:
    """Account for the prepared request and record metadata without copying its history."""
    deps = ctx.deps
    deps.model_call_id = "mc_" + uuid4().hex
    deps.search.record_model_call(
        call_id=deps.model_call_id,
        provider=request.model.system,
        model=request.model.model_name,
    )
    deps.turns += 1
    _record(
        deps.transcript,
        "prompt",
        turn=deps.turns,
        model_call_id=deps.model_call_id,
        tools=[
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_json_schema,
            }
            for tool in request.model_request_parameters.tool_defs.values()
        ],
    )
    return request


async def _record_response(
    ctx: RunContext[_ProposerDependencies],
    *,
    request_context: ModelRequestContext,  # Required by the hook's keyword-based signature.
    response: ModelResponse,
) -> ModelResponse:
    deps = ctx.deps
    usage = _token_usage(response)
    if response.state != "complete" or response.finish_reason not in ("stop", "tool_call"):
        raise UnexpectedModelBehavior("Model response failed or was incomplete; no tools executed")
    calls = response.tool_calls
    deps.action = calls[0] if calls else None
    _record(
        deps.transcript,
        "assistant",
        turn=deps.turns,
        model_call_id=deps.model_call_id,
        response_id=response.provider_response_id,
        status=response.state,
        model=response.model_name,
        output=_audit_messages([response]),
        usage=usage,
        error=None,
    )
    return response


# Intentionally async to avoid Pydantic AI's worker-thread dispatch.
async def evaluate_hypothesis(
    ctx: RunContext[_ProposerDependencies],
    hypothesis: HomogeneousProposal | BandProposal,
    reason: Reason,
) -> dict[str, object]:
    """Validate and evaluate one hypothesis inline before the next model request."""
    _check_action(ctx)
    try:
        return ctx.deps.search.evaluate(hypothesis.model_dump(), reason)
    except ProposalRejected as error:
        raise ModelRetry(str(error)) from None


async def finish_search(
    ctx: RunContext[_ProposerDependencies],
    outcome: Literal["adequate", "inadequate", "uncertain"],
    reason: Reason,
) -> dict[str, object]:
    """Freeze selection before protected evaluation. The controller checks adequate claims against conditioning and validation error thresholds."""  # noqa: E501
    _check_action(ctx)
    try:
        return ctx.deps.search.finish(outcome, reason)
    except ValueError as error:
        raise ModelRetry(str(error)) from None


def _require_action(_text: str) -> dict[str, object]:
    raise ModelRetry("No tool action received. Choose evaluate_hypothesis or finish_search.")


def _check_action(ctx: RunContext[_ProposerDependencies]) -> None:
    # A provider may ignore parallel_tool_calls=False; only the first requested action is eligible.
    action = ctx.deps.action
    if (
        action is None
        or ctx.tool_name != action.tool_name
        or ctx.tool_call_id != action.tool_call_id
    ):
        raise ModelRetry("Only one tool action is accepted per model response")
    ctx.deps.action = None


def _token_usage(response: ModelResponse) -> dict[str, object] | None:
    """Record normalized token counts without provider billing fields or estimated prices."""
    usage = response.usage
    if usage.unmeasured_requests:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
        },
        "output_tokens_details": {"reasoning_tokens": usage.output_reasoning_tokens},
    }


def _provider_error(error: ModelAPIError) -> str:
    """Describe provider failure without response bodies, headers, or credentials."""
    if isinstance(error, ModelHTTPError):
        if error.status_code in (401, 403):
            return (
                "Model provider rejected access; check credentials, model access, and paid credits"
            )
        return f"Model request failed (HTTP {error.status_code}); raw provider error omitted"
    return "Model request failed; raw provider error omitted and no retry was attempted"


def _audit_messages(messages: list[ModelMessage]) -> object:
    # Keep normalized conversation data, not unreviewed billing or provider metadata.
    return json.loads(
        ModelMessagesTypeAdapter.dump_json(
            messages,
            exclude={"__all__": {"usage", "provider_details"}},
        )
    )


def _record(transcript: TextIO, event: str, **fields: object) -> None:
    transcript.write(
        json.dumps({"schema_version": 3, "event": event, **fields}, allow_nan=False) + "\n"
    )
    transcript.flush()
