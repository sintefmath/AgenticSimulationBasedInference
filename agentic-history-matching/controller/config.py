"""Load an operator-only YAML config for the fixed-truth LLM experiment."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
)
from yaml.nodes import MappingNode


class RunConfig(BaseModel):
    """Configure the model, inference, and budgets without changing truth or observations."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    output: Path
    model: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    realizations: int = Field(default=50, strict=True, ge=2)
    seed: int = Field(default=7, strict=True, ge=0)
    max_evaluations: int = Field(default=3, strict=True, gt=0)
    max_model_calls: int = Field(default=6, strict=True, gt=0)

    @field_validator("output", mode="before")
    @classmethod
    def _nonempty_output(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("output must be a non-empty path")
        return value

    @field_validator("output")
    @classmethod
    def _resolve_output(cls, value: Path, info: ValidationInfo) -> Path:
        return (info.context["config_directory"] / value).resolve()


def load_config(path: Path) -> RunConfig:
    """Validate YAML settings and resolve output relative to the config file."""
    with path.open(encoding="utf-8") as stream:
        try:
            value: object = yaml.load(stream, Loader=_ConfigLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML: {error}") from error
    return RunConfig.model_validate(value, context={"config_directory": path.resolve().parent})


class _ConfigLoader(yaml.SafeLoader):
    """Use safe YAML loading while rejecting duplicate config keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, object]:
        if not isinstance(node, MappingNode):
            raise ValueError("config must be a mapping with string keys")
        keys: set[str] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("config keys must be strings")
            if key in keys:
                raise ValueError(f"duplicate config key: {key}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)
