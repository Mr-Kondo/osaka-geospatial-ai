"""Validated, versioned interfaces. Missing AI observations are never synthesized."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VisualPattern(StrictModel):
    region: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    confidence: float = Field(
        ge=0, le=1, description="Model self-assessment, not a calibrated probability"
    )


class CrossMapRelationship(StrictModel):
    maps: list[str] = Field(min_length=2)
    observation: str = Field(min_length=1)


class VisualFindings(StrictModel):
    visual_patterns: list[VisualPattern]
    anomalies: list[VisualPattern]
    cross_map_relationships: list[CrossMapRelationship]
    limitations: list[str]


class VLMAnalysis(VisualFindings):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["completed", "disabled", "unavailable", "failed"]
    provider: str
    model: str
    input_images: list[dict[str, str]]
    error: str | None = None

    @model_validator(mode="after")
    def no_invented_findings(self):
        if self.status != "completed" and (
            self.visual_patterns or self.anomalies or self.cross_map_relationships
        ):
            raise ValueError("An unexecuted/failed VLM must have no visual findings")
        return self


class Metrics(StrictModel):
    n: int = Field(gt=0)
    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    r2: float | None
    direction_accuracy: float = Field(ge=0, le=1)


class ModelSummary(StrictModel):
    name: str
    target: str
    test_target_year: int
    metrics: Metrics
    selection: str
    validation_metrics: dict[str, Metrics]
    test_comparison: dict[str, Metrics]


class AnalysisReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    metadata: dict
    dataset_summary: dict
    observed_statistics: dict
    spatial_analysis: dict
    model: ModelSummary
    prediction_summary: dict
    residual_analysis: dict
    vlm_analysis: VLMAnalysis
    limitations: list[str]
    sources: list[dict[str, str]]
