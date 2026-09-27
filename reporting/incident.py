"""
Jal-Rakshak — Incident Report Generator
==========================================
Generates comprehensive investigation reports.
Clearly distinguishes OBSERVED / INFERRED / PREDICTED / SIMULATED.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import json


class DataClassification:
    """Labels for data provenance."""
    OBSERVED = "OBSERVED"       # Directly measured/detected
    INFERRED = "INFERRED"       # Derived from analysis
    PREDICTED = "PREDICTED"     # Forecasted/estimated
    SIMULATED = "SIMULATED"     # Demo/synthetic data


@dataclass
class ReportSection:
    """A section of the incident report."""
    title: str
    content: Dict[str, Any]
    data_classification: str  # One of DataClassification values

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "data_classification": self.data_classification,
            "content": self.content,
        }


@dataclass
class IncidentReport:
    """Complete investigation report for an oil spill incident."""
    incident_id: str
    generated_at: datetime
    sections: List[ReportSection] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)

    def add_section(self, title: str, content: Dict, classification: str):
        self.sections.append(ReportSection(title, content, classification))

    def add_limitation(self, limitation: str):
        self.limitations.append(limitation)

    def add_data_source(self, source: str):
        if source not in self.data_sources:
            self.data_sources.append(source)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "generated_at": self.generated_at.isoformat(),
            "sections": [s.to_dict() for s in self.sections],
            "limitations": self.limitations,
            "data_sources": self.data_sources,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def generate_report(
    incident_id: str,
    detection_result: Dict = None,
    characterization: Dict = None,
    validation_result: Dict = None,
    hindcast_result: Dict = None,
    vessel_scores: List[Dict] = None,
    forecast_results: List[Dict] = None,
    risk_assessment: Dict = None,
    alert_log: Dict = None,
    ocean_data: Dict = None,
    is_demo: bool = False,
) -> IncidentReport:
    """
    Generate a comprehensive incident report from pipeline results.

    Each section is classified by data provenance.
    """
    report = IncidentReport(
        incident_id=incident_id,
        generated_at=datetime.now(timezone.utc),
    )

    # 1. Detection
    if detection_result:
        report.add_section(
            "Detection Summary",
            detection_result,
            DataClassification.SIMULATED if is_demo else DataClassification.OBSERVED,
        )
        report.add_data_source("YOLO v8 Segmentation Model")

    # 2. Spill Characterization
    if characterization:
        report.add_section(
            "Spill Characterization",
            characterization,
            DataClassification.SIMULATED if is_demo else DataClassification.OBSERVED,
        )

    # 3. Validation
    if validation_result:
        report.add_section(
            "Detection Validation",
            validation_result,
            DataClassification.INFERRED,
        )
        report.add_data_source("Classical SAR Analysis")

    # 4. Hindcast
    if hindcast_result:
        report.add_section(
            "Source Reconstruction (Hindcast)",
            hindcast_result,
            DataClassification.PREDICTED,
        )
        report.add_limitation(
            "Hindcast trajectory assumes constant current/wind fields. "
            "Real conditions may vary significantly over the drift period."
        )

    # 5. Vessel Attribution
    if vessel_scores:
        report.add_section(
            "Candidate Vessel Analysis",
            {"candidates": vessel_scores, "disclaimer":
             "These are CANDIDATE vessels with ASSOCIATION SCORES. "
             "Scores reflect spatial/temporal/trajectory consistency only "
             "and do NOT constitute proof of liability."},
            DataClassification.INFERRED,
        )
        report.add_limitation(
            "Vessel attribution is probabilistic and based on available AIS data. "
            "AIS data gaps, spoofing, or incomplete coverage may affect results."
        )

    # 6. Forecast
    if forecast_results:
        report.add_section(
            "Drift Prediction (Forecast)",
            {"forecasts": forecast_results},
            DataClassification.PREDICTED,
        )
        report.add_limitation(
            "Drift predictions are estimates based on current environmental conditions. "
            "Actual movement will depend on changing ocean and weather conditions."
        )

    # 7. Risk Assessment
    if risk_assessment:
        report.add_section(
            "Risk Assessment",
            risk_assessment,
            DataClassification.INFERRED,
        )

    # 8. Ocean/Weather
    if ocean_data:
        data_class = DataClassification.SIMULATED if is_demo else DataClassification.OBSERVED
        report.add_section("Ocean & Weather Conditions", ocean_data, data_class)

    # 9. Alerts
    if alert_log:
        report.add_section("Alerts Generated", alert_log, DataClassification.OBSERVED)

    # Standard limitations
    report.add_limitation(
        "This report is generated by an AI system for decision support. "
        "It does not independently establish legal liability."
    )
    if is_demo:
        report.add_limitation(
            "This report was generated using SIMULATED/DEMO data. "
            "All values are synthetic and for demonstration purposes only."
        )
        report.add_data_source("Demo/Simulated Data")

    return report
