"""
Jal-Rakshak — Alert Manager & Notification System
====================================================
Multi-channel alert system with simulation mode by default.
Generates actionable, community-appropriate alerts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger("jal_rakshak.alerts")


class AlertChannel(Enum):
    DASHBOARD = "dashboard"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    CONSOLE = "console"


class AlertPriority(Enum):
    INFO = "info"
    STANDARD = "standard"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class AlertTarget(Enum):
    AUTHORITIES = "authorities"
    COAST_GUARD = "coast_guard"
    PORT_AUTHORITY = "port_authority"
    FISHERMEN = "fishermen"
    COASTAL_RESIDENTS = "coastal_residents"
    ENVIRONMENTAL_AGENCY = "environmental_agency"


@dataclass
class Alert:
    """A single alert instance."""
    alert_id: str
    timestamp: datetime
    priority: AlertPriority
    target: AlertTarget
    channels: List[AlertChannel]
    title: str
    message: str
    location_description: str
    predicted_impact_zone: Optional[str] = None
    eta_hours: Optional[float] = None
    risk_level: Optional[str] = None
    recommended_action: str = ""
    data_status: str = "PREDICTED / ESTIMATED"
    is_simulated: bool = True
    delivered: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "target": self.target.value,
            "channels": [c.value for c in self.channels],
            "title": self.title,
            "message": self.message,
            "location": self.location_description,
            "predicted_impact_zone": self.predicted_impact_zone,
            "eta_hours": self.eta_hours,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "data_status": self.data_status,
            "is_simulated": self.is_simulated,
            "delivered": self.delivered,
        }

    def format_community_alert(self) -> str:
        """Format alert for community consumption."""
        parts = [
            "═" * 50,
            "⚠️  MARINE POLLUTION ALERT",
            "═" * 50,
            "",
            f"Potential oil spill detected near:",
            f"  📍 {self.location_description}",
            "",
        ]
        if self.predicted_impact_zone:
            parts.append(f"Predicted impact zone:")
            parts.append(f"  🗺️ {self.predicted_impact_zone}")
            parts.append("")
        if self.eta_hours is not None:
            parts.append(f"Estimated arrival:")
            parts.append(f"  ⏱️ {self.eta_hours:.1f} hours")
            parts.append("")
        if self.risk_level:
            parts.append(f"Risk level:")
            parts.append(f"  🔴 {self.risk_level}")
            parts.append("")
        if self.recommended_action:
            parts.append(f"Recommended action:")
            parts.append(f"  ℹ️ {self.recommended_action}")
            parts.append("")
        parts.extend([
            f"Last updated: {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Information status: {self.data_status}",
            f"Source: Jal-Rakshak Maritime Intelligence",
            "",
            "═" * 50,
        ])
        if self.is_simulated:
            parts.append("🔶 THIS IS A SIMULATED ALERT — NOT A REAL EMERGENCY")
        return "\n".join(parts)


@dataclass
class AlertLog:
    """Log of all alerts generated for an incident."""
    incident_id: str
    alerts: List[Alert] = field(default_factory=list)

    def add(self, alert: Alert):
        self.alerts.append(alert)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "total_alerts": len(self.alerts),
            "alerts": [a.to_dict() for a in self.alerts],
        }


class AlertManager:
    """
    Manages alert generation and delivery.

    DEFAULT MODE: SIMULATION — no real alerts are sent.
    Real delivery requires explicit configuration and credentials.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.alert_log = AlertLog(incident_id="")

    def set_incident(self, incident_id: str):
        self.alert_log = AlertLog(incident_id=incident_id)

    def generate_community_alert(self,
                                  location: str,
                                  risk_level: str,
                                  lat: float, lon: float,
                                  impact_zone: str = "",
                                  eta_hours: float = None,
                                  recommended_action: str = "",
                                  ) -> Alert:
        """Generate an alert for community stakeholders."""
        alert = Alert(
            alert_id=f"COMM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            priority=AlertPriority.URGENT if risk_level in ("HIGH", "CRITICAL") else AlertPriority.STANDARD,
            target=AlertTarget.COASTAL_RESIDENTS,
            channels=[AlertChannel.DASHBOARD, AlertChannel.SMS],
            title="Marine Pollution Alert",
            message=f"Potential oil spill detected near {location}. "
                    f"Risk level: {risk_level}. "
                    f"{'Avoid fishing/boating in the affected zone.' if risk_level in ('HIGH', 'CRITICAL') else 'Monitor for updates.'}",
            location_description=location,
            predicted_impact_zone=impact_zone,
            eta_hours=eta_hours,
            risk_level=risk_level,
            recommended_action=recommended_action or "Monitor official channels for updates",
            is_simulated=self.simulation_mode,
        )
        self.alert_log.add(alert)
        self._deliver(alert)
        return alert

    def generate_authority_alert(self,
                                  location: str,
                                  risk_level: str,
                                  spill_area_sq_km: float,
                                  candidate_vessels: List[str] = None,
                                  ) -> Alert:
        """Generate an alert for authorities."""
        vessel_info = ", ".join(candidate_vessels[:3]) if candidate_vessels else "under investigation"
        alert = Alert(
            alert_id=f"AUTH-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            priority=AlertPriority.IMMEDIATE if risk_level == "CRITICAL" else AlertPriority.URGENT,
            target=AlertTarget.AUTHORITIES,
            channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL],
            title="Oil Spill Incident Report",
            message=f"Detected spill: {spill_area_sq_km:.3f} km². "
                    f"Risk: {risk_level}. "
                    f"Candidate vessels: {vessel_info}.",
            location_description=location,
            risk_level=risk_level,
            recommended_action="Initiate response protocol and investigation",
            is_simulated=self.simulation_mode,
        )
        self.alert_log.add(alert)
        self._deliver(alert)
        return alert

    def _deliver(self, alert: Alert):
        """Deliver alert through configured channels."""
        if self.simulation_mode:
            provider = SimulationNotificationProvider()
            alert.delivered = provider.send(alert)
            return

        for channel in alert.channels:
            provider = get_notification_provider(channel)
            success = provider.send(alert)
            if success:
                alert.delivered = True


class NotificationProvider(ABC):
    """Abstract interface for external alert dispatch."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver alert and return True if successful."""
        ...


class SimulationNotificationProvider(NotificationProvider):
    """Logs simulated dispatch without external network calls."""

    def send(self, alert: Alert) -> bool:
        logger.info(f"[SIMULATION ALERT] ID: {alert.alert_id} | Target: {alert.target.value} | Priority: {alert.priority.value}")
        return True


class DashboardNotificationProvider(NotificationProvider):
    """In-app dashboard delivery."""

    def send(self, alert: Alert) -> bool:
        return True


class ConsoleNotificationProvider(NotificationProvider):
    """Prints formatted ASCII alert banner to console."""

    def send(self, alert: Alert) -> bool:
        print(alert.format_community_alert())
        return True


class EmailNotificationProvider(NotificationProvider):
    """SMTP email notification dispatch."""

    def send(self, alert: Alert) -> bool:
        logger.warning(f"Email delivery unconfigured (SMTP credentials not set) — alert {alert.alert_id} queued.")
        return False


class SMSNotificationProvider(NotificationProvider):
    """Cellular SMS gateway notification dispatch."""

    def send(self, alert: Alert) -> bool:
        logger.warning(f"SMS delivery unconfigured (SMS gateway not set) — alert {alert.alert_id} queued.")
        return False


def get_notification_provider(channel: AlertChannel) -> NotificationProvider:
    """Factory returning provider for requested alert channel."""
    if channel == AlertChannel.DASHBOARD:
        return DashboardNotificationProvider()
    elif channel == AlertChannel.CONSOLE:
        return ConsoleNotificationProvider()
    elif channel == AlertChannel.EMAIL:
        return EmailNotificationProvider()
    elif channel == AlertChannel.SMS:
        return SMSNotificationProvider()
    return SimulationNotificationProvider()
