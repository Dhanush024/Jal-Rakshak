"""
Jal-Rakshak — Coastal Impact Assessment Engine
================================================
Calculates shoreline proximity, landfall trajectory intersection, ETA,
Environmental Sensitivity Index (ESI) vulnerability, and response countermeasures.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

from geospatial.distance import haversine_km, bearing_deg, destination_point
from coastal.zones import (
    EnvironmentalSensitivityIndex,
    CoastalPoint,
    SensitiveArea,
    CHENNAI_CORRIDOR_SHORELINE,
    CHENNAI_SENSITIVE_AREAS,
    get_nearest_shoreline_point,
)


@dataclass
class ThreatAsset:
    """A sensitive coastal or infrastructure asset under threat."""
    asset: SensitiveArea
    distance_from_spill_km: float
    distance_from_drift_corridor_km: float
    threat_level: str               # "IMMINENT", "HIGH_RISK", "POTENTIAL", "MONITOR"
    estimated_arrival_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.asset.name,
            "category": self.asset.category,
            "esi": int(self.asset.esi),
            "priority": self.asset.priority,
            "distance_from_spill_km": round(self.distance_from_spill_km, 2),
            "distance_from_corridor_km": round(self.distance_from_drift_corridor_km, 2),
            "threat_level": self.threat_level,
            "estimated_arrival_hours": round(self.estimated_arrival_hours, 1) if self.estimated_arrival_hours is not None else None,
            "recommended_strategy": self.asset.recommended_strategy,
            "contact_authority": self.asset.contact_authority,
        }


@dataclass
class CoastalImpactResult:
    """Comprehensive result of coastal impact assessment."""
    spill_lat: float
    spill_lon: float
    nearest_shoreline_point: CoastalPoint
    shortest_distance_to_coast_km: float
    landfall_projected: bool
    landfall_point: Optional[Tuple[float, float]] = None
    landfall_shoreline_name: Optional[str] = None
    landfall_esi: Optional[int] = None
    eta_to_coast_hours: Optional[float] = None
    eta_uncertainty_range_hours: Optional[Tuple[float, float]] = None
    threatened_assets: List[ThreatAsset] = field(default_factory=list)
    coastal_vulnerability_score: float = 0.0     # 0 - 100
    risk_tier: str = "LOW"                       # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    containment_recommendations: List[str] = field(default_factory=list)
    dispersant_restrictions: List[str] = field(default_factory=list)
    assessment_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "spill_lat": round(self.spill_lat, 4),
            "spill_lon": round(self.spill_lon, 4),
            "nearest_shoreline_point": self.nearest_shoreline_point.to_dict(),
            "shortest_distance_to_coast_km": round(self.shortest_distance_to_coast_km, 2),
            "landfall_projected": self.landfall_projected,
            "landfall_point": [round(c, 4) for c in self.landfall_point] if self.landfall_point else None,
            "landfall_shoreline_name": self.landfall_shoreline_name,
            "landfall_esi": self.landfall_esi,
            "eta_to_coast_hours": round(self.eta_to_coast_hours, 1) if self.eta_to_coast_hours is not None else None,
            "eta_uncertainty_range_hours": [round(h, 1) for h in self.eta_uncertainty_range_hours] if self.eta_uncertainty_range_hours else None,
            "threatened_assets": [a.to_dict() for a in self.threatened_assets],
            "threatened_assets_count": len(self.threatened_assets),
            "coastal_vulnerability_score": round(self.coastal_vulnerability_score, 1),
            "risk_tier": self.risk_tier,
            "containment_recommendations": self.containment_recommendations,
            "dispersant_restrictions": self.dispersant_restrictions,
            "assessment_timestamp": self.assessment_timestamp,
        }


def assess_coastal_impact(
    spill_lat: float,
    spill_lon: float,
    current_speed_ms: float = 0.35,
    current_bearing_deg: float = 120.0,
    wind_speed_ms: float = 5.0,
    wind_bearing_deg: float = 135.0,
    wind_leeway_factor: float = 0.03,
    forecast_results: Optional[List[dict]] = None,
    shoreline: Optional[List[CoastalPoint]] = None,
    sensitive_assets: Optional[List[SensitiveArea]] = None,
    max_simulation_hours: float = 72.0,
    step_minutes: int = 20,
) -> CoastalImpactResult:
    """
    Perform rigorous coastal impact and shoreline threat assessment.

    Args:
        spill_lat: Latitude of observed spill
        spill_lon: Longitude of observed spill
        current_speed_ms: Surface current velocity in m/s
        current_bearing_deg: Surface current direction toward degrees
        wind_speed_ms: Wind speed in m/s
        wind_bearing_deg: Wind blowing toward degrees
        wind_leeway_factor: Wind leeway coefficient (typically 0.03 for surface slicks)
        forecast_results: Optional forward drift results from ocean.hindcast
        shoreline: Optional custom shoreline point sequence
        sensitive_assets: Optional custom sensitive assets
        max_simulation_hours: Maximum forward integration horizon
        step_minutes: Integration time step

    Returns:
        CoastalImpactResult with comprehensive proximity and threat analysis.
    """
    if shoreline is None:
        shoreline = CHENNAI_CORRIDOR_SHORELINE
    if sensitive_assets is None:
        sensitive_assets = CHENNAI_SENSITIVE_AREAS

    # 1. Shortest distance to nearest shoreline point
    nearest_pt, shortest_dist_km = get_nearest_shoreline_point(spill_lat, spill_lon, shoreline)

    # 2. Combined net drift velocity vector
    # Current vector
    cur_rad = math.radians(current_bearing_deg)
    u_curr = current_speed_ms * math.sin(cur_rad)
    v_curr = current_speed_ms * math.cos(cur_rad)

    # Wind vector (3% leeway)
    wind_rad = math.radians(wind_bearing_deg)
    wind_drift_speed = wind_speed_ms * wind_leeway_factor
    u_wind = wind_drift_speed * math.sin(wind_rad)
    v_wind = wind_drift_speed * math.cos(wind_rad)

    # Net drift components (m/s)
    u_net = u_curr + u_wind
    v_net = v_curr + v_wind
    net_speed_ms = math.sqrt(u_net**2 + v_net**2)
    net_bearing_deg = math.degrees(math.atan2(u_net, v_net)) % 360

    # Convert net speed to km/h
    net_speed_kmh = net_speed_ms * 3.6

    # 3. Forward drift simulation to detect shoreline intersection
    landfall_projected = False
    landfall_point = None
    landfall_shoreline_name = None
    landfall_esi = None
    eta_to_coast_hours = None
    eta_uncertainty_range_hours = None

    # Step-by-step projection
    current_lat = spill_lat
    current_lon = spill_lon
    total_steps = int((max_simulation_hours * 60) / step_minutes)

    drift_corridor: List[Tuple[float, float]] = [(spill_lat, spill_lon)]

    for step in range(1, total_steps + 1):
        elapsed_hours = (step * step_minutes) / 60.0
        step_dist_km = net_speed_kmh * (step_minutes / 60.0)

        next_lat, next_lon = destination_point(current_lat, current_lon, net_bearing_deg, step_dist_km)
        drift_corridor.append((next_lat, next_lon))
        current_lat, current_lon = next_lat, next_lon

        # Check proximity to any shoreline point (within 1.5 km is considered shoreline contact)
        near_shore_pt, dist_to_shore = get_nearest_shoreline_point(next_lat, next_lon, shoreline)
        if dist_to_shore <= 2.0:
            landfall_projected = True
            landfall_point = (next_lat, next_lon)
            landfall_shoreline_name = near_shore_pt.name
            landfall_esi = int(near_shore_pt.esi)
            eta_to_coast_hours = elapsed_hours

            # Calculate uncertainty bounds: ±25% speed variation
            eta_min = max(0.5, elapsed_hours * 0.75)
            eta_max = elapsed_hours * 1.35
            eta_uncertainty_range_hours = (eta_min, eta_max)
            break

    # If direct trajectory did not intersect within simulation, but drift moves in onshore direction:
    # Check if bearing is onshore (bearing has a westward component toward Chennai coastline, ~180° to 360°)
    if not landfall_projected and shortest_dist_km > 0:
        shore_bearing = bearing_deg(spill_lat, spill_lon, nearest_pt.lat, nearest_pt.lon)
        bearing_diff = abs((net_bearing_deg - shore_bearing + 180) % 360 - 180)

        # If heading within 60 degrees of the shoreline
        if bearing_diff < 75 and net_speed_kmh > 0.05:
            effective_onshore_speed = net_speed_kmh * math.cos(math.radians(bearing_diff))
            if effective_onshore_speed > 0.1:
                est_hours = shortest_dist_km / effective_onshore_speed
                if est_hours <= max_simulation_hours * 1.5:
                    landfall_projected = True
                    landfall_point = (nearest_pt.lat, nearest_pt.lon)
                    landfall_shoreline_name = nearest_pt.name
                    landfall_esi = int(nearest_pt.esi)
                    eta_to_coast_hours = est_hours
                    eta_uncertainty_range_hours = (est_hours * 0.7, est_hours * 1.4)

    # Fallback ETA calculation if shortest distance is very small (< 10 km)
    if eta_to_coast_hours is None and shortest_dist_km < 10.0 and net_speed_kmh > 0:
        eta_to_coast_hours = shortest_dist_km / max(0.5, net_speed_kmh)
        eta_uncertainty_range_hours = (eta_to_coast_hours * 0.6, eta_to_coast_hours * 1.5)

    # 4. Identify Threatened Sensitive Assets
    threatened_assets: List[ThreatAsset] = []

    for asset in sensitive_assets:
        dist_spill = haversine_km(spill_lat, spill_lon, asset.lat, asset.lon)

        # Distance from nearest point on drift corridor
        min_corridor_dist = min(
            haversine_km(cp_lat, cp_lon, asset.lat, asset.lon)
            for cp_lat, cp_lon in drift_corridor
        )

        # Check if asset is within threatened perimeter
        is_threatened = (dist_spill <= 35.0) or (min_corridor_dist <= 15.0)

        if is_threatened:
            # Determine threat level
            if (min_corridor_dist <= 5.0 and landfall_projected and eta_to_coast_hours and eta_to_coast_hours <= 12) or dist_spill <= 10.0:
                threat_level = "IMMINENT"
            elif min_corridor_dist <= 12.0 or dist_spill <= 20.0:
                threat_level = "HIGH_RISK"
            elif min_corridor_dist <= 20.0 or dist_spill <= 35.0:
                threat_level = "POTENTIAL"
            else:
                threat_level = "MONITOR"

            # Estimate arrival time to this specific asset
            if net_speed_kmh > 0.1:
                arrival_est = dist_spill / net_speed_kmh
            else:
                arrival_est = None

            threatened_assets.append(ThreatAsset(
                asset=asset,
                distance_from_spill_km=dist_spill,
                distance_from_drift_corridor_km=min_corridor_dist,
                threat_level=threat_level,
                estimated_arrival_hours=arrival_est,
            ))

    # Sort threatened assets: IMMINENT first, then by corridor distance
    priority_order = {"IMMINENT": 0, "HIGH_RISK": 1, "POTENTIAL": 2, "MONITOR": 3}
    threatened_assets.sort(key=lambda a: (priority_order.get(a.threat_level, 4), a.distance_from_drift_corridor_km))

    # 5. Composite Coastal Vulnerability Score (0 - 100)
    # Proximity score (0-35 pts)
    if shortest_dist_km <= 5:
        proximity_score = 35.0
    elif shortest_dist_km <= 15:
        proximity_score = 25.0
    elif shortest_dist_km <= 30:
        proximity_score = 15.0
    else:
        proximity_score = max(2.0, 35.0 - (shortest_dist_km / 2.0))

    # Urgency / ETA score (0-30 pts)
    if eta_to_coast_hours is not None:
        if eta_to_coast_hours <= 6:
            urgency_score = 30.0
        elif eta_to_coast_hours <= 12:
            urgency_score = 24.0
        elif eta_to_coast_hours <= 24:
            urgency_score = 16.0
        elif eta_to_coast_hours <= 48:
            urgency_score = 8.0
        else:
            urgency_score = 3.0
    else:
        urgency_score = 5.0

    # Sensitivity & Assets score (0-35 pts)
    asset_threat_sum = 0.0
    for ta in threatened_assets[:3]:
        asset_threat_sum += (ta.asset.vulnerability_score / 100.0) * (
            15.0 if ta.threat_level == "IMMINENT" else (10.0 if ta.threat_level == "HIGH_RISK" else 5.0)
        )
    sensitivity_score = min(35.0, asset_threat_sum)

    coastal_vulnerability_score = min(100.0, proximity_score + urgency_score + sensitivity_score)

    # Risk Tier
    if coastal_vulnerability_score >= 75.0 or (eta_to_coast_hours is not None and eta_to_coast_hours <= 8):
        risk_tier = "CRITICAL"
    elif coastal_vulnerability_score >= 50.0:
        risk_tier = "HIGH"
    elif coastal_vulnerability_score >= 25.0:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # 6. Containment & Countermeasure Strategies
    recommendations: List[str] = []
    dispersant_restrictions: List[str] = []

    if shortest_dist_km <= 20.0:
        recommendations.append("Deploy ocean containment booms at earliest daylight or calm sea state")
        dispersant_restrictions.append("CRITICAL: Chemical dispersants prohibited within 10 km of shoreline (NEPA/CPCB guidelines)")

    if any(ta.asset.category in ["wildlife", "mangrove"] for ta in threatened_assets):
        recommendations.append("Erect high-buoyancy deflection booms at river mouths and lagoon inlets to divert surface slick into sandy sacrifice collection areas")
        dispersant_restrictions.append("Dispersant application prohibited in waters <20m depth to protect benthic nurseries and mangroves")

    if any(ta.asset.category == "infrastructure" for ta in threatened_assets):
        recommendations.append("Notify municipal desalination plant intakes and thermal cooling canals for emergency barrier inflation or intake shutdown")

    if any(ta.asset.category == "fishing" for ta in threatened_assets):
        recommendations.append("Issue immediate coastal VHF and SMS notice to artisanal landing centers to prevent contamination of nets and fiber boats")

    recommendations.append("Position fast-response skimmers (vortex or weir type) along the predicted drift trajectory corridor")

    return CoastalImpactResult(
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        nearest_shoreline_point=nearest_pt,
        shortest_distance_to_coast_km=shortest_dist_km,
        landfall_projected=landfall_projected,
        landfall_point=landfall_point,
        landfall_shoreline_name=landfall_shoreline_name,
        landfall_esi=landfall_esi,
        eta_to_coast_hours=eta_to_coast_hours,
        eta_uncertainty_range_hours=eta_uncertainty_range_hours,
        threatened_assets=threatened_assets,
        coastal_vulnerability_score=coastal_vulnerability_score,
        risk_tier=risk_tier,
        containment_recommendations=recommendations,
        dispersant_restrictions=dispersant_restrictions,
        assessment_timestamp=datetime.now(timezone.utc).isoformat(),
    )
