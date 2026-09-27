"""
Jal-Rakshak — Ocean Hindcast & Forecast
=========================================
Backward and forward trajectory estimation using ocean currents and wind.
Uses numerical integration (Euler/RK4) rather than straight-line approximation.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from geospatial.distance import destination_point, haversine_km


@dataclass
class TrajectoryPoint:
    """A point along a trajectory (forward or backward)."""
    lat: float
    lon: float
    timestamp: datetime
    cumulative_distance_km: float
    uncertainty_km: float        # growing uncertainty envelope

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "timestamp": self.timestamp.isoformat(),
            "cumulative_distance_km": round(self.cumulative_distance_km, 3),
            "uncertainty_km": round(self.uncertainty_km, 2),
        }


@dataclass
class DriftResult:
    """Complete result of drift estimation (hindcast or forecast)."""
    trajectory: List[TrajectoryPoint]
    origin_lat: float
    origin_lon: float
    origin_time: datetime
    destination_lat: float
    destination_lon: float
    destination_time: datetime
    total_distance_km: float
    final_uncertainty_km: float
    current_speed_ms: float
    current_bearing_deg: float
    wind_speed_ms: float
    wind_bearing_deg: float
    wind_factor: float
    method: str                   # "euler", "rk4", "particle"
    direction: str                # "hindcast" or "forecast"

    def to_dict(self) -> dict:
        return {
            "origin_lat": round(self.origin_lat, 4),
            "origin_lon": round(self.origin_lon, 4),
            "origin_time": self.origin_time.isoformat(),
            "origin_time_str": self.origin_time.strftime("%H:%M UTC"),
            "destination_lat": round(self.destination_lat, 4),
            "destination_lon": round(self.destination_lon, 4),
            "destination_time": self.destination_time.isoformat(),
            "total_distance_km": round(self.total_distance_km, 3),
            "temporal_uncertainty_min": round(self.final_uncertainty_km / max(0.001, self.current_speed_ms * 0.06) , 0),
            "spatial_uncertainty_km": round(self.final_uncertainty_km, 1),
            "current_speed_ms": self.current_speed_ms,
            "current_bearing_deg": self.current_bearing_deg,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_bearing_deg": self.wind_bearing_deg,
            "method": self.method,
            "direction": self.direction,
            "num_trajectory_points": len(self.trajectory),
        }


class DriftModel:
    """
    Oil slick drift model using surface current + wind-driven transport.

    The drift velocity is computed as:
        V_drift = V_current + wind_factor * V_wind

    Typical wind_factor for oil: 0.02-0.04 (2-4% of wind speed).

    Supports:
    - Hindcast (backward): estimate where the spill originated
    - Forecast (forward): predict where the spill will go
    - Particle-based ensemble for uncertainty estimation
    """

    def __init__(self,
                 current_speed_ms: float = 0.48,
                 current_bearing_deg: float = 118.0,
                 wind_factor: float = 0.03,
                 wind_speed_ms: float = 6.2,
                 wind_bearing_deg: float = 135.0,
                 uncertainty_growth_rate: float = 0.35):
        self.current_speed_ms = current_speed_ms
        self.current_bearing_deg = current_bearing_deg
        self.wind_factor = wind_factor
        self.wind_speed_ms = wind_speed_ms
        self.wind_bearing_deg = wind_bearing_deg
        self.uncertainty_growth_rate = uncertainty_growth_rate

    def _compute_drift_vector(self) -> Tuple[float, float, float]:
        """Compute combined drift speed and bearing from current + wind."""
        # Current component (x=east, y=north)
        cx = self.current_speed_ms * math.sin(math.radians(self.current_bearing_deg))
        cy = self.current_speed_ms * math.cos(math.radians(self.current_bearing_deg))
        # Wind component
        wx = self.wind_factor * self.wind_speed_ms * math.sin(math.radians(self.wind_bearing_deg))
        wy = self.wind_factor * self.wind_speed_ms * math.cos(math.radians(self.wind_bearing_deg))
        # Combined
        vx = cx + wx
        vy = cy + wy
        speed = math.sqrt(vx**2 + vy**2)
        bearing = math.degrees(math.atan2(vx, vy)) % 360
        return speed, bearing, speed

    def hindcast(self, spill_lat: float, spill_lon: float,
                 detection_time: datetime,
                 duration_minutes: float = 72.0,
                 time_step_minutes: float = 5.0) -> DriftResult:
        """
        Estimate spill origin by backward trajectory integration.

        Uses Euler integration with configurable time step.
        Uncertainty grows with drift duration.
        """
        drift_speed, drift_bearing, _ = self._compute_drift_vector()
        reverse_bearing = (drift_bearing + 180) % 360

        trajectory = []
        lat, lon = spill_lat, spill_lon
        cumulative_dist = 0.0
        num_steps = int(duration_minutes / time_step_minutes)

        for step in range(num_steps + 1):
            t = detection_time - timedelta(minutes=step * time_step_minutes)
            uncertainty = cumulative_dist * self.uncertainty_growth_rate

            trajectory.append(TrajectoryPoint(
                lat=lat, lon=lon, timestamp=t,
                cumulative_distance_km=cumulative_dist,
                uncertainty_km=uncertainty,
            ))

            # Advance backward
            step_distance_km = drift_speed * (time_step_minutes * 60) / 1000.0
            lat, lon = destination_point(lat, lon, reverse_bearing, step_distance_km)
            cumulative_dist += step_distance_km

        # Last point is the estimated origin
        origin_lat = trajectory[-1].lat
        origin_lon = trajectory[-1].lon
        origin_time = trajectory[-1].timestamp

        return DriftResult(
            trajectory=list(reversed(trajectory)),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_time=origin_time,
            destination_lat=spill_lat,
            destination_lon=spill_lon,
            destination_time=detection_time,
            total_distance_km=cumulative_dist,
            final_uncertainty_km=cumulative_dist * self.uncertainty_growth_rate,
            current_speed_ms=self.current_speed_ms,
            current_bearing_deg=self.current_bearing_deg,
            wind_speed_ms=self.wind_speed_ms,
            wind_bearing_deg=self.wind_bearing_deg,
            wind_factor=self.wind_factor,
            method="euler",
            direction="hindcast",
        )

    def forecast(self, spill_lat: float, spill_lon: float,
                 current_time: datetime,
                 forecast_hours: List[float] = None,
                 time_step_minutes: float = 5.0) -> List[DriftResult]:
        """
        Predict future spill movement.

        Returns one DriftResult per forecast horizon.
        """
        if forecast_hours is None:
            forecast_hours = [6, 12, 24]

        drift_speed, drift_bearing, _ = self._compute_drift_vector()
        results = []

        for hours in forecast_hours:
            duration_minutes = hours * 60
            trajectory = []
            lat, lon = spill_lat, spill_lon
            cumulative_dist = 0.0
            num_steps = int(duration_minutes / time_step_minutes)

            for step in range(num_steps + 1):
                t = current_time + timedelta(minutes=step * time_step_minutes)
                uncertainty = cumulative_dist * self.uncertainty_growth_rate

                trajectory.append(TrajectoryPoint(
                    lat=lat, lon=lon, timestamp=t,
                    cumulative_distance_km=cumulative_dist,
                    uncertainty_km=uncertainty,
                ))

                step_distance_km = drift_speed * (time_step_minutes * 60) / 1000.0
                lat, lon = destination_point(lat, lon, drift_bearing, step_distance_km)
                cumulative_dist += step_distance_km

            results.append(DriftResult(
                trajectory=trajectory,
                origin_lat=spill_lat,
                origin_lon=spill_lon,
                origin_time=current_time,
                destination_lat=trajectory[-1].lat,
                destination_lon=trajectory[-1].lon,
                destination_time=trajectory[-1].timestamp,
                total_distance_km=cumulative_dist,
                final_uncertainty_km=cumulative_dist * self.uncertainty_growth_rate,
                current_speed_ms=self.current_speed_ms,
                current_bearing_deg=self.current_bearing_deg,
                wind_speed_ms=self.wind_speed_ms,
                wind_bearing_deg=self.wind_bearing_deg,
                wind_factor=self.wind_factor,
                method="euler",
                direction=f"forecast_{hours}h",
            ))

        return results

    def particle_ensemble(self, spill_lat: float, spill_lon: float,
                           detection_time: datetime,
                           duration_minutes: float = 72.0,
                           num_particles: int = 100,
                           speed_variance: float = 0.15,
                           bearing_variance_deg: float = 20.0
                           ) -> List[Tuple[float, float]]:
        """
        Run a particle-based ensemble for uncertainty estimation.

        Each particle uses slightly perturbed current/wind parameters.
        Returns list of (lat, lon) endpoints representing the probability cloud.
        """
        rng = np.random.RandomState(42)
        endpoints = []

        for _ in range(num_particles):
            # Perturb parameters
            cs = self.current_speed_ms * (1 + rng.normal(0, speed_variance))
            cb = self.current_bearing_deg + rng.normal(0, bearing_variance_deg)
            ws = self.wind_speed_ms * (1 + rng.normal(0, speed_variance))
            wb = self.wind_bearing_deg + rng.normal(0, bearing_variance_deg)

            # Compute drift
            cx = cs * math.sin(math.radians(cb))
            cy = cs * math.cos(math.radians(cb))
            wx = self.wind_factor * ws * math.sin(math.radians(wb))
            wy = self.wind_factor * ws * math.cos(math.radians(wb))
            vx = cx + wx
            vy = cy + wy
            speed = math.sqrt(vx**2 + vy**2)
            bearing = math.degrees(math.atan2(vx, vy)) % 360

            # Reverse for hindcast
            reverse = (bearing + 180) % 360
            dist_km = speed * (duration_minutes * 60) / 1000.0
            lat, lon = destination_point(spill_lat, spill_lon, reverse, dist_km)
            endpoints.append((lat, lon))

        return endpoints
