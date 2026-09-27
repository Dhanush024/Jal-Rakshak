"""
Jal-Rakshak — Coastal Zones & Sensitivity Definitions
======================================================
Defines shoreline geometries, Environmental Sensitivity Index (ESI) classifications,
and protected marine / infrastructural zones.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional, Tuple
from geospatial.distance import haversine_km, bearing_deg


class EnvironmentalSensitivityIndex(IntEnum):
    """
    Standard NOAA/IMO Environmental Sensitivity Index (ESI) scale (1 - 10).
    Higher values represent greater ecological vulnerability and cleanup difficulty.
    """
    ESI_1_EXPOSED_ROCKY_SHORE = 1          # Steep rocky cliffs, high wave energy, low cleanup need
    ESI_2_EXPOSED_WAVE_CUT_PLATFORM = 2    # Wave-cut rock platforms
    ESI_3_FINE_GRAINED_SAND_BEACH = 3      # Hard packed sand, oil remains on surface, high recreational use
    ESI_4_COARSE_GRAINED_SAND_BEACH = 4    # Coarse sand, rapid oil penetration
    ESI_5_MIXED_SAND_GRAVEL_BEACH = 5      # High penetration, difficult to clean
    ESI_6_GRAVEL_COBBLE_BEACH = 6          # Deep oil penetration, long residence time
    ESI_7_EXPOSED_TIDAL_FLAT = 7           # Sand/mud flats with low wave energy
    ESI_8_SHELTERED_ROCKY_SHORE = 8        # Low wave energy, oil persists for years
    ESI_9_SHELTERED_TIDAL_FLAT = 9         # Highly productive soft mud, benthic nursery
    ESI_10_MANGROVES_SALT_MARSHES = 10     # Critical mangrove/marsh habitats, extreme vulnerability


@dataclass
class CoastalPoint:
    """A discretized point along a shoreline polyline."""
    name: str
    lat: float
    lon: float
    esi: EnvironmentalSensitivityIndex
    substrate: str
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "esi": int(self.esi),
            "substrate": self.substrate,
            "description": self.description,
        }


@dataclass
class SensitiveArea:
    """Ecologically or economically sensitive coastal asset."""
    name: str
    lat: float
    lon: float
    category: str                       # "wildlife", "mangrove", "fishing", "infrastructure", "tourism"
    esi: EnvironmentalSensitivityIndex
    vulnerability_score: float          # 0 - 100
    description: str
    priority: str                       # "CRITICAL", "HIGH", "MEDIUM"
    recommended_strategy: str           # e.g., "Deflection booming", "Exclusion booming"
    contact_authority: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "category": self.category,
            "esi": int(self.esi),
            "vulnerability_score": round(self.vulnerability_score, 1),
            "description": self.description,
            "priority": self.priority,
            "recommended_strategy": self.recommended_strategy,
            "contact_authority": self.contact_authority,
        }


@dataclass
class CoastalZone:
    """Regional shoreline zone grouping multiple points and assets."""
    zone_id: str
    name: str
    points: List[CoastalPoint]
    sensitive_assets: List[SensitiveArea]


# ─────────────────────────────────────────────────────────────────────────────
# Chennai / Coromandel Coastline Reference Model
# ─────────────────────────────────────────────────────────────────────────────

CHENNAI_CORRIDOR_SHORELINE: List[CoastalPoint] = [
    CoastalPoint("Pulicat North", 13.48, 80.33, EnvironmentalSensitivityIndex.ESI_9_SHELTERED_TIDAL_FLAT, "Tidal lagoon barrier", "Lagoon mouth and barrier island"),
    CoastalPoint("Pulicat South", 13.40, 80.32, EnvironmentalSensitivityIndex.ESI_9_SHELTERED_TIDAL_FLAT, "Mudflats and salt pans", "Bird nesting grounds and shallow bay"),
    CoastalPoint("Kattupalli", 13.31, 80.34, EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH, "Industrial sand spit", "Port approach and breakwaters"),
    CoastalPoint("Ennore Shoals", 13.24, 80.33, EnvironmentalSensitivityIndex.ESI_10_MANGROVES_SALT_MARSHES, "Tidal creek and mangrove fringe", "Ennore Creek estuary and fisheries"),
    CoastalPoint("Thiruvottiyur", 13.16, 80.30, EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH, "Exposed sandy coast", "Dense artisanal fishing settlements"),
    CoastalPoint("Chennai Port / Royapuram", 13.10, 80.30, EnvironmentalSensitivityIndex.ESI_1_EXPOSED_ROCKY_SHORE, "Reinforced riprap and seawalls", "Commercial harbor breakwater"),
    CoastalPoint("Marina Beach North", 13.06, 80.28, EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH, "Wide sandy beach", "High-density public recreation & nesting zone"),
    CoastalPoint("Adyar River Mouth", 13.01, 80.27, EnvironmentalSensitivityIndex.ESI_10_MANGROVES_SALT_MARSHES, "Estuarine tidal mudflats", "Protected estuarine eco-park & fish nursery"),
    CoastalPoint("Besant Nagar / Thiruvanmiyur", 12.98, 80.26, EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH, "Sandy recreational beach", "Olive Ridley sea turtle nesting beach"),
    CoastalPoint("Kottivakkam / Neelankarai", 12.94, 80.25, EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH, "Sandy residential coast", "Small craft artisanal landing center"),
    CoastalPoint("Covelong Bay", 12.79, 80.25, EnvironmentalSensitivityIndex.ESI_7_EXPOSED_TIDAL_FLAT, "Cove and shallow sandbar", "Active fishing harbor, watersports, and heritage"),
    CoastalPoint("Mahabalipuram North", 12.63, 80.20, EnvironmentalSensitivityIndex.ESI_2_EXPOSED_WAVE_CUT_PLATFORM, "Granite boulders and sandy beaches", "UNESCO World Heritage shoreline & reefs"),
    CoastalPoint("Kalpakkam", 12.56, 80.17, EnvironmentalSensitivityIndex.ESI_4_COARSE_GRAINED_SAND_BEACH, "Sandy coast with thermal intake", "Nuclear power station seawater intake canal"),
]


CHENNAI_SENSITIVE_AREAS: List[SensitiveArea] = [
    SensitiveArea(
        name="Pulicat Lake Bird Sanctuary",
        lat=13.42,
        lon=80.32,
        category="wildlife",
        esi=EnvironmentalSensitivityIndex.ESI_10_MANGROVES_SALT_MARSHES,
        vulnerability_score=96.0,
        description="India's second largest brackish-water lagoon; critical wintering ground for over 15,000 flamingos and migratory waterfowl.",
        priority="CRITICAL",
        recommended_strategy="Deploy heavy-duty exclusion booming across lagoon inlets to prevent tidal intrusion.",
        contact_authority="Tamil Nadu Forest & Wildlife Department / Coast Guard District HQ 5",
    ),
    SensitiveArea(
        name="Ennore Creek & Mangrove Ecosystem",
        lat=13.22,
        lon=80.32,
        category="mangrove",
        esi=EnvironmentalSensitivityIndex.ESI_10_MANGROVES_SALT_MARSHES,
        vulnerability_score=94.0,
        description="Tidal creek with remnant Avicennia marina mangroves, mudflats, and traditional artisanal fishing grounds.",
        priority="CRITICAL",
        recommended_strategy="Multi-tier deflection booming into calm sacrificial recovery zones; strict dispersant ban within 5 km.",
        contact_authority="State Coastal Zone Management Authority (TNSCZMA)",
    ),
    SensitiveArea(
        name="Minjur Seawater Desalination Plant Intake",
        lat=13.19,
        lon=80.33,
        category="infrastructure",
        esi=EnvironmentalSensitivityIndex.ESI_1_EXPOSED_ROCKY_SHORE,
        vulnerability_score=90.0,
        description="Key municipal drinking water supply (100 MLD) for Greater Chennai municipal region.",
        priority="CRITICAL",
        recommended_strategy="Deploy high-buoyancy oil containment booms around intake offshore buoy; activate intake shut-off alert.",
        contact_authority="Chennai Metropolitan Water Supply and Sewerage Board (CMWSSB)",
    ),
    SensitiveArea(
        name="Marina & Besant Nagar Sea Turtle Nesting Corridor",
        lat=13.02,
        lon=80.27,
        category="wildlife",
        esi=EnvironmentalSensitivityIndex.ESI_3_FINE_GRAINED_SAND_BEACH,
        vulnerability_score=82.0,
        description="Vulnerable nesting habitat for endangered Olive Ridley sea turtles (Lepidochelys olivacea).",
        priority="HIGH",
        recommended_strategy="Offshore interceptor skimming; manual non-mechanized shoreline barrier protection.",
        contact_authority="Wildlife Warden Chennai / Tree Foundation Turtle Network",
    ),
    SensitiveArea(
        name="Covelong Artisanal Fishing Harbor",
        lat=12.79,
        lon=80.25,
        category="fishing",
        esi=EnvironmentalSensitivityIndex.ESI_7_EXPOSED_TIDAL_FLAT,
        vulnerability_score=78.0,
        description="Landing center for 450+ catamarans and fiber boats; active artisanal shrimp and crab fisheries.",
        priority="HIGH",
        recommended_strategy="Harbor entrance harbor boom; emergency advisory to haul artisanal craft above high tide mark.",
        contact_authority="Department of Fisheries, Tamil Nadu / Local Fishermen Co-op",
    ),
    SensitiveArea(
        name="Madras Atomic Power Station (MAPS) Seawater Intake",
        lat=12.56,
        lon=80.18,
        category="infrastructure",
        esi=EnvironmentalSensitivityIndex.ESI_1_EXPOSED_ROCKY_SHORE,
        vulnerability_score=88.0,
        description="Seawater intake canal providing secondary cooling for nuclear reactor turbines.",
        priority="CRITICAL",
        recommended_strategy="Rapid deployment of deep-skirt harbor containment booms and surface skimmers at canal entrance.",
        contact_authority="Nuclear Power Corporation of India (NPCIL) Emergency Directorate",
    ),
]


def get_nearest_shoreline_point(
    lat: float,
    lon: float,
    shoreline: Optional[List[CoastalPoint]] = None,
) -> Tuple[CoastalPoint, float]:
    """
    Find the closest shoreline point to a given coordinate.

    Returns:
        Tuple of (CoastalPoint, distance_in_km)
    """
    if shoreline is None:
        shoreline = CHENNAI_CORRIDOR_SHORELINE

    nearest_pt = shoreline[0]
    min_dist = float("inf")

    for pt in shoreline:
        d = haversine_km(lat, lon, pt.lat, pt.lon)
        if d < min_dist:
            min_dist = d
            nearest_pt = pt

    return nearest_pt, round(min_dist, 2)
