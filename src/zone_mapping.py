"""Maps raw `locality` strings to one of five MMR macro micro-market zones.

Two-stage strategy:
1. Curated keyword lookup covering the high-volume, well-known localities.
2. Haversine nearest-centroid fallback (using each zone's mean lat/lon,
   computed from stage-1-mapped rows) for any locality stage 1 misses --
   so every one of the 400+ raw locality strings still resolves to a zone
   without needing an exhaustive hand-written table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    MACRO_ZONES,
    ZONE_CENTRAL_SUBURBS,
    ZONE_NAVI_MUMBAI,
    ZONE_SOUTH_MUMBAI,
    ZONE_THANE_EXTENDED,
    ZONE_WESTERN_SUBURBS,
    get_logger,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Stage 1: curated keyword -> zone lookup.
# Keys are matched as case-insensitive substrings against the (normalized)
# locality string. Order matters: first match wins, so more specific keys
# are listed before broader ones.
# --------------------------------------------------------------------------- #
_ZONE_KEYWORDS: dict[str, str] = {
    # --- South Mumbai (island city, south of Mahim/Sion) ---
    "colaba": ZONE_SOUTH_MUMBAI, "cuffe parade": ZONE_SOUTH_MUMBAI,
    "fort": ZONE_SOUTH_MUMBAI, "churchgate": ZONE_SOUTH_MUMBAI,
    "marine drive": ZONE_SOUTH_MUMBAI, "marine lines": ZONE_SOUTH_MUMBAI,
    "nariman": ZONE_SOUTH_MUMBAI, "malabar hill": ZONE_SOUTH_MUMBAI,
    "breach candy": ZONE_SOUTH_MUMBAI, "cumballa hill": ZONE_SOUTH_MUMBAI,
    "altamount": ZONE_SOUTH_MUMBAI, "napean sea": ZONE_SOUTH_MUMBAI,
    "napeansea": ZONE_SOUTH_MUMBAI, "peddar road": ZONE_SOUTH_MUMBAI,
    "girgaon": ZONE_SOUTH_MUMBAI, "kalbadevi": ZONE_SOUTH_MUMBAI,
    "bhuleshwar": ZONE_SOUTH_MUMBAI, "crawford market": ZONE_SOUTH_MUMBAI,
    "dongri": ZONE_SOUTH_MUMBAI, "umerkhadi": ZONE_SOUTH_MUMBAI,
    "gamdevi": ZONE_SOUTH_MUMBAI, "grant road": ZONE_SOUTH_MUMBAI,
    "tardeo": ZONE_SOUTH_MUMBAI, "mahalaxmi": ZONE_SOUTH_MUMBAI,
    "worli": ZONE_SOUTH_MUMBAI, "lower parel": ZONE_SOUTH_MUMBAI,
    "prabhadevi": ZONE_SOUTH_MUMBAI, "byculla": ZONE_SOUTH_MUMBAI,
    "mazgaon": ZONE_SOUTH_MUMBAI, "mazagaon": ZONE_SOUTH_MUMBAI,
    "dockyard": ZONE_SOUTH_MUMBAI, "jacob circle": ZONE_SOUTH_MUMBAI,
    "nagpada": ZONE_SOUTH_MUMBAI, "madanpura": ZONE_SOUTH_MUMBAI,
    "kamathipura": ZONE_SOUTH_MUMBAI, "mumbai central": ZONE_SOUTH_MUMBAI,
    "agripada": ZONE_SOUTH_MUMBAI, "century mills": ZONE_SOUTH_MUMBAI,
    "bdd chawls": ZONE_SOUTH_MUMBAI, "shivaji park": ZONE_SOUTH_MUMBAI,
    "dadar": ZONE_SOUTH_MUMBAI, "matunga": ZONE_SOUTH_MUMBAI,
    "sion": ZONE_SOUTH_MUMBAI, "mahim": ZONE_SOUTH_MUMBAI,
    "parel": ZONE_SOUTH_MUMBAI, "dharavi": ZONE_SOUTH_MUMBAI,
    "wadala": ZONE_SOUTH_MUMBAI, "sewri": ZONE_SOUTH_MUMBAI,
    "bhakti park": ZONE_SOUTH_MUMBAI, "antop hill": ZONE_SOUTH_MUMBAI,
    "irani wadi": ZONE_SOUTH_MUMBAI, "natakwala": ZONE_SOUTH_MUMBAI,
    "murarbaug": ZONE_SOUTH_MUMBAI,

    # --- Western Suburbs (Bandra to Dahisar / Mira-Bhayandar) ---
    "bandra": ZONE_WESTERN_SUBURBS, "bkc": ZONE_WESTERN_SUBURBS,
    "khar": ZONE_WESTERN_SUBURBS, "santacruz": ZONE_WESTERN_SUBURBS,
    "santacuz": ZONE_WESTERN_SUBURBS, "vakola": ZONE_WESTERN_SUBURBS,
    "kalina": ZONE_WESTERN_SUBURBS, "juhu": ZONE_WESTERN_SUBURBS,
    "vile parle": ZONE_WESTERN_SUBURBS, "ville parle": ZONE_WESTERN_SUBURBS,
    "navpada vile parle": ZONE_WESTERN_SUBURBS, "jvpd": ZONE_WESTERN_SUBURBS,
    "andheri": ZONE_WESTERN_SUBURBS, "jogeshwari": ZONE_WESTERN_SUBURBS,
    "oshiwara": ZONE_WESTERN_SUBURBS, "lokhandwala": ZONE_WESTERN_SUBURBS,
    "versova": ZONE_WESTERN_SUBURBS, "yari road": ZONE_WESTERN_SUBURBS,
    "veera desai": ZONE_WESTERN_SUBURBS, "chakala": ZONE_WESTERN_SUBURBS,
    "marol": ZONE_WESTERN_SUBURBS, "jvlr": ZONE_WESTERN_SUBURBS,
    "j b nagar": ZONE_WESTERN_SUBURBS, "saki naka": ZONE_WESTERN_SUBURBS,
    "sakinaka": ZONE_WESTERN_SUBURBS, "dn nagar": ZONE_WESTERN_SUBURBS,
    "amboli": ZONE_WESTERN_SUBURBS, "evershine": ZONE_WESTERN_SUBURBS,
    "goregaon": ZONE_WESTERN_SUBURBS, "malad": ZONE_WESTERN_SUBURBS,
    "kandivali": ZONE_WESTERN_SUBURBS, "charkop": ZONE_WESTERN_SUBURBS,
    "borivali": ZONE_WESTERN_SUBURBS, "dahisar": ZONE_WESTERN_SUBURBS,
    "gorai": ZONE_WESTERN_SUBURBS, "eksar": ZONE_WESTERN_SUBURBS,
    "magathane": ZONE_WESTERN_SUBURBS, "yogi nagar": ZONE_WESTERN_SUBURBS,
    "kharodi": ZONE_WESTERN_SUBURBS, "madh": ZONE_WESTERN_SUBURBS,
    "mira road": ZONE_WESTERN_SUBURBS, "mira bhayand": ZONE_WESTERN_SUBURBS,
    "bhayandar": ZONE_WESTERN_SUBURBS, "bolinj": ZONE_WESTERN_SUBURBS,
    "vasai": ZONE_WESTERN_SUBURBS, "nalasopara": ZONE_WESTERN_SUBURBS,
    "nala sopara": ZONE_WESTERN_SUBURBS, "nallasopara": ZONE_WESTERN_SUBURBS,
    "virar": ZONE_WESTERN_SUBURBS, "naigaon": ZONE_WESTERN_SUBURBS,
    "vazira": ZONE_WESTERN_SUBURBS, "navghar": ZONE_WESTERN_SUBURBS,
    "navgharh": ZONE_WESTERN_SUBURBS, "uttan": ZONE_WESTERN_SUBURBS,
    "boisar": ZONE_WESTERN_SUBURBS, "palghar": ZONE_WESTERN_SUBURBS,
    "saphale": ZONE_WESTERN_SUBURBS, "satpati": ZONE_WESTERN_SUBURBS,
    "dahanu": ZONE_WESTERN_SUBURBS, "umroli": ZONE_WESTERN_SUBURBS,
    "tembhode": ZONE_WESTERN_SUBURBS, "vajreshwari": ZONE_WESTERN_SUBURBS,
    "bandstand": ZONE_WESTERN_SUBURBS, "pali hill": ZONE_WESTERN_SUBURBS,
    "vakas": ZONE_WESTERN_SUBURBS, "battipada": ZONE_WESTERN_SUBURBS,

    # --- Central Suburbs (Central line: Kurla to Mulund, Powai, Chembur belt) ---
    "kurla": ZONE_CENTRAL_SUBURBS, "vidya vihar": ZONE_CENTRAL_SUBURBS,
    "ghatkopar": ZONE_CENTRAL_SUBURBS, "vikhroli": ZONE_CENTRAL_SUBURBS,
    "vikroli": ZONE_CENTRAL_SUBURBS, "kanjur": ZONE_CENTRAL_SUBURBS,
    "bhandup": ZONE_CENTRAL_SUBURBS, "mulund": ZONE_CENTRAL_SUBURBS,
    "nahur": ZONE_CENTRAL_SUBURBS, "powai": ZONE_CENTRAL_SUBURBS,
    "hiranandani": ZONE_CENTRAL_SUBURBS, "chandivali": ZONE_CENTRAL_SUBURBS,
    "chembur": ZONE_CENTRAL_SUBURBS, "govandi": ZONE_CENTRAL_SUBURBS,
    "mankhurd": ZONE_CENTRAL_SUBURBS, "deonar": ZONE_CENTRAL_SUBURBS,
    "tilak nagar": ZONE_CENTRAL_SUBURBS, "chedda nagar": ZONE_CENTRAL_SUBURBS,
    "vinobha bhave": ZONE_CENTRAL_SUBURBS, "sindhi society": ZONE_CENTRAL_SUBURBS,
    "vallabh baug": ZONE_CENTRAL_SUBURBS, "pant nagar": ZONE_CENTRAL_SUBURBS,
    "kannamwar": ZONE_CENTRAL_SUBURBS, "tagore nagar": ZONE_CENTRAL_SUBURBS,
    "ashok nagar": ZONE_CENTRAL_SUBURBS, "subhash nagar": ZONE_CENTRAL_SUBURBS,
    "sahkar nagar": ZONE_CENTRAL_SUBURBS, "asalpha": ZONE_CENTRAL_SUBURBS,
    "lbs marg": ZONE_CENTRAL_SUBURBS, "ern express highway": ZONE_CENTRAL_SUBURBS,
    "gokuldham": ZONE_CENTRAL_SUBURBS, "p l lokhande": ZONE_CENTRAL_SUBURBS,

    # --- Navi Mumbai ---
    "navi mumbai": ZONE_NAVI_MUMBAI, "vashi": ZONE_NAVI_MUMBAI,
    "nerul": ZONE_NAVI_MUMBAI, "belapur": ZONE_NAVI_MUMBAI,
    "cbd belapur": ZONE_NAVI_MUMBAI, "kharghar": ZONE_NAVI_MUMBAI,
    "kamothe": ZONE_NAVI_MUMBAI, "kalamboli": ZONE_NAVI_MUMBAI,
    "panvel": ZONE_NAVI_MUMBAI, "khanda": ZONE_NAVI_MUMBAI,
    "kharkopar": ZONE_NAVI_MUMBAI, "roadpali": ZONE_NAVI_MUMBAI,
    "taloja": ZONE_NAVI_MUMBAI, "taloje": ZONE_NAVI_MUMBAI,
    "karanjade": ZONE_NAVI_MUMBAI, "koproli": ZONE_NAVI_MUMBAI,
    "old panvel": ZONE_NAVI_MUMBAI, "new panvel": ZONE_NAVI_MUMBAI,
    "ulwe": ZONE_NAVI_MUMBAI, "dronagiri": ZONE_NAVI_MUMBAI,
    "uran": ZONE_NAVI_MUMBAI, "juinagar": ZONE_NAVI_MUMBAI,
    "sanpada": ZONE_NAVI_MUMBAI, "seawoods": ZONE_NAVI_MUMBAI,
    "sector": ZONE_NAVI_MUMBAI,  # generic "Sector N <place>" strings are Navi Mumbai
    "ghansoli": ZONE_NAVI_MUMBAI, "rabale": ZONE_NAVI_MUMBAI,
    "koparkhairane": ZONE_NAVI_MUMBAI, "koper khairane": ZONE_NAVI_MUMBAI,
    "airoli": ZONE_NAVI_MUMBAI, "greater khanda": ZONE_NAVI_MUMBAI,
    "haware city": ZONE_NAVI_MUMBAI, "kolhare": ZONE_NAVI_MUMBAI,
    "kongaon": ZONE_NAVI_MUMBAI, "kon": ZONE_NAVI_MUMBAI,
    "kewale": ZONE_NAVI_MUMBAI, "navade": ZONE_NAVI_MUMBAI,
    "vichumbe": ZONE_NAVI_MUMBAI, "pisarve": ZONE_NAVI_MUMBAI,
    "shirgaon": ZONE_NAVI_MUMBAI, "usarghar": ZONE_NAVI_MUMBAI,
    "khalapur": ZONE_NAVI_MUMBAI, "khopoli": ZONE_NAVI_MUMBAI,
    "lodhivali": ZONE_NAVI_MUMBAI, "karjat": ZONE_NAVI_MUMBAI,
    "chikan ghar": ZONE_NAVI_MUMBAI,

    # --- Thane & extended MMR (Thane city, Kalyan-Dombivli, Bhiwandi, Kasara belt) ---
    "thane": ZONE_THANE_EXTENDED, "kolshet": ZONE_THANE_EXTENDED,
    "majiwada": ZONE_THANE_EXTENDED, "ghodbunder": ZONE_THANE_EXTENDED,
    "manpada": ZONE_THANE_EXTENDED, "patlipada": ZONE_THANE_EXTENDED,
    "vasant vihar": ZONE_THANE_EXTENDED, "louis wadi": ZONE_THANE_EXTENDED,
    "panch pakh": ZONE_THANE_EXTENDED, "naupada": ZONE_THANE_EXTENDED,
    "balkum": ZONE_THANE_EXTENDED, "kasar vadavali": ZONE_THANE_EXTENDED,
    "kasaradavali": ZONE_THANE_EXTENDED, "owale": ZONE_THANE_EXTENDED,
    "pokhran": ZONE_THANE_EXTENDED, "vartak nagar": ZONE_THANE_EXTENDED,
    "kalwa": ZONE_THANE_EXTENDED, "mumbra": ZONE_THANE_EXTENDED,
    "diva": ZONE_THANE_EXTENDED, "dive": ZONE_THANE_EXTENDED,
    "kopri": ZONE_THANE_EXTENDED, "bhiwandi": ZONE_THANE_EXTENDED,
    "kalyan": ZONE_THANE_EXTENDED, "dombivali": ZONE_THANE_EXTENDED,
    "dombivli": ZONE_THANE_EXTENDED, "titwala": ZONE_THANE_EXTENDED,
    "ambivali": ZONE_THANE_EXTENDED, "ambivli": ZONE_THANE_EXTENDED,
    "khadakpada": ZONE_THANE_EXTENDED, "thakurli": ZONE_THANE_EXTENDED,
    "nilje": ZONE_THANE_EXTENDED, "shilphata": ZONE_THANE_EXTENDED,
    "shil phata": ZONE_THANE_EXTENDED, "haji malang": ZONE_THANE_EXTENDED,
    "ambarnath": ZONE_THANE_EXTENDED, "ambernath": ZONE_THANE_EXTENDED,
    "ulhasnagar": ZONE_THANE_EXTENDED, "badlapur": ZONE_THANE_EXTENDED,
    "vangani": ZONE_THANE_EXTENDED, "shelu": ZONE_THANE_EXTENDED,
    "neral": ZONE_THANE_EXTENDED, "asangaon": ZONE_THANE_EXTENDED,
    "atgaon": ZONE_THANE_EXTENDED, "khardi": ZONE_THANE_EXTENDED,
    "vasind": ZONE_THANE_EXTENDED, "shahapur": ZONE_THANE_EXTENDED,
    "shahpur": ZONE_THANE_EXTENDED, "murbad": ZONE_THANE_EXTENDED,
    "kasheli": ZONE_THANE_EXTENDED, "anjurdive": ZONE_THANE_EXTENDED,
    "purna": ZONE_THANE_EXTENDED, "temghar": ZONE_THANE_EXTENDED,
    "manor": ZONE_THANE_EXTENDED, "wada": ZONE_THANE_EXTENDED,
    "jawhar": ZONE_THANE_EXTENDED, "vikramgad": ZONE_THANE_EXTENDED,
    "lonavala": ZONE_THANE_EXTENDED, "khopoli road": ZONE_THANE_EXTENDED,
    "rasayani": ZONE_THANE_EXTENDED, "palava": ZONE_THANE_EXTENDED,
    "shri hari nagar": ZONE_THANE_EXTENDED, "sambhaji nagar": ZONE_THANE_EXTENDED,
}

_EARTH_RADIUS_KM = 6371.0


def _normalize(text: str) -> str:
    return str(text).strip().lower()


def _keyword_zone(locality_norm: str) -> str | None:
    for keyword, zone in _ZONE_KEYWORDS.items():
        if keyword in locality_norm:
            return zone
    return None


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorized great-circle distance (km) from arrays of points to one point."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def assign_macro_zone(df: pd.DataFrame) -> pd.Series:
    """Assign each row a macro micro-market zone.

    Stage 1 applies the curated keyword lookup. Stage 2 computes each zone's
    lat/lon centroid from stage-1-mapped rows and assigns every still-unmapped
    row to its geographically nearest centroid, guaranteeing full coverage.

    Args:
        df: DataFrame with `locality`, `latitude`, `longitude` columns.

    Returns:
        Series of zone labels aligned to df's index.
    """
    locality_norm = df["locality"].astype(str).map(_normalize)
    zone = locality_norm.map(_keyword_zone)

    n_unmapped = int(zone.isna().sum())
    if n_unmapped:
        logger.info("Keyword stage mapped %d/%d rows; resolving %d via geo-centroid fallback.",
                    len(df) - n_unmapped, len(df), n_unmapped)

        mapped_mask = zone.notna()
        centroids = (
            df.loc[mapped_mask, ["latitude", "longitude"]]
            .groupby(zone[mapped_mask])
            .mean()
        )
        # Guarantee every declared zone has a centroid even if a category is thin.
        for z in MACRO_ZONES:
            if z not in centroids.index:
                centroids.loc[z] = df.loc[mapped_mask, ["latitude", "longitude"]].mean()

        unmapped_idx = zone[~mapped_mask].index
        lat = df.loc[unmapped_idx, "latitude"].to_numpy()
        lon = df.loc[unmapped_idx, "longitude"].to_numpy()

        dist_matrix = np.column_stack([
            _haversine_km(lat, lon, row["latitude"], row["longitude"])
            for _, row in centroids.iterrows()
        ])
        nearest = np.array(centroids.index)[dist_matrix.argmin(axis=1)]
        zone.loc[unmapped_idx] = nearest

    return zone.astype("category")
