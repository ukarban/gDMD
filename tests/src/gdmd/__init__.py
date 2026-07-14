"""Utilities for gap-based dynamic mode decomposition (gDMD)."""

from .analysis import GDMDAnalysisOptions, run_gdmd_analysis
from .dmd import DMDResult, dmd_pair
from .dns_gap import compute_dns_gap_fields
from .gaps import compute_gap_snapshot_pair, second_moment_gaps
from .mode_matching import match_dmd_modes
from .ptv import (
    compute_ptv_fields_ray,
    compute_ptv_mean_velocity,
    custom_y_grid_from_reference,
    find_ptv_borders,
    load_ptv_pair,
)
from .state import rectangular_window_mask, restrict_state_matrix
from .weighting import apply_spatial_weights, tke_weights_from_velocity, undo_spatial_weights

__all__ = [
    "DMDResult",
    "GDMDAnalysisOptions",
    "dmd_pair",
    "second_moment_gaps",
    "compute_gap_snapshot_pair",
    "compute_dns_gap_fields",
    "match_dmd_modes",
    "load_ptv_pair",
    "find_ptv_borders",
    "custom_y_grid_from_reference",
    "compute_ptv_mean_velocity",
    "compute_ptv_fields_ray",
    "run_gdmd_analysis",
    "restrict_state_matrix",
    "rectangular_window_mask",
    "tke_weights_from_velocity",
    "apply_spatial_weights",
    "undo_spatial_weights",
]
