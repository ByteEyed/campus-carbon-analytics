"""
Campus Carbon Analytics - Executive Dashboard Module
====================================================
Academic Project: BDS-36 (T.Y. B.Sc. Data Science, Semester V)
Academic Year: 2026-27

Phase 17: Executive Dashboard
Provides a presentation layer consuming Phase 16 integrated artifacts.
"""

from src.dashboard.adapter import DashboardDataError, load_dashboard_data

__all__ = [
    "DashboardDataError",
    "load_dashboard_data",
]
