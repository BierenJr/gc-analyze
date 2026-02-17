#!/usr/bin/env python3
"""GC Log Analyzer v3.0 - Enhanced with G1 & CMS GC Support.

Modern, type-safe GC log analyzer supporting:
- Parallel GC (PSYoungGen/ParOldGen)
- G1 GC (unified logging and legacy formats)
- CMS GC (Concurrent Mark Sweep with ParNew)
- Rich terminal output with tables and panels
- Memory leak detection with severity scoring (0-100)
- Metaspace analysis
- G1-specific metrics (mixed GC, humongous allocations, evacuation failures)
- CMS-specific metrics (concurrent cycles, failures)
- Optional Markdown export
"""

from __future__ import annotations

import cProfile
import heapq
import pstats
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Annotated, Any, Literal, Protocol, TypeAlias

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ============================================================
# TYPE ALIASES
# ============================================================

GCKind: TypeAlias = Literal["Young", "Full", "Mixed", "Remark", "Cleanup", "Concurrent"]
KilobytesValue: TypeAlias = int
PercentageValue: TypeAlias = float
SecondsValue: TypeAlias = float

# ============================================================
# PYDANTIC MODELS
# ============================================================


class GCEvent(BaseModel):
    """Normalized GC event structure (GC-agnostic)."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    uptime_seconds: float
    gc_kind: GCKind
    pause_time_seconds: SecondsValue

    heap_before_kb: KilobytesValue
    heap_after_kb: KilobytesValue
    heap_total_kb: KilobytesValue
    heap_used_percentage: PercentageValue

    young_before_kb: KilobytesValue | None = None
    young_after_kb: KilobytesValue | None = None
    young_total_kb: KilobytesValue | None = None

    old_before_kb: KilobytesValue | None = None
    old_after_kb: KilobytesValue | None = None
    old_total_kb: KilobytesValue | None = None
    old_used_percentage: PercentageValue | None = None

    # G1-specific fields
    humongous_detected: bool = False
    evacuation_failed: bool = False
    to_space_exhausted: bool = False

    # Metaspace
    metaspace_before_kb: KilobytesValue | None = None
    metaspace_after_kb: KilobytesValue | None = None
    metaspace_total_kb: KilobytesValue | None = None


class DiagnosticThresholds(BaseModel):
    """Configurable thresholds for diagnostic warnings."""

    heap_critical_percentage: float = 95.0
    heap_warning_percentage: float = 80.0
    heap_minimum_headroom_percentage: float = 5.0

    full_gc_critical_per_hour: float = 10.0
    full_gc_warning_per_hour: float = 1.0

    gc_overhead_critical_percentage: float = 20.0
    gc_overhead_warning_percentage: float = 10.0

    pause_critical_seconds: float = 10.0
    pause_warning_seconds: float = 5.0
    pause_extreme_seconds: float = 60.0

    old_gen_growth_critical_percentage: float = 20.0
    old_gen_growth_warning_percentage: float = 10.0

    reclamation_decline_threshold: float = 0.7  # Trigger at 30% decline (was 50%)
    full_gc_acceleration_threshold: float = 0.7

    high_promotion_rate_kb: int = 10240


class LeakSeverity(BaseModel):
    """Memory leak severity assessment.

    Note: Does not include time-to-OOM predictions as linear extrapolation
    is unreliable for real-world memory leak scenarios.
    """

    score: float = Field(ge=0.0, le=100.0)
    severity_level: Literal["None", "Low", "Medium", "High", "Critical"]
    indicators: list[str] = Field(default_factory=list)


class MetaspaceAnalysis(BaseModel):
    """Metaspace usage statistics and diagnostics."""

    metaspace_events_count: int
    average_metaspace_used_percentage: float
    peak_metaspace_used_percentage: float
    metaspace_growth_kb: int
    metaspace_growing: bool
    metaspace_warnings: list[str] = Field(default_factory=list)


class JVMHeapConfig(BaseModel):
    """JVM heap and GC tuning configuration extracted from CommandLine flags."""

    # Heap sizing
    initial_heap_size_bytes: int | None = None  # -Xms or -XX:InitialHeapSize
    max_heap_size_bytes: int | None = None  # -Xmx or -XX:MaxHeapSize
    new_size_bytes: int | None = None  # -Xmn or -XX:NewSize
    max_new_size_bytes: int | None = None  # -XX:MaxNewSize
    old_size_bytes: int | None = None  # -XX:OldSize
    max_ram_bytes: int | None = None  # -XX:MaxRAM
    initial_ram_percentage: int | None = None  # -XX:InitialRAMPercentage
    min_ram_percentage: int | None = None  # -XX:MinRAMPercentage
    max_ram_percentage: int | None = None  # -XX:MaxRAMPercentage
    max_ram_fraction: int | None = None  # -XX:MaxRAMFraction (legacy)
    soft_max_heap_size_bytes: int | None = None  # -XX:SoftMaxHeapSize (ZGC)
    always_pre_touch: bool | None = None  # -XX:+AlwaysPreTouch
    thread_stack_size_kb: int | None = None  # -XX:ThreadStackSize
    new_ratio: int | None = None  # -XX:NewRatio
    survivor_ratio: int | None = None  # -XX:SurvivorRatio
    initial_survivor_ratio: int | None = None  # -XX:InitialSurvivorRatio
    target_survivor_ratio: int | None = None  # -XX:TargetSurvivorRatio
    max_tenuring_threshold: int | None = None  # -XX:MaxTenuringThreshold
    max_heap_free_ratio: int | None = None  # -XX:MaxHeapFreeRatio
    min_heap_free_ratio: int | None = None  # -XX:MinHeapFreeRatio

    # Metaspace
    metaspace_size_bytes: int | None = None  # -XX:MetaspaceSize
    max_metaspace_size_bytes: int | None = None  # -XX:MaxMetaspaceSize
    max_direct_memory_size_bytes: int | None = None  # -XX:MaxDirectMemorySize

    # GC Threading
    parallel_gc_threads: int | None = None  # -XX:ParallelGCThreads
    conc_gc_threads: int | None = None  # -XX:ConcGCThreads

    # GC Behavior Tuning
    max_gc_pause_millis: int | None = None  # -XX:MaxGCPauseMillis
    gc_time_ratio: int | None = None  # -XX:GCTimeRatio
    adaptive_size_policy_weight: int | None = None  # -XX:AdaptiveSizePolicyWeight
    use_adaptive_size_policy: bool | None = None  # -XX:+UseAdaptiveSizePolicy
    use_adaptive_size_policy_footprint_goal: bool | None = (
        None  # -XX:+UseAdaptiveSizePolicyFootprintGoal
    )

    # G1-specific
    g1_heap_region_size_bytes: int | None = None  # -XX:G1HeapRegionSize
    initiating_heap_occupancy_percent: int | None = None  # -XX:InitiatingHeapOccupancyPercent
    g1_reserve_percent: int | None = None  # -XX:G1ReservePercent
    g1_new_size_percent: int | None = None  # -XX:G1NewSizePercent
    g1_max_new_size_percent: int | None = None  # -XX:G1MaxNewSizePercent
    g1_mixed_gc_count_target: int | None = None  # -XX:G1MixedGCCountTarget
    g1_heap_waste_percent: int | None = None  # -XX:G1HeapWastePercent
    g1_mixed_gc_live_threshold_percent: int | None = None  # -XX:G1MixedGCLiveThresholdPercent
    g1_rset_updating_pause_time_percent: int | None = None  # -XX:G1RSetUpdatingPauseTimePercent
    g1_old_cset_region_threshold_percent: int | None = None  # -XX:G1OldCSetRegionThresholdPercent

    # CMS-specific
    cms_initiating_occupancy_fraction: int | None = None  # -XX:CMSInitiatingOccupancyFraction
    use_cms_initiating_occupancy_only: bool | None = None  # -XX:+UseCMSInitiatingOccupancyOnly
    cms_full_gcs_before_compaction: int | None = None  # -XX:CMSFullGCsBeforeCompaction
    cms_wait_duration: int | None = None  # -XX:CMSWaitDuration

    def format_size(self, size_bytes: int | None) -> str:
        """Format bytes to human-readable size."""
        if size_bytes is None:
            return "Not set"

        # Convert to appropriate unit
        if size_bytes >= 1024**3:  # GB
            return f"{size_bytes / (1024**3):.1f}G"
        elif size_bytes >= 1024**2:  # MB
            return f"{size_bytes / (1024**2):.1f}M"
        elif size_bytes >= 1024:  # KB
            return f"{size_bytes / 1024:.1f}K"
        return f"{size_bytes}B"


class G1Metrics(BaseModel):
    """G1 GC-specific diagnostic metrics."""

    mixed_gc_count: int
    mixed_gc_frequency_per_hour: float
    average_mixed_gc_pause_seconds: float

    humongous_allocation_count: int
    evacuation_failure_count: int
    to_space_exhausted_count: int

    g1_warnings: list[str] = Field(default_factory=list)


class CMSMetrics(BaseModel):
    """CMS GC-specific diagnostic metrics."""

    model_config = ConfigDict(frozen=True)

    concurrent_mode_failures: int = 0
    promotion_failures: int = 0
    cms_cycle_frequency_per_hour: float = 0.0
    average_concurrent_pause_seconds: float = 0.0


class PeriodStatistics(BaseModel):
    """Statistics for a time period."""

    average_old_gen_percentage: float
    full_gc_count: int


class GCSummary(BaseModel):
    """Complete GC analysis summary."""

    # Runtime
    runtime_hours: float
    runtime_seconds: float
    total_event_count: int
    young_gc_count: int
    full_gc_count: int
    mixed_gc_count: int = 0
    remark_gc_count: int = 0
    cleanup_gc_count: int = 0

    # Frequency
    young_gc_per_hour: float
    full_gc_per_hour: float
    avg_full_gc_interval_min: float | None = None
    min_full_gc_interval_min: float | None = None
    time_to_first_full_gc_hours: float | None = None
    full_gc_accelerating: bool = False
    consecutive_full_gcs: int = 0

    # Overhead
    total_gc_time_sec: float
    gc_overhead_pct: float
    throughput_pct: float

    # Young GC pauses
    avg_young_pause: float
    max_young_pause: float
    median_young_pause: float
    p95_young_pause: float
    p99_young_pause: float

    # Full GC pauses
    avg_full_pause: float
    max_full_pause: float
    median_full_pause: float
    p95_full_pause: float
    p99_full_pause: float

    # Mixed GC pauses
    avg_mixed_pause: float = 0.0
    max_mixed_pause: float = 0.0
    median_mixed_pause: float = 0.0
    p95_mixed_pause: float = 0.0
    p99_mixed_pause: float = 0.0

    # Remark pauses (G1)
    avg_remark_pause: float = 0.0
    max_remark_pause: float = 0.0
    median_remark_pause: float = 0.0
    p95_remark_pause: float = 0.0
    p99_remark_pause: float = 0.0

    # Cleanup pauses (G1)
    avg_cleanup_pause: float = 0.0
    max_cleanup_pause: float = 0.0
    median_cleanup_pause: float = 0.0
    p95_cleanup_pause: float = 0.0
    p99_cleanup_pause: float = 0.0

    # Overall pauses
    p50_pause: float
    p95_pause: float
    p99_pause: float

    # Pause violations
    pauses_over_1s: int
    pauses_over_5s: int
    pauses_over_30s: int
    pauses_over_60s: int

    # Old gen
    old_start: float | None = None
    old_end: float | None = None
    old_growth: float = 0.0
    peak_old_pct: float = 0.0
    avg_old_pct: float = 0.0

    # Heap
    peak_heap_pct: float
    avg_heap_pct: float
    heap_headroom: float

    # Full GC effectiveness
    full_gc_reducing: int = 0
    full_gc_growing: int = 0
    avg_post_full_gc_pct: float | None = None
    avg_reclaimed_kb: float = 0.0
    reclamation_declining: bool = False

    # Young gen
    avg_promotion_kb: float = 0.0
    allocation_rate_mb_sec: float | None = None

    # Time-based
    period_stats: dict[str, PeriodStatistics] = Field(default_factory=dict)

    # Enhanced diagnostics
    leak_severity: LeakSeverity
    metaspace_analysis: MetaspaceAnalysis | None = None
    g1_metrics: G1Metrics | None = None
    cms_metrics: CMSMetrics | None = None
    jvm_heap_config: JVMHeapConfig | None = None

    # Warnings
    warnings: list[str] = Field(default_factory=list)


# ============================================================
# GC PARSER ABSTRACTION
# ============================================================


class GCParser(Protocol):
    """Protocol defining GC parser interface."""

    gc_type_name: str

    def parse(self, log_lines: list[str]) -> list[dict[str, Any]]:
        """Parse raw log lines into event dictionaries."""
        ...


class ParallelGCParser:
    """Parser for Parallel GC (PSYoungGen/ParOldGen) logs."""

    gc_type_name: str = "Parallel GC"

    YOUNG_GC_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+-]+):\s+"
        r"(?P<uptime>[\d.]+):.*?\[GC.*?\[PSYoungGen:\s+"
        r"(?P<young_before>\d+)K->(?P<young_after>\d+)K\((?P<young_total>\d+)K\)\]\s+"
        r"(?P<heap_before>\d+)K->(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\).*?,\s+"
        r"(?P<pause>[\d.]+)\s+secs"
    )

    FULL_GC_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+-]+):\s+"
        r"(?P<uptime>[\d.]+):.*?\[Full GC.*?\[PSYoungGen:\s+"
        r"(?P<young_before>\d+)K->(?P<young_after>\d+)K\((?P<young_total>\d+)K\)\]\s+"
        r"\[ParOldGen:\s+(?P<old_before>\d+)K->(?P<old_after>\d+)K\((?P<old_total>\d+)K\)\]\s+"
        r"(?P<heap_before>\d+)K->(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\).*?,\s+"
        r"(?P<pause>[\d.]+)\s+secs"
    )

    # Matches both PSPermGen (Java 8) and Metaspace (Java 9+)
    METASPACE_PATTERN: re.Pattern[str] = re.compile(
        r"\[(?:PSPermGen|Metaspace):\s+(?P<before>\d+)K->(?P<after>\d+)K\((?P<total>\d+)K\)\]"
    )

    def parse(self, log_lines: list[str]) -> list[dict[str, Any]]:
        """Parse Parallel GC logs into raw event dictionaries."""
        events: list[dict[str, Any]] = []

        for line in log_lines:
            # Substring guard: only run regex if "Full GC" present
            if "Full GC" in line and (match := self.FULL_GC_PATTERN.search(line)):
                event = self._extract_full_gc_event(match)
                # Check for metaspace/permgen with substring guard
                if ("Metaspace" in line or "PSPermGen" in line) and (
                    meta_match := self.METASPACE_PATTERN.search(line)
                ):
                    event["metaspace_before"] = int(meta_match.group("before"))
                    event["metaspace_after"] = int(meta_match.group("after"))
                    event["metaspace_total"] = int(meta_match.group("total"))
                events.append(event)
            # Substring guard: only run regex if "PSYoungGen" present
            elif "PSYoungGen" in line and (match := self.YOUNG_GC_PATTERN.search(line)):
                event = self._extract_young_gc_event(match)
                # Check for metaspace/permgen with substring guard
                if ("Metaspace" in line or "PSPermGen" in line) and (
                    meta_match := self.METASPACE_PATTERN.search(line)
                ):
                    event["metaspace_before"] = int(meta_match.group("before"))
                    event["metaspace_after"] = int(meta_match.group("after"))
                    event["metaspace_total"] = int(meta_match.group("total"))
                events.append(event)

        return events

    def _extract_young_gc_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract young GC event from regex match."""
        young_before = int(match.group("young_before"))
        young_after = int(match.group("young_after"))
        young_total = int(match.group("young_total"))
        heap_before = int(match.group("heap_before"))
        heap_after = int(match.group("heap_after"))
        heap_total = int(match.group("heap_total"))

        old_before, old_after, old_total = validate_old_gen_values(
            heap_before, heap_after, heap_total, young_before, young_after, young_total
        )

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Young",
            "pause": float(match.group("pause")),
            "young_before": young_before,
            "young_after": young_after,
            "young_total": young_total,
            "heap_before": heap_before,
            "heap_after": heap_after,
            "heap_total": heap_total,
            "old_before": old_before,
            "old_after": old_after,
            "old_total": old_total,
        }

    def _extract_full_gc_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract full GC event from regex match."""
        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Full",
            "pause": float(match.group("pause")),
            "young_before": int(match.group("young_before")),
            "young_after": int(match.group("young_after")),
            "young_total": int(match.group("young_total")),
            "old_before": int(match.group("old_before")),
            "old_after": int(match.group("old_after")),
            "old_total": int(match.group("old_total")),
            "heap_before": int(match.group("heap_before")),
            "heap_after": int(match.group("heap_after")),
            "heap_total": int(match.group("heap_total")),
        }


class G1GCParser:
    """Parser for G1 GC logs (both legacy and unified logging)."""

    gc_type_name: str = "G1 GC"

    # Unified logging patterns (Java 9+)
    UNIFIED_PAUSE_PATTERN: re.Pattern[str] = re.compile(
        r"\[(?P<timestamp>[\d-]+T[\d:.+-]+)\](?:\[(?P<uptime>[\d.]+)s\])?"
        r"(?:\[\d+\])?(?:\[\d+\])?\[info\]"  # Optional PID and TID fields
        r"\[gc[^\]]*\]\s+"
        r"GC\((?P<gc_id>\d+)\)\s+Pause\s+(?P<type>Young|Full|Remark|Cleanup)\s+"
        r"(?:\((?P<subtype>[^)]+)\)\s*)?"
        r"(?:\((?P<reason>[^)]+)\)\s*)?"
        r"(?P<heap_before>\d+)M->(?P<heap_after>\d+)M\((?P<heap_total>\d+)M\)\s+"
        r"(?P<pause>[\d.]+)ms"
    )

    UNIFIED_TO_SPACE_EXHAUSTED: re.Pattern[str] = re.compile(r"To-space exhausted")
    UNIFIED_HUMONGOUS: re.Pattern[str] = re.compile(r"Humongous")
    UNIFIED_METASPACE: re.Pattern[str] = re.compile(
        r"Metaspace:\s+(?P<before>\d+)K->(?P<after>\d+)K\((?P<total>\d+)K\)"
    )
    UNIFIED_REGION_SIZE: re.Pattern[str] = re.compile(
        r"GC\((?P<gc_id>\d+)\)\s+Heap region size:\s+(?P<size>\d+)M"
    )
    UNIFIED_REGION_COUNTS: re.Pattern[str] = re.compile(
        r"GC\((?P<gc_id>\d+)\)\s+(?P<region>Eden|Survivor|Old|Humongous) regions:\s+"
        r"(?P<before>\d+)->(?P<after>\d+)(?:\((?P<total>\d+)\))?"
    )
    UNIFIED_GC_ID: re.Pattern[str] = re.compile(r"GC\((?P<gc_id>\d+)\)")

    # Legacy format patterns (Java 8)
    LEGACY_PAUSE_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.+-]+):\s+"
        r"(?P<uptime>[\d.]+):\s+\[GC pause\s+\((?P<reason>[^)]+)\)\s+"
        r"(?:\((?P<type1>young|mixed)\)\s*)?"
        r"(?:\((?P<type2>initial-mark|remark)\)\s*)?,?\s+"
        r"(?P<pause>[\d.]+)\s+secs\]"
    )

    LEGACY_HEAP_PATTERN: re.Pattern[str] = re.compile(
        r"Heap:\s+(?P<heap_before>[\d.]+)M\((?P<heap_total>[\d.]+)M\)"
        r"->(?P<heap_after>[\d.]+)M\([\d.]+M\)"
    )

    # Combined Eden/Survivors/Heap pattern (common in Java 8 G1 logs)
    LEGACY_COMBINED_PATTERN: re.Pattern[str] = re.compile(
        r"\[Eden:\s+(?P<eden_before>[\d.]+[BKMG])\((?P<eden_total>[\d.]+[BKMG])\)->"
        r"(?P<eden_after>[\d.]+[BKMG])\((?P<eden_total_after>[\d.]+[BKMG])\)\s+"
        r"Survivors:\s+(?P<survivor_before>[\d.]+[BKMG])->(?P<survivor_after>[\d.]+[BKMG])\s+"
        r"Heap:\s+(?P<heap_before>[\d.]+[BKMG])\((?P<heap_total>[\d.]+[BKMG])\)->"
        r"(?P<heap_after>[\d.]+[BKMG])\((?P<heap_total_after>[\d.]+[BKMG])\)\]"
    )

    LEGACY_EDEN_PATTERN: re.Pattern[str] = re.compile(
        r"Eden:\s+(?P<before>[\d.]+[BKMG])\((?P<total>[\d.]+[BKMG])\)->"
        r"(?P<after>[\d.]+[BKMG])\((?P<after_total>[\d.]+[BKMG])\)"
    )
    LEGACY_SURVIVOR_PATTERN: re.Pattern[str] = re.compile(
        r"Survivors:\s+(?P<before>[\d.]+[BKMG])(?:\((?P<total>[\d.]+[BKMG])\))?"
        r"->(?P<after>[\d.]+[BKMG])(?:\((?P<after_total>[\d.]+[BKMG])\))?"
    )

    LEGACY_TO_SPACE_EXHAUSTED: re.Pattern[str] = re.compile(r"to-space\s+(exhausted|overflow)")

    def parse(self, log_lines: list[str]) -> list[dict[str, Any]]:
        """Parse G1 GC logs supporting both unified and legacy formats."""
        events: list[dict[str, Any]] = []
        current_event: dict[str, Any] | None = None
        events_by_id: dict[int, dict[str, Any]] = {}
        region_size_kb: int | None = None

        for line in log_lines:
            # Try unified logging first (substring guard: "Pause")
            if "Pause" in line:
                if match := self.UNIFIED_PAUSE_PATTERN.search(line):
                    event = self._extract_unified_pause(match, line)
                    events.append(event)
                    current_event = event
                    events_by_id[event["gc_id"]] = event

            # Try legacy format (substring guard: "GC pause")
            elif "GC pause" in line and (match := self.LEGACY_PAUSE_PATTERN.search(line)):
                current_event = self._extract_legacy_pause(match)

            # Check for combined Eden/Survivors/Heap line (common format)
            if (
                current_event
                and "[Eden:" in line
                and "Survivors:" in line
                and "Heap:" in line
                and (match := self.LEGACY_COMBINED_PATTERN.search(line))
            ):
                # Extract all values from the combined line
                current_event["eden_before_kb"] = parse_size_to_kb(match.group("eden_before"))
                current_event["eden_after_kb"] = parse_size_to_kb(match.group("eden_after"))
                current_event["eden_total_kb"] = parse_size_to_kb(match.group("eden_total"))

                current_event["survivor_before_kb"] = parse_size_to_kb(
                    match.group("survivor_before")
                )
                current_event["survivor_after_kb"] = parse_size_to_kb(match.group("survivor_after"))

                current_event["heap_before"] = parse_size_to_kb(match.group("heap_before"))
                current_event["heap_after"] = parse_size_to_kb(match.group("heap_after"))
                current_event["heap_total"] = parse_size_to_kb(match.group("heap_total"))

                self._apply_legacy_young_old_estimates(current_event)
                events.append(current_event)
                current_event = None
                continue

            # Check for heap info in legacy format (substring guard: "Heap:")
            elif (
                current_event
                and "Heap:" in line
                and (match := self.LEGACY_HEAP_PATTERN.search(line))
            ):
                current_event["heap_before"] = int(float(match.group("heap_before")) * 1024)
                current_event["heap_after"] = int(float(match.group("heap_after")) * 1024)
                current_event["heap_total"] = int(float(match.group("heap_total")) * 1024)
                self._apply_legacy_young_old_estimates(current_event)
                events.append(current_event)
                current_event = None

            # Legacy Eden/Survivor details (substring guards)
            if (
                current_event
                and "Eden:" in line
                and (eden_match := self.LEGACY_EDEN_PATTERN.search(line))
            ):
                current_event["eden_before_kb"] = parse_size_to_kb(eden_match.group("before"))
                current_event["eden_after_kb"] = parse_size_to_kb(eden_match.group("after"))
                current_event["eden_total_kb"] = parse_size_to_kb(eden_match.group("total"))
            if (
                current_event
                and "Survivors:" in line
                and (survivor_match := self.LEGACY_SURVIVOR_PATTERN.search(line))
            ):
                current_event["survivor_before_kb"] = parse_size_to_kb(
                    survivor_match.group("before")
                )
                current_event["survivor_after_kb"] = parse_size_to_kb(survivor_match.group("after"))
                if survivor_match.group("total"):
                    current_event["survivor_total_kb"] = parse_size_to_kb(
                        survivor_match.group("total")
                    )

            # Unified region size (substring guard: "Heap region size:")
            if "Heap region size:" in line and (
                region_match := self.UNIFIED_REGION_SIZE.search(line)
            ):
                region_size_kb = int(region_match.group("size")) * 1024
                target = events_by_id.get(int(region_match.group("gc_id")))
                if target:
                    target["region_size_kb"] = region_size_kb
                    self._apply_region_counts(target, region_size_kb)

            # Unified region counts (substring guard: " regions:")
            if " regions:" in line and (region_counts := self.UNIFIED_REGION_COUNTS.search(line)):
                target = events_by_id.get(int(region_counts.group("gc_id")))
                if target:
                    region_name = region_counts.group("region").lower()
                    target[f"{region_name}_before_regions"] = int(region_counts.group("before"))
                    target[f"{region_name}_after_regions"] = int(region_counts.group("after"))
                    if region_counts.group("total"):
                        target[f"{region_name}_total_regions"] = int(region_counts.group("total"))
                    if region_size_kb:
                        self._apply_region_counts(target, region_size_kb)

            # Check for G1-specific events (substring guards)
            if current_event or events:
                target = self._resolve_unified_target(line, current_event, events_by_id, events)
                if target:
                    if ("exhausted" in line or "overflow" in line) and (
                        self.UNIFIED_TO_SPACE_EXHAUSTED.search(line)
                        or self.LEGACY_TO_SPACE_EXHAUSTED.search(line)
                    ):
                        target["to_space_exhausted"] = True
                        target["evacuation_failed"] = True

                    if "Humongous" in line and self.UNIFIED_HUMONGOUS.search(line):
                        target["humongous_detected"] = True

                    if "Metaspace:" in line and (match := self.UNIFIED_METASPACE.search(line)):
                        target["metaspace_before"] = int(match.group("before"))
                        target["metaspace_after"] = int(match.group("after"))
                        target["metaspace_total"] = int(match.group("total"))

        return events

    def _resolve_unified_target(
        self,
        line: str,
        current_event: dict[str, Any] | None,
        events_by_id: dict[int, dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve the correct event for unified log lines using GC id when present."""
        if match := self.UNIFIED_GC_ID.search(line):
            gc_id = int(match.group("gc_id"))
            if gc_id in events_by_id:
                return events_by_id[gc_id]
        if current_event is not None:
            return current_event
        return events[-1] if events else None

    def _extract_unified_pause(self, match: re.Match[str], line: str) -> dict[str, Any]:
        """Extract pause from unified logging format."""
        pause_type = match.group("type")
        subtype = match.group("subtype")

        # Determine GC kind
        if pause_type == "Full":
            gc_kind: GCKind = "Full"
        elif pause_type == "Remark":
            gc_kind = "Remark"
        elif pause_type == "Cleanup":
            gc_kind = "Cleanup"
        elif subtype and "Mixed" in subtype:
            gc_kind = "Mixed"
        else:
            gc_kind = "Young"

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")) if match.group("uptime") else 0.0,
            "gc_kind": gc_kind,
            "pause": float(match.group("pause")) / 1000,  # Convert ms to seconds
            "heap_before": int(match.group("heap_before")) * 1024,  # MB to KB
            "heap_after": int(match.group("heap_after")) * 1024,
            "heap_total": int(match.group("heap_total")) * 1024,
            "gc_id": int(match.group("gc_id")),
            "humongous_detected": "Humongous" in line,
            "to_space_exhausted": False,
            "evacuation_failed": False,
        }

    def _extract_legacy_pause(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract pause from legacy format."""
        type1 = match.group("type1")
        type2 = match.group("type2")

        # Determine GC kind - type2 takes precedence for mixed/initial-mark
        if type1 == "mixed" or (type2 and "initial-mark" not in type2):
            gc_kind: GCKind = "Mixed"
        elif type2 == "initial-mark":
            gc_kind = "Young"  # initial-mark is part of a young collection
        elif "Full GC" in match.group(0):
            gc_kind = "Full"
        else:
            gc_kind = "Young"

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": gc_kind,
            "pause": float(match.group("pause")),
            "humongous_detected": "Humongous" in match.group("reason"),
            "to_space_exhausted": False,
            "evacuation_failed": False,
        }

    def _apply_region_counts(self, event: dict[str, Any], region_size_kb: int) -> None:
        """Apply region counts to compute young/old sizes when available."""
        eden_before = event.get("eden_before_regions")
        eden_after = event.get("eden_after_regions")
        survivor_before = event.get("survivor_before_regions")
        survivor_after = event.get("survivor_after_regions")
        eden_total = event.get("eden_total_regions")
        survivor_total = event.get("survivor_total_regions")

        if eden_before is not None and survivor_before is not None:
            event["young_before"] = (eden_before + survivor_before) * region_size_kb
        if eden_after is not None and survivor_after is not None:
            event["young_after"] = (eden_after + survivor_after) * region_size_kb
        if eden_total is not None and survivor_total is not None:
            event["young_total"] = (eden_total + survivor_total) * region_size_kb

        old_before = event.get("old_before_regions")
        old_after = event.get("old_after_regions")
        hum_before = event.get("humongous_before_regions") or 0
        hum_after = event.get("humongous_after_regions") or 0
        old_total = event.get("old_total_regions")

        if old_before is not None:
            event["old_before"] = (old_before + hum_before) * region_size_kb
        if old_after is not None:
            event["old_after"] = (old_after + hum_after) * region_size_kb
        if old_total is not None:
            event["old_total"] = old_total * region_size_kb

    def _apply_legacy_young_old_estimates(self, event: dict[str, Any]) -> None:
        """Derive young/old sizes from legacy Eden/Survivor/Heap lines when available."""
        if (
            "heap_before" not in event
            or "heap_after" not in event
            or "heap_total" not in event
            or "eden_before_kb" not in event
            or "eden_after_kb" not in event
            or "eden_total_kb" not in event
            or "survivor_before_kb" not in event
            or "survivor_after_kb" not in event
        ):
            return

        young_before = event["eden_before_kb"] + event["survivor_before_kb"]
        young_after = event["eden_after_kb"] + event["survivor_after_kb"]
        event["young_before"] = young_before
        event["young_after"] = young_after

        if "survivor_total_kb" in event:
            young_total = event["eden_total_kb"] + event["survivor_total_kb"]
            event["young_total"] = young_total
            event["old_total"] = max(0, event["heap_total"] - young_total)

        event["old_before"] = max(0, event["heap_before"] - young_before)
        event["old_after"] = max(0, event["heap_after"] - young_after)


class CMSGCParser:
    """Parser for CMS (Concurrent Mark Sweep) GC logs."""

    gc_type_name: str = "CMS GC"

    # CMS Young GC (ParNew) patterns
    PARNEW_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>[\d\-T:.+]+):\s+"
        r"(?P<uptime>[\d.]+):\s+"
        r"\[GC\s+\([\w\s]+\)\s+"
        r"\[ParNew:\s+(?P<young_before>\d+)K->(?P<young_after>\d+)K\((?P<young_total>\d+)K\),\s+"
        r"(?P<young_pause>[\d.]+)\s+secs\]\s+"
        r"(?P<heap_before>\d+)K->(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\),\s+"
        r"(?P<pause>[\d.]+)\s+secs\]"
    )

    # CMS Initial Mark (STW pause)
    INITIAL_MARK_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>[\d\-T:.+]+):\s+"
        r"(?P<uptime>[\d.]+):\s+"
        r"\[GC\s+\(CMS Initial Mark\)\s+"
        r"\[1 CMS-initial-mark:\s+(?P<old_after>\d+)K\((?P<old_total>\d+)K\)\]\s+"
        r"(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\),\s+"
        r"(?P<pause>[\d.]+)\s+secs\]"
    )

    # CMS Remark (STW pause)
    REMARK_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>[\d\-T:.+]+):\s+"
        r"(?P<uptime>[\d.]+):\s+"
        r"\[GC\s+\(CMS Final Remark\).*?"
        r"\[1 CMS-remark:\s+(?P<old_after>\d+)K\((?P<old_total>\d+)K\)\]\s+"
        r"(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\),\s+"
        r"(?P<pause>[\d.]+)\s+secs\]"
    )

    # CMS Full GC (fallback when concurrent mode fails)
    FULL_GC_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>[\d\-T:.+]+):\s+"
        r"(?P<uptime>[\d.]+):\s+"
        r"\[Full GC\s+\([\w\s]+\)\s+"
        r"(?:\[CMS:\s+(?P<old_before>\d+)K->(?P<old_after>\d+)K\((?P<old_total>\d+)K\),\s+"
        r"[\d.]+\s+secs\]\s+)?"
        r"(?P<heap_before>\d+)K->(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\),\s+"
        r"(?:\[Metaspace:\s+(?P<meta_before>\d+)K->(?P<meta_after>\d+)K\((?P<meta_total>\d+)K\)\],?\s*)?"
        r"(?P<pause>[\d.]+)\s+secs\]"
    )

    # Concurrent phases (non-STW, informational)
    CONCURRENT_PHASE_PATTERN: re.Pattern[str] = re.compile(
        r"(?P<timestamp>[\d\-T:.+]+):\s+"
        r"(?P<uptime>[\d.]+):\s+"
        r"\[CMS-concurrent-(?P<phase>[\w-]+)(?:-start|-end)?(?::\s+(?P<duration>[\d.]+)/[\d.]+\s+secs)?\]"
    )

    def parse(self, log_lines: list[str]) -> list[dict[str, Any]]:
        """Parse CMS GC log lines into structured events."""
        events: list[dict[str, Any]] = []

        for line in log_lines:
            # Try ParNew (Young GC) - substring guard: "ParNew"
            if "ParNew" in line:
                match = self.PARNEW_PATTERN.search(line)
                if match:
                    events.append(self._extract_parnew_event(match))
                    continue

            # Try Initial Mark - substring guard: "CMS Initial Mark"
            if "CMS Initial Mark" in line:
                match = self.INITIAL_MARK_PATTERN.search(line)
                if match:
                    events.append(self._extract_initial_mark_event(match))
                    continue

            # Try Remark - substring guard: "CMS Final Remark"
            if "CMS Final Remark" in line:
                match = self.REMARK_PATTERN.search(line)
                if match:
                    events.append(self._extract_remark_event(match))
                    continue

            # Try Full GC - substring guard: "Full GC"
            if "Full GC" in line:
                match = self.FULL_GC_PATTERN.search(line)
                if match:
                    events.append(self._extract_full_gc_event(match))
                    continue

            # Concurrent phases are non-STW, skip for now
            # (they don't contribute to pause time analysis)

        return events

    def _extract_parnew_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract ParNew (Young GC) event."""
        young_before = int(match.group("young_before"))
        young_after = int(match.group("young_after"))
        young_total = int(match.group("young_total"))
        heap_before = int(match.group("heap_before"))
        heap_after = int(match.group("heap_after"))
        heap_total = int(match.group("heap_total"))

        old_before, old_after, old_total = validate_old_gen_values(
            heap_before, heap_after, heap_total, young_before, young_after, young_total
        )

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Young",
            "pause": float(match.group("pause")),
            "young_before": young_before,
            "young_after": young_after,
            "young_total": young_total,
            "heap_before": heap_before,
            "heap_after": heap_after,
            "heap_total": heap_total,
            "old_before": old_before,
            "old_after": old_after,
            "old_total": old_total,
        }

    def _extract_initial_mark_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract CMS Initial Mark event (STW)."""
        old_after = int(match.group("old_after"))
        old_total = int(match.group("old_total"))
        heap_after = int(match.group("heap_after"))
        heap_total = int(match.group("heap_total"))

        # For CMS, old gen is directly reported, calculate young gen safely
        young_before = max(0, heap_after - old_after)
        young_after = max(0, heap_after - old_after)
        young_total = max(0, heap_total - old_total)

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Concurrent",  # Mark as Concurrent phase
            "pause": float(match.group("pause")),
            "heap_before": heap_after,  # Approximation
            "heap_after": heap_after,
            "heap_total": heap_total,
            "old_before": old_after,
            "old_after": old_after,
            "old_total": old_total,
            "young_before": young_before,
            "young_after": young_after,
            "young_total": young_total,
        }

    def _extract_remark_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract CMS Remark event (STW)."""
        old_after = int(match.group("old_after"))
        old_total = int(match.group("old_total"))
        heap_after = int(match.group("heap_after"))
        heap_total = int(match.group("heap_total"))

        # For CMS, old gen is directly reported, calculate young gen safely
        young_before = max(0, heap_after - old_after)
        young_after = max(0, heap_after - old_after)
        young_total = max(0, heap_total - old_total)

        return {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Concurrent",  # Mark as Concurrent phase
            "pause": float(match.group("pause")),
            "heap_before": heap_after,
            "heap_after": heap_after,
            "heap_total": heap_total,
            "old_before": old_after,
            "old_after": old_after,
            "old_total": old_total,
            "young_before": young_before,
            "young_after": young_after,
            "young_total": young_total,
        }

    def _extract_full_gc_event(self, match: re.Match[str]) -> dict[str, Any]:
        """Extract CMS Full GC event (concurrent mode failure)."""
        event: dict[str, Any] = {
            "timestamp": match.group("timestamp"),
            "uptime": float(match.group("uptime")),
            "gc_kind": "Full",
            "pause": float(match.group("pause")),
            "heap_before": int(match.group("heap_before")),
            "heap_after": int(match.group("heap_after")),
            "heap_total": int(match.group("heap_total")),
        }

        # Old gen stats if available
        if match.group("old_before"):
            event["old_before"] = int(match.group("old_before"))
            event["old_after"] = int(match.group("old_after"))
            event["old_total"] = int(match.group("old_total"))

        # Metaspace stats if available
        if match.group("meta_before"):
            event["metaspace_before"] = int(match.group("meta_before"))
            event["metaspace_after"] = int(match.group("meta_after"))
            event["metaspace_total"] = int(match.group("meta_total"))

        return event


def detect_gc_type_and_create_parser(log_lines: list[str]) -> GCParser:
    """Detect GC type and return appropriate parser."""
    sample = "".join(log_lines[:300])

    if "PSYoungGen" in sample or "ParOldGen" in sample:
        return ParallelGCParser()
    elif any(pattern in sample for pattern in ["G1 Evacuation", "[gc,", "G1Young", "G1Mixed"]):
        return G1GCParser()
    elif any(pattern in sample for pattern in ["ParNew", "CMS", "CMS-concurrent"]):
        return CMSGCParser()
    else:
        raise ValueError(
            "Unsupported or unrecognized GC log format. "
            "Supported formats: Parallel GC, G1 GC (unified and legacy), CMS GC"
        )


def validate_old_gen_values(
    heap_before: int,
    heap_after: int,
    heap_total: int,
    young_before: int,
    young_after: int,
    young_total: int,
) -> tuple[int, int, int]:
    """Validate and clamp old gen values to prevent negatives from corrupted log data."""
    old_before = max(0, heap_before - young_before)
    old_after = max(0, heap_after - young_after)
    old_total = max(0, heap_total - young_total)
    return old_before, old_after, old_total


# ============================================================
# EVENT PROCESSING
# ============================================================


def normalize_event(raw_event: dict[str, Any]) -> GCEvent | None:
    """Convert parser dict to validated Pydantic GCEvent."""
    # Parse timestamp
    timestamp_str = raw_event["timestamp"]
    # Handle various timestamp formats
    timestamp_str = timestamp_str.replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", timestamp_str):
        timestamp_str = f"{timestamp_str[:-2]}:{timestamp_str[-2:]}"
    if "+" not in timestamp_str and "-" not in timestamp_str[-6:]:
        timestamp_str += "+00:00"

    timestamp = datetime.fromisoformat(timestamp_str)

    # Calculate heap percentage
    heap_total = raw_event.get("heap_total", 0)
    heap_after = raw_event.get("heap_after", 0)

    if heap_total <= 0:
        return None

    # Calculate heap percentage with bounds checking
    heap_used_pct = (heap_after / heap_total) * 100
    heap_used_pct = max(0.0, min(100.0, heap_used_pct))  # Clamp to [0, 100]

    # Calculate old gen percentage if available
    old_total = raw_event.get("old_total")
    old_after = raw_event.get("old_after")
    old_used_pct = None
    if old_total and old_total > 0 and old_after is not None:
        old_used_pct = (old_after / old_total) * 100
        old_used_pct = max(0.0, min(100.0, old_used_pct))  # Clamp to [0, 100]

    # Validate pause time is positive
    pause_time = raw_event.get("pause", 0.0)
    if pause_time < 0:
        pause_time = 0.0

    return GCEvent(
        timestamp=timestamp,
        uptime_seconds=raw_event.get("uptime", 0.0),
        gc_kind=raw_event["gc_kind"],
        pause_time_seconds=pause_time,
        heap_before_kb=raw_event["heap_before"],
        heap_after_kb=heap_after,
        heap_total_kb=heap_total,
        heap_used_percentage=heap_used_pct,
        young_before_kb=raw_event.get("young_before"),
        young_after_kb=raw_event.get("young_after"),
        young_total_kb=raw_event.get("young_total"),
        old_before_kb=raw_event.get("old_before"),
        old_after_kb=raw_event.get("old_after"),
        old_total_kb=raw_event.get("old_total"),
        old_used_percentage=old_used_pct,
        humongous_detected=raw_event.get("humongous_detected", False),
        evacuation_failed=raw_event.get("evacuation_failed", False),
        to_space_exhausted=raw_event.get("to_space_exhausted", False),
        metaspace_before_kb=raw_event.get("metaspace_before"),
        metaspace_after_kb=raw_event.get("metaspace_after"),
        metaspace_total_kb=raw_event.get("metaspace_total"),
    )


def parse_size_to_kb(size_text: str) -> int:
    """Parse a JVM size token like '1024K', '1.5M', '0.0B' into KB."""
    match = re.fullmatch(r"(?P<value>[\d.]+)(?P<unit>[BKMG])", size_text.strip())
    if not match:
        raise ValueError(f"Unrecognized size token: {size_text}")
    value = float(match.group("value"))
    unit = match.group("unit")
    if unit == "B":
        return int(value / 1024)
    if unit == "K":
        return int(value)
    if unit == "M":
        return int(value * 1024)
    if unit == "G":
        return int(value * 1024 * 1024)
    raise ValueError(f"Unsupported size unit: {unit}")


def percentile_sorted(sorted_data: list[float], pct: float) -> float:
    """Calculate percentile from a pre-sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


# ============================================================
# DIAGNOSTIC CALCULATORS
# ============================================================


def calculate_leak_severity(summary: GCSummary, thresholds: DiagnosticThresholds) -> LeakSeverity:
    """Calculate memory leak severity score (0-100) based on multiple indicators."""
    score = 0.0
    indicators: list[str] = []
    has_old_gen_data = summary.old_start is not None and summary.old_end is not None

    if not has_old_gen_data:
        return LeakSeverity(
            score=0.0,
            severity_level="None",
            indicators=["Insufficient old-gen data for leak scoring"],
        )

    # Factor 1: Old gen monotonic growth (0-30 points)
    if has_old_gen_data:
        if summary.old_growth > thresholds.old_gen_growth_critical_percentage:
            score += 30.0
            indicators.append(f"Old gen grew {summary.old_growth:.1f} percentage points")
        elif summary.old_growth > thresholds.old_gen_growth_warning_percentage:
            score += 15.0
            indicators.append(f"Old gen grew {summary.old_growth:.1f} percentage points")

    # Factor 2: Reclamation declining (0-25 points)
    if summary.reclamation_declining:
        score += 25.0
        indicators.append("Memory reclamation declining over time")

    # Factor 3: Full GC acceleration (0-20 points)
    if summary.full_gc_accelerating:
        score += 20.0
        indicators.append("Full GC frequency accelerating")

    # Factor 4: High post-Full-GC residual (0-15 points)
    if summary.avg_post_full_gc_pct and summary.avg_post_full_gc_pct > 90:
        score += 15.0
        indicators.append(f"Old gen remains {summary.avg_post_full_gc_pct:.1f}% full after Full GC")
    elif summary.avg_post_full_gc_pct and summary.avg_post_full_gc_pct > 80:
        score += 8.0
        indicators.append(f"Old gen remains {summary.avg_post_full_gc_pct:.1f}% full after Full GC")

    # Factor 5: Heap headroom shrinking (0-10 points)
    if summary.heap_headroom < thresholds.heap_minimum_headroom_percentage:
        score += 10.0
        indicators.append(f"Only {summary.heap_headroom:.1f}% heap headroom remaining")
    elif summary.heap_headroom < 10:
        score += 5.0
        indicators.append(f"Only {summary.heap_headroom:.1f}% heap headroom remaining")

    # Determine severity level
    if score >= 80:
        level: Literal["None", "Low", "Medium", "High", "Critical"] = "Critical"
    elif score >= 60:
        level = "High"
    elif score >= 40:
        level = "Medium"
    elif score >= 20:
        level = "Low"
    else:
        level = "None"

    return LeakSeverity(
        score=score,
        severity_level=level,
        indicators=indicators,
    )


def analyze_metaspace(events: list[GCEvent]) -> MetaspaceAnalysis | None:
    """Analyze metaspace usage across GC events."""
    metaspace_events = [
        e for e in events if e.metaspace_before_kb is not None and e.metaspace_after_kb is not None
    ]

    if not metaspace_events:
        return None

    used_percentages: list[float] = [
        (e.metaspace_after_kb / e.metaspace_total_kb * 100)  # type: ignore[operator]
        for e in metaspace_events
        if e.metaspace_total_kb and e.metaspace_total_kb > 0
    ]

    first_used = metaspace_events[0].metaspace_after_kb or 0
    last_used = metaspace_events[-1].metaspace_after_kb or 0
    growth_kb = last_used - first_used

    warnings: list[str] = []
    if used_percentages:
        peak = max(used_percentages)
        if peak > 90:
            warnings.append(f"Metaspace usage peaked at {peak:.1f}%")
        if growth_kb > 50 * 1024:  # >50MB growth
            warnings.append(f"Metaspace grew by {growth_kb // 1024}MB")

    return MetaspaceAnalysis(
        metaspace_events_count=len(metaspace_events),
        average_metaspace_used_percentage=mean(used_percentages) if used_percentages else 0.0,
        peak_metaspace_used_percentage=max(used_percentages) if used_percentages else 0.0,
        metaspace_growth_kb=growth_kb,
        metaspace_growing=growth_kb > 10 * 1024,  # >10MB
        metaspace_warnings=warnings,
    )


def calculate_g1_metrics(events: list[GCEvent], runtime_hours: float) -> G1Metrics | None:
    """Calculate G1-specific metrics from events."""
    # Filter G1-related events
    mixed_gcs = [e for e in events if e.gc_kind == "Mixed"]
    humongous_events = [e for e in events if e.humongous_detected]
    evacuation_failures = [e for e in events if e.evacuation_failed]
    to_space_exhausted = [e for e in events if e.to_space_exhausted]

    # Only return metrics if we have G1-specific events
    if not any([mixed_gcs, humongous_events, evacuation_failures, to_space_exhausted]):
        return None

    warnings: list[str] = []

    if evacuation_failures:
        warnings.append(
            f"{len(evacuation_failures)} evacuation failures detected (heap fragmentation)"
        )

    if to_space_exhausted:
        warnings.append(
            f"{len(to_space_exhausted)} to-space exhaustion events (increase survivor space)"
        )

    if len(humongous_events) > len(events) * 0.1:  # >10% of events
        warnings.append("High frequency of humongous allocations (consider increasing region size)")

    return G1Metrics(
        mixed_gc_count=len(mixed_gcs),
        mixed_gc_frequency_per_hour=len(mixed_gcs) / runtime_hours if runtime_hours else 0.0,
        average_mixed_gc_pause_seconds=mean([e.pause_time_seconds for e in mixed_gcs])
        if mixed_gcs
        else 0.0,
        humongous_allocation_count=len(humongous_events),
        evacuation_failure_count=len(evacuation_failures),
        to_space_exhausted_count=len(to_space_exhausted),
        g1_warnings=warnings,
    )


def calculate_cms_metrics(events: list[GCEvent], runtime_hours: float) -> CMSMetrics | None:
    """Calculate CMS-specific metrics from events."""
    # Filter CMS-related events
    concurrent_events = [e for e in events if e.gc_kind == "Concurrent"]

    # Only return metrics if we have CMS concurrent events (distinguishes CMS from other collectors)
    if not concurrent_events:
        return None

    # Count CMS cycles (pairs of Initial Mark + Remark events)
    # Approximate as half the concurrent events (since each cycle has multiple phases)
    cms_cycles = len(concurrent_events) // 2

    # Calculate average concurrent pause time
    avg_concurrent_pause = (
        mean([e.pause_time_seconds for e in concurrent_events]) if concurrent_events else 0.0
    )

    # LIMITATION: concurrent_mode_failures and promotion_failures require parsing
    # the GC cause from log text (e.g., "concurrent mode failure", "promotion failed"),
    # which isn't currently captured in the GCEvent model.
    #
    # GC causes (concurrent mode failure, promotion failure) are not parsed.
    # To avoid incorrect reporting, do not infer these counts.
    concurrent_mode_failures = 0
    promotion_failures = 0

    return CMSMetrics(
        concurrent_mode_failures=concurrent_mode_failures,
        promotion_failures=promotion_failures,
        cms_cycle_frequency_per_hour=cms_cycles / runtime_hours if runtime_hours else 0.0,
        average_concurrent_pause_seconds=avg_concurrent_pause,
    )


def parse_jvm_size_to_bytes(value: str, unit: str | None) -> int:
    """Convert JVM size notation to bytes.

    Args:
        value: Numeric value as string
        unit: Unit suffix (k/m/g or None for bytes)

    Returns:
        Size in bytes
    """
    num = int(value)
    if unit is None:
        return num

    unit_lower = unit.lower()
    if unit_lower == "k":
        return num * 1024
    elif unit_lower == "m":
        return num * 1024 * 1024
    elif unit_lower == "g":
        return num * 1024 * 1024 * 1024
    return num


def parse_jvm_heap_config(log_lines: list[str]) -> JVMHeapConfig | None:
    """Extract JVM heap and GC tuning arguments from CommandLine flags.

    Args:
        log_lines: Raw GC log lines

    Returns:
        JVMHeapConfig if CommandLine flags found, None otherwise
    """
    # Pattern to match CommandLine flags line
    commandline_pattern = re.compile(r"CommandLine flags:\s*(.+)")

    # Heap-related flag patterns (byte values)
    byte_patterns = {
        "initial_heap_size_bytes": re.compile(r"-XX:InitialHeapSize=(\d+)"),
        "max_heap_size_bytes": re.compile(r"-XX:MaxHeapSize=(\d+)"),
        "new_size_bytes": re.compile(r"-XX:NewSize=(\d+)"),
        "max_new_size_bytes": re.compile(r"-XX:MaxNewSize=(\d+)"),
        "old_size_bytes": re.compile(r"-XX:OldSize=(\d+)"),
        "max_ram_bytes": re.compile(r"-XX:MaxRAM=(\d+)"),
        "soft_max_heap_size_bytes": re.compile(r"-XX:SoftMaxHeapSize=(\d+)"),
        "metaspace_size_bytes": re.compile(r"-XX:MetaspaceSize=(\d+)"),
        "max_metaspace_size_bytes": re.compile(r"-XX:MaxMetaspaceSize=(\d+)"),
        "max_direct_memory_size_bytes": re.compile(r"-XX:MaxDirectMemorySize=(\d+)"),
        "g1_heap_region_size_bytes": re.compile(r"-XX:G1HeapRegionSize=(\d+)"),
    }

    # Integer/percentage patterns
    int_patterns = {
        "new_ratio": re.compile(r"-XX:NewRatio=(\d+)"),
        "survivor_ratio": re.compile(r"-XX:SurvivorRatio=(\d+)"),
        "initial_survivor_ratio": re.compile(r"-XX:InitialSurvivorRatio=(\d+)"),
        "target_survivor_ratio": re.compile(r"-XX:TargetSurvivorRatio=(\d+)"),
        "max_tenuring_threshold": re.compile(r"-XX:MaxTenuringThreshold=(\d+)"),
        "max_heap_free_ratio": re.compile(r"-XX:MaxHeapFreeRatio=(\d+)"),
        "min_heap_free_ratio": re.compile(r"-XX:MinHeapFreeRatio=(\d+)"),
        "parallel_gc_threads": re.compile(r"-XX:ParallelGCThreads=(\d+)"),
        "conc_gc_threads": re.compile(r"-XX:ConcGCThreads=(\d+)"),
        "max_gc_pause_millis": re.compile(r"-XX:MaxGCPauseMillis=(\d+)"),
        "gc_time_ratio": re.compile(r"-XX:GCTimeRatio=(\d+)"),
        "adaptive_size_policy_weight": re.compile(r"-XX:AdaptiveSizePolicyWeight=(\d+)"),
        "initial_ram_percentage": re.compile(r"-XX:InitialRAMPercentage=(\d+)"),
        "min_ram_percentage": re.compile(r"-XX:MinRAMPercentage=(\d+)"),
        "max_ram_percentage": re.compile(r"-XX:MaxRAMPercentage=(\d+)"),
        "max_ram_fraction": re.compile(r"-XX:MaxRAMFraction=(\d+)"),
        "thread_stack_size_kb": re.compile(r"-XX:ThreadStackSize=(\d+)"),
        "initiating_heap_occupancy_percent": re.compile(
            r"-XX:InitiatingHeapOccupancyPercent=(\d+)"
        ),
        "g1_reserve_percent": re.compile(r"-XX:G1ReservePercent=(\d+)"),
        "g1_new_size_percent": re.compile(r"-XX:G1NewSizePercent=(\d+)"),
        "g1_max_new_size_percent": re.compile(r"-XX:G1MaxNewSizePercent=(\d+)"),
        "g1_mixed_gc_count_target": re.compile(r"-XX:G1MixedGCCountTarget=(\d+)"),
        "g1_heap_waste_percent": re.compile(r"-XX:G1HeapWastePercent=(\d+)"),
        "g1_mixed_gc_live_threshold_percent": re.compile(
            r"-XX:G1MixedGCLiveThresholdPercent=(\d+)"
        ),
        "g1_rset_updating_pause_time_percent": re.compile(
            r"-XX:G1RSetUpdatingPauseTimePercent=(\d+)"
        ),
        "g1_old_cset_region_threshold_percent": re.compile(
            r"-XX:G1OldCSetRegionThresholdPercent=(\d+)"
        ),
        "cms_initiating_occupancy_fraction": re.compile(
            r"-XX:CMSInitiatingOccupancyFraction=(\d+)"
        ),
        "cms_full_gcs_before_compaction": re.compile(r"-XX:CMSFullGCsBeforeCompaction=(\d+)"),
        "cms_wait_duration": re.compile(r"-XX:CMSWaitDuration=(\d+)"),
    }

    bool_patterns = {
        "always_pre_touch": re.compile(r"-XX:\+AlwaysPreTouch"),
        "use_adaptive_size_policy": re.compile(r"-XX:\+UseAdaptiveSizePolicy"),
        "use_adaptive_size_policy_footprint_goal": re.compile(
            r"-XX:\+UseAdaptiveSizePolicyFootprintGoal"
        ),
        "use_cms_initiating_occupancy_only": re.compile(r"-XX:\+UseCMSInitiatingOccupancyOnly"),
    }

    # Also support -Xms, -Xmx, -Xmn formats
    xms_pattern = re.compile(r"-Xms(\d+)([kmgKMG])?")
    xmx_pattern = re.compile(r"-Xmx(\d+)([kmgKMG])?")
    xmn_pattern = re.compile(r"-Xmn(\d+)([kmgKMG])?")

    # Search for CommandLine flags
    for line in log_lines:
        if match := commandline_pattern.search(line):
            flags_str = match.group(1)
            config_dict: dict[str, Any] = {}

            # Parse byte-value flags
            for field, pattern in byte_patterns.items():
                if field_match := pattern.search(flags_str):
                    value = int(field_match.group(1))
                    config_dict[field] = value

            # Parse integer/percentage flags
            for field, pattern in int_patterns.items():
                if field_match := pattern.search(flags_str):
                    value = int(field_match.group(1))
                    config_dict[field] = value

            # Parse -Xms, -Xmx, -Xmn (convert to bytes, override if present)
            if xms := xms_pattern.search(flags_str):
                config_dict["initial_heap_size_bytes"] = parse_jvm_size_to_bytes(
                    xms.group(1), xms.group(2)
                )
            if xmx := xmx_pattern.search(flags_str):
                config_dict["max_heap_size_bytes"] = parse_jvm_size_to_bytes(
                    xmx.group(1), xmx.group(2)
                )
            if xmn := xmn_pattern.search(flags_str):
                config_dict["new_size_bytes"] = parse_jvm_size_to_bytes(xmn.group(1), xmn.group(2))

            # Parse boolean flags
            for field, pattern in bool_patterns.items():
                if pattern.search(flags_str):
                    config_dict[field] = True

            if config_dict:
                return JVMHeapConfig(**config_dict)

    # If no CommandLine flags found, try unified logging format (Java 9+)
    # Patterns for unified logging (G1 GC with -Xlog:gc*)
    unified_heap_size_pattern = re.compile(
        r"Heap address:.*?size:\s*(\d+)\s*(MB|GB|KB|B)?", re.IGNORECASE
    )
    unified_region_size_pattern = re.compile(r"Heap region size:\s*(\d+)\s*(MB|KB|M|K)?")
    unified_workers_pattern = re.compile(r"Using\s+(\d+)\s+workers?\s+of\s+(\d+)")

    unified_config: dict[str, Any] = {}

    for line in log_lines:
        # Parse heap size from unified format
        if match := unified_heap_size_pattern.search(line):
            size_value = int(match.group(1))
            size_unit = match.group(2) if match.group(2) else "MB"
            # Convert to bytes
            if size_unit.upper() in ("MB", "M"):
                unified_config["max_heap_size_bytes"] = size_value * 1024 * 1024
                unified_config["initial_heap_size_bytes"] = size_value * 1024 * 1024
            elif size_unit.upper() in ("GB", "G"):
                unified_config["max_heap_size_bytes"] = size_value * 1024 * 1024 * 1024
                unified_config["initial_heap_size_bytes"] = size_value * 1024 * 1024 * 1024
            elif size_unit.upper() in ("KB", "K"):
                unified_config["max_heap_size_bytes"] = size_value * 1024
                unified_config["initial_heap_size_bytes"] = size_value * 1024
            else:
                unified_config["max_heap_size_bytes"] = size_value
                unified_config["initial_heap_size_bytes"] = size_value

        # Parse G1 region size
        if match := unified_region_size_pattern.search(line):
            size_value = int(match.group(1))
            size_unit = match.group(2) if match.group(2) else "M"
            # Convert to bytes
            if size_unit.upper() in ("MB", "M"):
                unified_config["g1_heap_region_size_bytes"] = size_value * 1024 * 1024
            elif size_unit.upper() in ("KB", "K"):
                unified_config["g1_heap_region_size_bytes"] = size_value * 1024

        # Parse GC threads (use the max workers available)
        if match := unified_workers_pattern.search(line):
            total_workers = int(match.group(2))
            if "parallel_gc_threads" not in unified_config:
                unified_config["parallel_gc_threads"] = total_workers

    if unified_config:
        return JVMHeapConfig(**unified_config)

    return None


def build_comprehensive_summary(
    events: list[GCEvent], thresholds: DiagnosticThresholds, log_lines: list[str]
) -> GCSummary:
    """Build comprehensive GC summary with enhanced diagnostics."""
    pauses_young = [e.pause_time_seconds for e in events if e.gc_kind == "Young"]
    pauses_full = [e.pause_time_seconds for e in events if e.gc_kind == "Full"]
    pauses_mixed = [e.pause_time_seconds for e in events if e.gc_kind == "Mixed"]
    pauses_remark = [e.pause_time_seconds for e in events if e.gc_kind == "Remark"]
    pauses_cleanup = [e.pause_time_seconds for e in events if e.gc_kind == "Cleanup"]
    pauses_all = [e.pause_time_seconds for e in events if e.pause_time_seconds > 0]
    pauses_all_sorted = sorted(pauses_all)
    pauses_young_sorted = sorted(pauses_young)
    pauses_full_sorted = sorted(pauses_full)
    pauses_mixed_sorted = sorted(pauses_mixed)
    pauses_remark_sorted = sorted(pauses_remark)
    pauses_cleanup_sorted = sorted(pauses_cleanup)
    old = [e.old_used_percentage for e in events if e.old_used_percentage is not None]
    heap = [e.heap_used_percentage for e in events]

    young_events = [e for e in events if e.gc_kind == "Young"]
    full_events = [e for e in events if e.gc_kind == "Full"]

    # Runtime (timestamp-based)
    runtime_hours = 0.0
    runtime_seconds = 0.0
    timestamps = [e.timestamp for e in events]
    if len(timestamps) >= 2:
        runtime_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        runtime_hours = runtime_seconds / 3600

    # GC Frequency & Rate
    young_gc_per_hour = len(pauses_young) / runtime_hours if runtime_hours else 0.0
    full_gc_per_hour = len(pauses_full) / runtime_hours if runtime_hours else 0.0

    # Time between Full GCs
    full_gc_intervals: list[float] = []
    if len(full_events) >= 2:
        for i in range(1, len(full_events)):
            interval_minutes: float = (
                full_events[i].timestamp - full_events[i - 1].timestamp
            ).total_seconds() / 60
            full_gc_intervals.append(interval_minutes)

    avg_full_gc_interval = mean(full_gc_intervals) if full_gc_intervals else None
    min_full_gc_interval = min(full_gc_intervals) if full_gc_intervals else None

    # Detect if Full GCs are accelerating
    full_gc_accelerating = False
    if len(full_gc_intervals) >= 3:
        first_half_avg = mean(full_gc_intervals[: len(full_gc_intervals) // 2])
        second_half_avg = mean(full_gc_intervals[len(full_gc_intervals) // 2 :])
        if second_half_avg < first_half_avg * thresholds.full_gc_acceleration_threshold:
            full_gc_accelerating = True

    # Time to first Full GC
    time_to_first_full_gc = None
    if full_events and runtime_hours:
        time_to_first_full_gc = (full_events[0].timestamp - timestamps[0]).total_seconds() / 3600

    # GC Overhead / Throughput
    total_gc_time = sum(pauses_all)
    gc_overhead_pct = (total_gc_time / runtime_seconds * 100) if runtime_seconds else 0.0
    throughput_pct = max(0.0, 100 - gc_overhead_pct)  # Ensure non-negative

    # Pause Distribution & Percentiles
    p50_pause = percentile_sorted(pauses_all_sorted, 50)
    p95_pause = percentile_sorted(pauses_all_sorted, 95)
    p99_pause = percentile_sorted(pauses_all_sorted, 99)

    p95_young = percentile_sorted(pauses_young_sorted, 95)
    p99_young = percentile_sorted(pauses_young_sorted, 99)

    p95_full = percentile_sorted(pauses_full_sorted, 95)
    p99_full = percentile_sorted(pauses_full_sorted, 99)

    p95_mixed = percentile_sorted(pauses_mixed_sorted, 95)
    p99_mixed = percentile_sorted(pauses_mixed_sorted, 99)

    p95_remark = percentile_sorted(pauses_remark_sorted, 95)
    p99_remark = percentile_sorted(pauses_remark_sorted, 99)

    p95_cleanup = percentile_sorted(pauses_cleanup_sorted, 95)
    p99_cleanup = percentile_sorted(pauses_cleanup_sorted, 99)

    # Pause threshold violations
    pauses_over_1s = sum(1 for p in pauses_all if p > 1.0)
    pauses_over_5s = sum(1 for p in pauses_all if p > 5.0)
    pauses_over_30s = sum(1 for p in pauses_all if p > 30.0)
    pauses_over_60s = sum(1 for p in pauses_all if p > 60.0)

    # Consecutive Full GCs (track maximum streak within 5 minutes)
    max_consecutive_full_gcs = 1
    current_streak = 1
    if len(full_events) >= 2:
        for i in range(1, len(full_events)):
            gap_seconds = (full_events[i].timestamp - full_events[i - 1].timestamp).total_seconds()
            if gap_seconds < 300:  # 5 minutes
                current_streak += 1
                max_consecutive_full_gcs = max(max_consecutive_full_gcs, current_streak)
            else:
                current_streak = 1
    consecutive_full_gcs = max_consecutive_full_gcs if max_consecutive_full_gcs > 1 else 0

    # Full GC Effectiveness
    full_gc_reducing = 0
    full_gc_growing = 0
    valid_post_pcts: list[float] = []
    reclaimed_amounts: list[float] = []

    for e in full_events:
        if e.old_before_kb is not None and e.old_after_kb is not None:
            delta = e.old_after_kb - e.old_before_kb
            reclaimed = e.old_before_kb - e.old_after_kb

            if delta < 0:
                full_gc_reducing += 1
            if delta > 0:
                full_gc_growing += 1

            if reclaimed > 0:
                reclaimed_amounts.append(reclaimed)

            if e.old_used_percentage is not None:
                valid_post_pcts.append(e.old_used_percentage)

    avg_post_full_gc_pct = mean(valid_post_pcts) if valid_post_pcts else None
    avg_reclaimed_kb = mean(reclaimed_amounts) if reclaimed_amounts else 0.0

    # Check if reclamation is trending down
    reclamation_declining = False
    if len(reclaimed_amounts) >= 4:
        first_half = mean(reclaimed_amounts[: len(reclaimed_amounts) // 2])
        second_half = mean(reclaimed_amounts[len(reclaimed_amounts) // 2 :])
        if second_half < first_half * thresholds.reclamation_decline_threshold:
            reclamation_declining = True

    # Heap Utilization & Trends
    peak_heap_pct = max(heap) if heap else 0.0
    avg_heap_pct = mean(heap) if heap else 0.0
    heap_headroom = 100 - peak_heap_pct

    peak_old_pct = max(old) if old else 0.0
    avg_old_pct = mean(old) if old else 0.0

    # Old gen growth (percentage point change)
    old_start = old[0] if old else None
    old_end = old[-1] if old else None
    old_growth = (old_end - old_start) if old_start is not None and old_end is not None else 0.0

    # Young Gen Analysis
    promotions: list[int] = []
    for e in young_events:
        if e.old_before_kb is not None and e.old_after_kb is not None:
            promoted: int = e.old_after_kb - e.old_before_kb
            if promoted > 0:
                promotions.append(promoted)

    avg_promotion_kb = mean(promotions) if promotions else 0.0

    # Allocation rate estimate
    allocation_rate_mb_sec = None
    if len(young_events) >= 2 and runtime_seconds:
        collected_samples: list[int] = []
        for e in young_events:
            if e.young_before_kb is None or e.young_after_kb is None:
                continue
            collected = e.young_before_kb - e.young_after_kb
            if collected >= 0:
                collected_samples.append(collected)
        if collected_samples:
            total_young_collected = sum(collected_samples)
            allocation_rate_mb_sec = (total_young_collected / 1024) / runtime_seconds

    # Time-based Analysis
    period_stats: dict[str, PeriodStatistics] = {}
    if len(events) >= 3 and len(timestamps) >= 2:
        # Use time-based periods instead of count-based for better statistical accuracy
        runtime_span = timestamps[-1] - timestamps[0]
        period_duration = runtime_span / 3

        early_end = timestamps[0] + period_duration
        middle_end = early_end + period_duration

        time_periods = {
            "early": [e for e in events if e.timestamp < early_end],
            "middle": [e for e in events if early_end <= e.timestamp < middle_end],
            "late": [e for e in events if e.timestamp >= middle_end],
        }

        for period_name, period_events in time_periods.items():
            if period_events:
                period_old = [
                    e.old_used_percentage
                    for e in period_events
                    if e.old_used_percentage is not None
                ]
                period_full_gcs = [e for e in period_events if e.gc_kind == "Full"]
                period_stats[period_name] = PeriodStatistics(
                    average_old_gen_percentage=mean(period_old) if period_old else 0.0,
                    full_gc_count=len(period_full_gcs),
                )

    # Warning Flags
    warnings: list[str] = []

    # Warn if log is too short for reliable trend analysis
    if runtime_seconds < 60:
        warnings.append(
            f"⚠️ WARNING: Log runtime only {runtime_seconds:.0f} seconds - "
            "trend analysis unreliable with <1 minute of data"
        )

    if not old:
        warnings.append(
            "⚠️ WARNING: Old generation utilization not available in log data - "
            "leak detection and old-gen trend metrics are limited"
        )

    if peak_heap_pct > thresholds.heap_critical_percentage:
        warnings.append(
            f"⚠️ CRITICAL: Peak heap utilization {peak_heap_pct:.1f}% "
            f"(threshold: {thresholds.heap_critical_percentage:.0f}%) - "
            "extreme memory pressure, OOM risk imminent"
        )
    elif peak_heap_pct > thresholds.heap_warning_percentage:
        warnings.append(
            f"⚠️ WARNING: Peak heap utilization {peak_heap_pct:.1f}% "
            f"(threshold: {thresholds.heap_warning_percentage:.0f}%) - "
            "high memory pressure detected"
        )

    if full_gc_per_hour > thresholds.full_gc_critical_per_hour:
        warnings.append(
            f"⚠️ CRITICAL: {full_gc_per_hour:.0f} Full GCs per hour "
            f"(threshold: {thresholds.full_gc_critical_per_hour:.0f}) - "
            "indicates severe memory pressure or undersized heap"
        )
    elif full_gc_per_hour > thresholds.full_gc_warning_per_hour:
        warnings.append(
            f"⚠️ WARNING: {full_gc_per_hour:.1f} Full GCs per hour "
            f"(threshold: {thresholds.full_gc_warning_per_hour:.0f}) - "
            "frequent full collections may impact performance"
        )

    if gc_overhead_pct > thresholds.gc_overhead_critical_percentage:
        warnings.append(
            f"⚠️ CRITICAL: GC overhead {gc_overhead_pct:.1f}% "
            f"(threshold: {thresholds.gc_overhead_critical_percentage:.0f}%) - "
            "application spending excessive time in GC"
        )
    elif gc_overhead_pct > thresholds.gc_overhead_warning_percentage:
        warnings.append(
            f"⚠️ WARNING: GC overhead {gc_overhead_pct:.1f}% "
            f"(threshold: {thresholds.gc_overhead_warning_percentage:.0f}%) - "
            "GC impacting throughput"
        )

    if p99_pause > thresholds.pause_critical_seconds:
        warnings.append(
            f"⚠️ CRITICAL: P99 pause time {p99_pause:.3f}s "
            f"(threshold: {thresholds.pause_critical_seconds:.0f}s) - "
            "severe latency impact on application responsiveness"
        )
    elif p99_pause > thresholds.pause_warning_seconds:
        warnings.append(
            f"⚠️ WARNING: P99 pause time {p99_pause:.3f}s "
            f"(threshold: {thresholds.pause_warning_seconds:.0f}s) - "
            "noticeable pause times may affect user experience"
        )

    if consecutive_full_gcs > 1:
        warnings.append(
            f"⚠️ CRITICAL: Consecutive Full GC streak of {consecutive_full_gcs} events "
            "within 5 minutes - indicates severe memory thrashing"
        )

    if full_gc_accelerating:
        warnings.append(
            "⚠️ CRITICAL: Full GC frequency accelerating over time - "
            "indicates worsening memory pressure or leak"
        )

    if reclamation_declining:
        warnings.append(
            "⚠️ CRITICAL: Memory reclamation declining over time - "
            "Full GCs recovering less memory, possible leak or fragmentation"
        )

    # Low heap headroom warning
    if heap_headroom < 20:
        warnings.append(
            f"⚠️ WARNING: Low heap headroom ({heap_headroom:.1f}%) - "
            "consider increasing -Xmx if memory leaks are ruled out"
        )

    # Create initial summary (without leak severity yet)
    summary = GCSummary(
        runtime_hours=runtime_hours,
        runtime_seconds=runtime_seconds,
        total_event_count=len(events),
        young_gc_count=len(pauses_young),
        full_gc_count=len(pauses_full),
        mixed_gc_count=len(pauses_mixed),
        remark_gc_count=len(pauses_remark),
        cleanup_gc_count=len(pauses_cleanup),
        young_gc_per_hour=young_gc_per_hour,
        full_gc_per_hour=full_gc_per_hour,
        avg_full_gc_interval_min=avg_full_gc_interval,
        min_full_gc_interval_min=min_full_gc_interval,
        time_to_first_full_gc_hours=time_to_first_full_gc,
        full_gc_accelerating=full_gc_accelerating,
        consecutive_full_gcs=consecutive_full_gcs,
        total_gc_time_sec=total_gc_time,
        gc_overhead_pct=gc_overhead_pct,
        throughput_pct=throughput_pct,
        avg_young_pause=mean(pauses_young) if pauses_young else 0.0,
        max_young_pause=max(pauses_young) if pauses_young else 0.0,
        median_young_pause=percentile_sorted(pauses_young_sorted, 50),
        p95_young_pause=p95_young,
        p99_young_pause=p99_young,
        avg_full_pause=mean(pauses_full) if pauses_full else 0.0,
        max_full_pause=max(pauses_full) if pauses_full else 0.0,
        median_full_pause=percentile_sorted(pauses_full_sorted, 50),
        p95_full_pause=p95_full,
        p99_full_pause=p99_full,
        avg_mixed_pause=mean(pauses_mixed) if pauses_mixed else 0.0,
        max_mixed_pause=max(pauses_mixed) if pauses_mixed else 0.0,
        median_mixed_pause=percentile_sorted(pauses_mixed_sorted, 50),
        p95_mixed_pause=p95_mixed,
        p99_mixed_pause=p99_mixed,
        avg_remark_pause=mean(pauses_remark) if pauses_remark else 0.0,
        max_remark_pause=max(pauses_remark) if pauses_remark else 0.0,
        median_remark_pause=percentile_sorted(pauses_remark_sorted, 50),
        p95_remark_pause=p95_remark,
        p99_remark_pause=p99_remark,
        avg_cleanup_pause=mean(pauses_cleanup) if pauses_cleanup else 0.0,
        max_cleanup_pause=max(pauses_cleanup) if pauses_cleanup else 0.0,
        median_cleanup_pause=percentile_sorted(pauses_cleanup_sorted, 50),
        p95_cleanup_pause=p95_cleanup,
        p99_cleanup_pause=p99_cleanup,
        p50_pause=p50_pause,
        p95_pause=p95_pause,
        p99_pause=p99_pause,
        pauses_over_1s=pauses_over_1s,
        pauses_over_5s=pauses_over_5s,
        pauses_over_30s=pauses_over_30s,
        pauses_over_60s=pauses_over_60s,
        old_start=old_start,
        old_end=old_end,
        old_growth=old_growth,
        peak_old_pct=peak_old_pct,
        avg_old_pct=avg_old_pct,
        peak_heap_pct=peak_heap_pct,
        avg_heap_pct=avg_heap_pct,
        heap_headroom=heap_headroom,
        full_gc_reducing=full_gc_reducing,
        full_gc_growing=full_gc_growing,
        avg_post_full_gc_pct=avg_post_full_gc_pct,
        avg_reclaimed_kb=avg_reclaimed_kb,
        reclamation_declining=reclamation_declining,
        avg_promotion_kb=avg_promotion_kb,
        allocation_rate_mb_sec=allocation_rate_mb_sec,
        period_stats=period_stats,
        leak_severity=LeakSeverity(score=0.0, severity_level="None"),  # Placeholder
        warnings=warnings,
    )

    # Now calculate leak severity
    summary.leak_severity = calculate_leak_severity(summary, thresholds)

    # Add metaspace analysis
    summary.metaspace_analysis = analyze_metaspace(events)

    # Add G1 metrics if applicable
    summary.g1_metrics = calculate_g1_metrics(events, runtime_hours)

    # Add CMS metrics if applicable
    summary.cms_metrics = calculate_cms_metrics(events, runtime_hours)
    if summary.cms_metrics and full_events:
        summary.warnings.append(
            "⚠️ WARNING: CMS Full GC causes not parsed - concurrent mode failures are not reported"
        )

    # Add JVM heap configuration
    summary.jvm_heap_config = parse_jvm_heap_config(log_lines)

    return summary


# ============================================================
# RICH OUTPUT RENDERING
# ============================================================

GC_ANALYZE_THEME = Theme(
    {
        "critical": "bold red",
        "warning": "bold yellow",
        "success": "bold green",
        "info": "cyan",
        "metric": "white",
        "label": "dim white",
        "header": "bold magenta",
    }
)

console = Console(theme=GC_ANALYZE_THEME)


def create_key_value_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """Create a simple two-column key/value table."""
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="label")
    table.add_column("Value", style="metric")
    for label, value in rows:
        table.add_row(label, value)
    return table


def render_warning_banner(warnings: list[str]) -> Panel:
    """Render critical warnings in a prominent banner."""
    if not warnings:
        return Panel(
            Text("✓ No critical warnings", style="success"), title="Status", border_style="green"
        )

    critical_count = sum(1 for warning in warnings if "CRITICAL" in warning)
    warning_count = sum(1 for warning in warnings if "WARNING" in warning)

    if critical_count > 0:
        title_text = "[critical]Critical Warnings[/critical]"
        border_style = "red"
    elif warning_count > 0:
        title_text = "[warning]Warnings[/warning]"
        border_style = "yellow"
    else:
        title_text = "Status"
        border_style = "green"

    warning_text = Text()
    for index, warning in enumerate(warnings):
        clean_warning = warning.removeprefix("⚠️ ").removeprefix("⚠ ")
        line_ending = "\n" if index < len(warnings) - 1 else ""
        if "CRITICAL" in warning:
            warning_text.append("• ", style="critical")
            warning_text.append(clean_warning + line_ending, style="critical")
        else:
            warning_text.append("• ", style="warning")
            warning_text.append(clean_warning + line_ending, style="warning")

    return Panel(warning_text, title=title_text, border_style=border_style, expand=True)


def create_at_a_glance_panel(summary: GCSummary) -> Panel:
    """Create a compact at-a-glance summary panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="label")
    table.add_column("Value", style="metric")

    for label, value in build_at_a_glance_rows(summary):
        table.add_row(label, value)

    return Panel(table, title="At-a-Glance", border_style="info")


def create_runtime_overview_table(summary: GCSummary) -> Table:
    """Create runtime overview table."""
    return create_key_value_table("JVM Runtime Overview", build_runtime_overview_rows(summary))


def create_parsing_coverage_table(
    total_log_lines: int,
    raw_event_count: int,
    usable_event_count: int,
    unparsed_full_gc_headers: int | None = None,
) -> Table:
    """Create parsing coverage table to make data quality explicit."""
    return create_key_value_table(
        "Parsing Coverage",
        build_parsing_coverage_rows(
            total_log_lines=total_log_lines,
            raw_event_count=raw_event_count,
            usable_event_count=usable_event_count,
            unparsed_full_gc_headers=unparsed_full_gc_headers,
        ),
    )


def create_time_window_table(
    first_timestamp: datetime,
    last_timestamp: datetime,
    first_uptime_seconds: float,
    last_uptime_seconds: float,
) -> Table:
    """Create explicit time window table for correlation with incidents/deployments."""
    return create_key_value_table(
        "Time Window",
        build_time_window_rows(
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            first_uptime_seconds=first_uptime_seconds,
            last_uptime_seconds=last_uptime_seconds,
        ),
    )


def create_pause_distribution_table(summary: GCSummary) -> Table:
    """Create detailed pause time distribution table."""
    table = Table(title="Pause Time Analysis", show_header=True, header_style="header")

    table.add_column("GC Type", style="info")
    table.add_column("Count", justify="right", style="metric")
    table.add_column("Average", justify="right", style="metric")
    table.add_column("Median", justify="right", style="metric")
    table.add_column("P95", justify="right", style="metric")
    table.add_column("P99", justify="right", style="metric")
    table.add_column("Maximum", justify="right", style="metric")

    rows = build_pause_distribution_rows(summary)
    if rows:
        overall = rows[0]
        table.add_row(
            overall["type"],
            overall["count"],
            overall["avg"],
            overall["median"],
            overall["p95"],
            overall["p99"],
            overall["max"],
            style=overall.get("style") or None,
        )
        table.add_row("", "", "", "", "", "", "")
        for row in rows[1:]:
            table.add_row(
                row["type"],
                row["count"],
                row["avg"],
                row["median"],
                row["p95"],
                row["p99"],
                row["max"],
                style=row.get("style") or None,
            )

    return table


def render_leak_severity_panel(leak_severity: LeakSeverity) -> Panel:
    """Render memory leak severity assessment."""
    color_map = {
        "None": "green",
        "Low": "yellow",
        "Medium": "yellow",
        "High": "red",
        "Critical": "bold red",
    }

    color = color_map.get(leak_severity.severity_level, "white")

    content = Text()
    content.append("Leak Severity Score: ", style="label")
    content.append(f"{leak_severity.score:.1f}/100\n", style=color)
    content.append("Level: ", style="label")
    content.append(f"{leak_severity.severity_level}\n\n", style=color)

    if any("Insufficient old-gen data" in indicator for indicator in leak_severity.indicators):
        content.append("Note: ", style="label")
        content.append("Score not computed without old-gen data.\n\n", style="warning")

    if leak_severity.indicators:
        content.append("Indicators:\n", style="label")
        for index, indicator in enumerate(leak_severity.indicators):
            line_ending = "\n" if index < len(leak_severity.indicators) - 1 else ""
            content.append(f"  • {indicator}{line_ending}", style=color)

    return Panel(content, title=f"[{color}]Memory Leak Assessment[/{color}]", border_style=color)


def classify_warning_for_health(warning: str) -> Literal["health", "data"]:
    """Classify warnings into health vs data/coverage limitations."""
    data_markers = (
        "Old generation utilization not available",
        "Log runtime only",
        "CMS Full GC causes not parsed",
        "trend analysis unreliable",
    )
    if any(marker in warning for marker in data_markers):
        return "data"
    return "health"


def determine_quick_diagnosis(summary: GCSummary) -> str:
    """Determine concise two-line diagnosis: GC health + leak risk."""
    health_warnings = [w for w in summary.warnings if classify_warning_for_health(w) == "health"]
    critical_count = sum(1 for warning in health_warnings if "CRITICAL" in warning)
    warning_count = sum(1 for warning in health_warnings if "WARNING" in warning)

    if critical_count > 0:
        health_line = "🔴 GC HEALTH: Critical issues detected"
    elif warning_count > 0:
        health_line = "🟡 GC HEALTH: Degraded; review warnings"
    else:
        health_line = "✅ GC HEALTH: Stable"

    severity = summary.leak_severity.severity_level
    has_old_gen_data = not any(
        "Insufficient old-gen data" in indicator for indicator in summary.leak_severity.indicators
    )

    if not has_old_gen_data:
        leak_line = "⚪ LEAK RISK: Not assessed (insufficient old-gen data)"
    elif severity == "Critical":
        leak_line = f"🔴 LEAK RISK: Critical (old gen +{summary.old_growth:.1f} pts)"
    elif severity == "High":
        leak_line = f"🔴 LEAK RISK: High (old gen +{summary.old_growth:.1f} pts)"
    elif severity == "Medium":
        leak_line = "🟡 LEAK RISK: Medium"
    elif severity == "Low":
        leak_line = "🟡 LEAK RISK: Low"
    else:
        leak_line = "🟢 LEAK RISK: None detected"

    return f"{health_line}\n{leak_line}"


def build_evidence_based_recommendations(summary: GCSummary) -> list[str]:
    """Build evidence-based recommendations tied to observed log metrics.

    When JVM heap config is available, recommendations reference specific
    JVM flags and their current values for actionable tuning guidance.
    """
    recommendations: list[str] = []
    config = summary.jvm_heap_config

    # Post-Full-GC old gen retention
    if summary.full_gc_count > 0 and summary.avg_post_full_gc_pct is not None:
        if summary.avg_post_full_gc_pct > 90:
            rec = (
                "Old gen remains >90% after Full GC; capture heap dump to identify retained objects"
            )
            if config and config.max_heap_size_bytes:
                current_xmx = _format_bytes_human(config.max_heap_size_bytes)
                rec += (
                    f". Current -Xmx is {current_xmx};"
                    " consider increasing after root cause analysis"
                )
            recommendations.append(rec)
        elif summary.avg_post_full_gc_pct > 80:
            rec = "Old gen remains >80% after Full GC; review retained objects and GC ergonomics"
            if config and config.max_heap_size_bytes:
                current_xmx = _format_bytes_human(config.max_heap_size_bytes)
                rec += f". Current -Xmx is {current_xmx}; may need increase"
            recommendations.append(rec)

    # GC overhead
    if summary.gc_overhead_pct > 20:
        rec = "GC overhead >20%; reduce allocation rate or tune GC to improve throughput"
        if config and config.gc_time_ratio is not None:
            rec += f". Current -XX:GCTimeRatio={config.gc_time_ratio}"
        recommendations.append(rec)
    elif summary.gc_overhead_pct > 10:
        rec = "GC overhead >10%; investigate allocation hotspots and GC tuning"
        if config and config.gc_time_ratio is not None:
            rec += f". Current -XX:GCTimeRatio={config.gc_time_ratio}"
        recommendations.append(rec)

    # Pause time
    if summary.p99_pause > 10:
        recommendations.append(
            "P99 GC pause >10s; review pause sources and consider heap/GC tuning"
        )
    elif summary.p99_pause > 5:
        recommendations.append(
            "P99 GC pause >5s; consider reducing pause times with tuning or allocation changes"
        )

    # Consecutive Full GCs
    if summary.consecutive_full_gcs > 1:
        recommendations.append(
            "Consecutive Full GCs within 5 minutes; investigate memory thrashing and retained heap"
        )

    # Frequent Full GCs
    if summary.full_gc_per_hour > 1:
        rec = "Frequent Full GCs; review old-gen pressure and allocation patterns"
        if config and config.initiating_heap_occupancy_percent is not None:
            rec += (
                f". Current -XX:InitiatingHeapOccupancyPercent="
                f"{config.initiating_heap_occupancy_percent};"
                " try lowering to trigger concurrent GC earlier"
            )
        recommendations.append(rec)

    # G1-specific: humongous allocations
    if summary.g1_metrics and summary.g1_metrics.humongous_allocation_count > 0:
        humongous_count = summary.g1_metrics.humongous_allocation_count
        rec = (
            f"G1 humongous allocations detected ({humongous_count} events); "
            "objects larger than half a G1 region bypass normal allocation"
        )
        if config and config.g1_heap_region_size_bytes:
            region_size = _format_bytes_human(config.g1_heap_region_size_bytes)
            rec += (
                f". Current -XX:G1HeapRegionSize={region_size}; "
                "consider increasing to reduce humongous allocations"
            )
        recommendations.append(rec)

    # Heap headroom
    if summary.heap_headroom < 10:
        rec = f"Low heap headroom ({summary.heap_headroom:.1f}%); risk of GC thrashing and OOM"
        if config and config.max_heap_size_bytes:
            current_xmx = _format_bytes_human(config.max_heap_size_bytes)
            suggested_bytes = int(config.max_heap_size_bytes * 1.25)
            suggested_xmx = _format_bytes_human(suggested_bytes)
            rec += f". Current -Xmx={current_xmx}; consider increasing to {suggested_xmx}"
        recommendations.append(rec)

    # Metaspace
    if summary.metaspace_analysis:
        if summary.metaspace_analysis.peak_metaspace_used_percentage > 90:
            recommendations.append(
                "Metaspace >90% peak; investigate classloader leaks or metaspace sizing"
            )
        if summary.metaspace_analysis.metaspace_growth_kb > 50 * 1024:
            recommendations.append("Metaspace grew >50MB; investigate classloading churn")

    # Allocation rate
    if summary.allocation_rate_mb_sec and summary.allocation_rate_mb_sec > 500:
        recommendations.append(
            f"High allocation rate ({summary.allocation_rate_mb_sec:.0f} MB/sec); "
            "review allocation-heavy code paths and consider object pooling"
        )

    return recommendations


def format_seconds(seconds: float) -> str:
    """Format seconds for human-readable output."""
    if seconds > 60:
        return f"{seconds:.1f}s ({seconds / 60:.1f}m)"
    return f"{seconds:.3f}s"


def sanitize_text(text: str) -> str:
    """Strip emoji markers for Markdown output."""
    return (
        text.replace("⚠️ ", "")
        .replace("⚠ ", "")
        .replace("🔴 ", "")
        .replace("🟠 ", "")
        .replace("🟡 ", "")
        .replace("🟢 ", "")
        .replace("✖ ", "")
        .replace("✓ ", "")
    )


def build_parsing_coverage_rows(
    total_log_lines: int,
    raw_event_count: int,
    usable_event_count: int,
    unparsed_full_gc_headers: int | None = None,
) -> list[tuple[str, str]]:
    """Build rows describing parsing coverage."""
    rows = [
        ("Total log lines", str(total_log_lines)),
        ("Parsed events (raw)", str(raw_event_count)),
        ("Usable events", str(usable_event_count)),
        ("Dropped events", str(max(0, raw_event_count - usable_event_count))),
    ]
    if raw_event_count > 0:
        usable_pct = usable_event_count / raw_event_count * 100.0
        rows.append(("Usable rate", f"{usable_pct:.1f}%"))
    if unparsed_full_gc_headers is not None and unparsed_full_gc_headers > 0:
        rows.append(("Unparsed Full GC headers", str(unparsed_full_gc_headers)))
    return rows


def build_time_window_rows(
    first_timestamp: datetime,
    last_timestamp: datetime,
    first_uptime_seconds: float,
    last_uptime_seconds: float,
) -> list[tuple[str, str]]:
    """Build rows describing the time window."""
    return [
        ("First event timestamp", first_timestamp.isoformat()),
        ("Last event timestamp", last_timestamp.isoformat()),
        ("First event uptime", f"{first_uptime_seconds:.3f}s"),
        ("Last event uptime", f"{last_uptime_seconds:.3f}s"),
    ]


def _interpret_gc_overhead(overhead_pct: float) -> str:
    """Return threshold-based interpretation for GC overhead."""
    if overhead_pct > 20:
        return "critical, application severely impacted; target <5%"
    if overhead_pct > 10:
        return "elevated, investigate allocation hotspots; target <5%"
    if overhead_pct > 5:
        return "moderate; target <5%"
    return "healthy"


def _interpret_p99_pause(p99_seconds: float) -> str:
    """Return threshold-based interpretation for P99 STW pause."""
    if p99_seconds > 10:
        return "1-in-100 pauses this long; severe impact on latency"
    if p99_seconds > 5:
        return "1-in-100 pauses this long; may impact latency-sensitive workloads"
    if p99_seconds > 1:
        return "1-in-100 pauses this long; monitor for latency impact"
    return "1-in-100 pauses this long; within normal range"


def _interpret_peak_heap(peak_pct: float) -> str:
    """Return threshold-based interpretation for peak heap usage."""
    headroom = 100.0 - peak_pct
    if peak_pct > 95:
        return f"only {headroom:.0f}% headroom; OOM risk is imminent"
    if peak_pct > 90:
        return f"only {headroom:.0f}% headroom before OOM pressure"
    if peak_pct > 80:
        return f"{headroom:.0f}% headroom; approaching pressure zone"
    return f"{headroom:.0f}% headroom; comfortable"


def _interpret_full_gc_rate(full_gc_per_hour: float) -> str:
    """Return threshold-based interpretation for Full GC rate."""
    if full_gc_per_hour > 10:
        return "each Full GC stops the entire JVM; critical frequency"
    if full_gc_per_hour > 1:
        return "each Full GC stops the entire JVM; investigate cause"
    if full_gc_per_hour > 0:
        return "occasional; monitor for increases"
    return "none observed"


def build_at_a_glance_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build compact high-level summary rows with threshold-based interpretations."""
    overhead_interp = _interpret_gc_overhead(summary.gc_overhead_pct)
    p99_interp = _interpret_p99_pause(summary.p99_pause)
    peak_heap_interp = _interpret_peak_heap(summary.peak_heap_pct)
    full_gc_interp = _interpret_full_gc_rate(summary.full_gc_per_hour)

    return [
        ("Runtime", f"{summary.runtime_hours:.2f} hrs"),
        ("GC Events", f"{summary.total_event_count}"),
        (
            "GC Overhead (pause time)",
            f"{summary.gc_overhead_pct:.2f}% ({overhead_interp})",
        ),
        ("P99 STW Pause", f"{summary.p99_pause:.3f}s ({p99_interp})"),
        ("Peak Heap", f"{summary.peak_heap_pct:.1f}% ({peak_heap_interp})"),
        ("Full GCs/hr", f"{summary.full_gc_per_hour:.2f} ({full_gc_interp})"),
        ("Leak Level", f"{summary.leak_severity.severity_level}"),
    ]


def build_jvm_config_rows(config: JVMHeapConfig) -> list[tuple[str, str]]:
    """Build JVM heap and GC configuration rows."""
    rows: list[tuple[str, str]] = []

    # Heap Sizing
    if config.initial_heap_size_bytes:
        rows.append(
            ("Initial Heap Size (-Xms)", config.format_size(config.initial_heap_size_bytes))
        )
    if config.max_heap_size_bytes:
        rows.append(("Maximum Heap Size (-Xmx)", config.format_size(config.max_heap_size_bytes)))
    if config.new_size_bytes:
        rows.append(("New Generation Size (-Xmn)", config.format_size(config.new_size_bytes)))
    if config.max_new_size_bytes:
        rows.append(("Max New Generation Size", config.format_size(config.max_new_size_bytes)))
    if config.old_size_bytes:
        rows.append(("Old Generation Size", config.format_size(config.old_size_bytes)))
    if config.max_ram_bytes:
        rows.append(("Max RAM", config.format_size(config.max_ram_bytes)))
    if config.initial_ram_percentage is not None:
        rows.append(("Initial RAM %", f"{config.initial_ram_percentage}%"))
    if config.min_ram_percentage is not None:
        rows.append(("Min RAM %", f"{config.min_ram_percentage}%"))
    if config.max_ram_percentage is not None:
        rows.append(("Max RAM %", f"{config.max_ram_percentage}%"))
    if config.max_ram_fraction is not None:
        rows.append(("Max RAM Fraction", str(config.max_ram_fraction)))
    if config.soft_max_heap_size_bytes:
        rows.append(("Soft Max Heap Size", config.format_size(config.soft_max_heap_size_bytes)))
    if config.always_pre_touch is True:
        rows.append(("Always Pre Touch", "enabled"))
    if config.thread_stack_size_kb is not None:
        rows.append(("Thread Stack Size", f"{config.thread_stack_size_kb}K"))

    # Ratios and Thresholds
    if config.new_ratio is not None:
        rows.append(("New Ratio (Old/New)", str(config.new_ratio)))
    if config.survivor_ratio is not None:
        rows.append(("Survivor Ratio (Eden/Survivor)", str(config.survivor_ratio)))
    if config.initial_survivor_ratio is not None:
        rows.append(("Initial Survivor Ratio", str(config.initial_survivor_ratio)))
    if config.target_survivor_ratio is not None:
        rows.append(("Target Survivor Ratio", f"{config.target_survivor_ratio}%"))
    if config.max_tenuring_threshold is not None:
        rows.append(("Max Tenuring Threshold", str(config.max_tenuring_threshold)))
    if config.min_heap_free_ratio is not None:
        rows.append(("Min Heap Free Ratio", f"{config.min_heap_free_ratio}%"))
    if config.max_heap_free_ratio is not None:
        rows.append(("Max Heap Free Ratio", f"{config.max_heap_free_ratio}%"))

    # Metaspace
    if config.metaspace_size_bytes:
        rows.append(("Metaspace Size", config.format_size(config.metaspace_size_bytes)))
    if config.max_metaspace_size_bytes:
        rows.append(("Max Metaspace Size", config.format_size(config.max_metaspace_size_bytes)))
    if config.max_direct_memory_size_bytes:
        rows.append(
            ("Max Direct Memory Size", config.format_size(config.max_direct_memory_size_bytes))
        )

    # GC Threading
    if config.parallel_gc_threads is not None:
        rows.append(("Parallel GC Threads", str(config.parallel_gc_threads)))
    if config.conc_gc_threads is not None:
        rows.append(("Concurrent GC Threads", str(config.conc_gc_threads)))

    # GC Behavior Tuning
    if config.max_gc_pause_millis is not None:
        rows.append(("Max GC Pause Target", f"{config.max_gc_pause_millis}ms"))
    if config.gc_time_ratio is not None:
        rows.append(("GC Time Ratio", str(config.gc_time_ratio)))
    if config.adaptive_size_policy_weight is not None:
        rows.append(("Adaptive Size Policy Weight", str(config.adaptive_size_policy_weight)))
    if config.use_adaptive_size_policy is True:
        rows.append(("Adaptive Size Policy", "enabled"))
    if config.use_adaptive_size_policy_footprint_goal is True:
        rows.append(("Adaptive Size Policy Footprint Goal", "enabled"))

    # G1-Specific
    if config.g1_heap_region_size_bytes:
        rows.append(("G1 Heap Region Size", config.format_size(config.g1_heap_region_size_bytes)))
    if config.initiating_heap_occupancy_percent is not None:
        rows.append(("Initiating Heap Occupancy %", f"{config.initiating_heap_occupancy_percent}%"))
    if config.g1_reserve_percent is not None:
        rows.append(("G1 Reserve Percent", f"{config.g1_reserve_percent}%"))
    if config.g1_new_size_percent is not None:
        rows.append(("G1 New Size Percent", f"{config.g1_new_size_percent}%"))
    if config.g1_max_new_size_percent is not None:
        rows.append(("G1 Max New Size Percent", f"{config.g1_max_new_size_percent}%"))
    if config.g1_mixed_gc_count_target is not None:
        rows.append(("G1 Mixed GC Count Target", str(config.g1_mixed_gc_count_target)))
    if config.g1_heap_waste_percent is not None:
        rows.append(("G1 Heap Waste Percent", f"{config.g1_heap_waste_percent}%"))
    if config.g1_mixed_gc_live_threshold_percent is not None:
        rows.append(
            (
                "G1 Mixed GC Live Threshold Percent",
                f"{config.g1_mixed_gc_live_threshold_percent}%",
            )
        )
    if config.g1_rset_updating_pause_time_percent is not None:
        rows.append(
            (
                "G1 RSet Updating Pause Time Percent",
                f"{config.g1_rset_updating_pause_time_percent}%",
            )
        )
    if config.g1_old_cset_region_threshold_percent is not None:
        rows.append(
            (
                "G1 Old CSet Region Threshold Percent",
                f"{config.g1_old_cset_region_threshold_percent}%",
            )
        )

    # CMS-Specific
    if config.cms_initiating_occupancy_fraction is not None:
        rows.append(
            (
                "CMS Initiating Occupancy Fraction",
                f"{config.cms_initiating_occupancy_fraction}%",
            )
        )
    if config.use_cms_initiating_occupancy_only is True:
        rows.append(("Use CMS Initiating Occupancy Only", "enabled"))
    if config.cms_full_gcs_before_compaction is not None:
        rows.append(("CMS Full GCs Before Compaction", str(config.cms_full_gcs_before_compaction)))
    if config.cms_wait_duration is not None:
        rows.append(("CMS Wait Duration", f"{config.cms_wait_duration}ms"))

    return rows


def build_runtime_overview_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build runtime overview rows."""
    rows = [
        (
            "Observed runtime window",
            f"{summary.runtime_hours:.2f} hours ({summary.runtime_seconds:.0f} seconds)",
        ),
        ("Total GC events", f"{summary.total_event_count}"),
        ("Young GC events", f"{summary.young_gc_count}"),
        ("Full GC events", f"{summary.full_gc_count}"),
    ]
    if summary.mixed_gc_count > 0:
        rows.append(("Mixed GC events", f"{summary.mixed_gc_count}"))
    if summary.remark_gc_count > 0:
        rows.append(("Remark events (G1)", f"{summary.remark_gc_count}"))
    if summary.cleanup_gc_count > 0:
        rows.append(("Cleanup events (G1)", f"{summary.cleanup_gc_count}"))
    return rows


def build_frequency_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build GC frequency rows."""
    rows = [
        ("Young GCs per hour", f"{summary.young_gc_per_hour:.1f}"),
        ("Full GCs per hour", f"{summary.full_gc_per_hour:.2f}"),
    ]
    if summary.avg_full_gc_interval_min:
        rows.append(
            ("Avg time between Full GCs", f"{summary.avg_full_gc_interval_min:.1f} minutes")
        )
    if summary.min_full_gc_interval_min is not None:
        rows.append(
            (
                "Min Full GC interval (worst burst)",
                f"{summary.min_full_gc_interval_min:.1f} minutes",
            )
        )
    if summary.time_to_first_full_gc_hours is not None:
        rows.append(
            (
                "Time to first Full GC",
                f"{summary.time_to_first_full_gc_hours:.2f} hours",
            )
        )
    return rows


def build_overhead_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build GC overhead rows."""
    return [
        ("Total time in GC", f"{summary.total_gc_time_sec:.1f} seconds"),
        ("GC overhead", f"{summary.gc_overhead_pct:.2f}%"),
        ("Application throughput", f"{summary.throughput_pct:.2f}%"),
    ]


def _format_bytes_human(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f}G"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f}M"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}K"
    return f"{size_bytes}B"


def _format_pct_with_absolute(pct: float, total_bytes: int | None) -> str:
    """Format a percentage with optional absolute size when heap config is available."""
    if total_bytes is not None:
        used_bytes = int(total_bytes * pct / 100.0)
        return (
            f"{pct:.1f}% ({_format_bytes_human(used_bytes)} of {_format_bytes_human(total_bytes)})"
        )
    return f"{pct:.1f}%"


def build_heap_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build heap utilization rows."""
    max_heap_bytes = (
        summary.jvm_heap_config.max_heap_size_bytes if summary.jvm_heap_config else None
    )
    return [
        (
            "Peak heap usage",
            _format_pct_with_absolute(summary.peak_heap_pct, max_heap_bytes),
        ),
        (
            "Average heap usage",
            _format_pct_with_absolute(summary.avg_heap_pct, max_heap_bytes),
        ),
        ("Heap headroom", f"{summary.heap_headroom:.1f}%"),
    ]


def build_pause_distribution_rows(summary: GCSummary) -> list[dict[str, str]]:
    """Build pause distribution rows for output rendering."""
    rows: list[dict[str, str]] = []
    total_count = (
        summary.young_gc_count
        + summary.full_gc_count
        + summary.mixed_gc_count
        + summary.remark_gc_count
        + summary.cleanup_gc_count
    )
    rows.append(
        {
            "type": "Overall (All Types)",
            "count": str(total_count),
            "avg": "—",
            "median": format_seconds(summary.p50_pause),
            "p95": format_seconds(summary.p95_pause),
            "p99": format_seconds(summary.p99_pause),
            "max": "—",
            "style": "bold",
        }
    )
    if summary.young_gc_count > 0:
        rows.append(
            {
                "type": "Young GC",
                "count": str(summary.young_gc_count),
                "avg": format_seconds(summary.avg_young_pause),
                "median": format_seconds(summary.median_young_pause),
                "p95": format_seconds(summary.p95_young_pause),
                "p99": format_seconds(summary.p99_young_pause),
                "max": format_seconds(summary.max_young_pause),
                "style": "",
            }
        )
    if summary.full_gc_count > 0:
        rows.append(
            {
                "type": "Full GC",
                "count": str(summary.full_gc_count),
                "avg": format_seconds(summary.avg_full_pause),
                "median": format_seconds(summary.median_full_pause),
                "p95": format_seconds(summary.p95_full_pause),
                "p99": format_seconds(summary.p99_full_pause),
                "max": format_seconds(summary.max_full_pause),
                "style": "critical" if summary.max_full_pause > 30 else "",
            }
        )
    if summary.mixed_gc_count > 0:
        rows.append(
            {
                "type": "Mixed GC (G1)",
                "count": str(summary.mixed_gc_count),
                "avg": format_seconds(summary.avg_mixed_pause),
                "median": format_seconds(summary.median_mixed_pause),
                "p95": format_seconds(summary.p95_mixed_pause),
                "p99": format_seconds(summary.p99_mixed_pause),
                "max": format_seconds(summary.max_mixed_pause),
                "style": "",
            }
        )
    if summary.remark_gc_count > 0:
        rows.append(
            {
                "type": "Remark (G1)",
                "count": str(summary.remark_gc_count),
                "avg": format_seconds(summary.avg_remark_pause),
                "median": format_seconds(summary.median_remark_pause),
                "p95": format_seconds(summary.p95_remark_pause),
                "p99": format_seconds(summary.p99_remark_pause),
                "max": format_seconds(summary.max_remark_pause),
                "style": "",
            }
        )
    if summary.cleanup_gc_count > 0:
        rows.append(
            {
                "type": "Cleanup (G1)",
                "count": str(summary.cleanup_gc_count),
                "avg": format_seconds(summary.avg_cleanup_pause),
                "median": format_seconds(summary.median_cleanup_pause),
                "p95": format_seconds(summary.p95_cleanup_pause),
                "p99": format_seconds(summary.p99_cleanup_pause),
                "max": format_seconds(summary.max_cleanup_pause),
                "style": "",
            }
        )
    return rows


def build_pause_violation_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build rows for pause threshold violations, only when violations exist."""
    rows: list[tuple[str, str]] = []
    if summary.pauses_over_1s > 0:
        rows.append(("Pauses > 1s", str(summary.pauses_over_1s)))
    if summary.pauses_over_5s > 0:
        rows.append(("Pauses > 5s", str(summary.pauses_over_5s)))
    if summary.pauses_over_30s > 0:
        rows.append(("Pauses > 30s", str(summary.pauses_over_30s)))
    if summary.pauses_over_60s > 0:
        rows.append(("Pauses > 60s", str(summary.pauses_over_60s)))
    return rows


def build_top_pauses_rows(events: list[GCEvent], top_n: int) -> list[dict[str, str]]:
    """Build top pause rows."""
    rows: list[dict[str, str]] = []
    top_pauses = heapq.nlargest(top_n, events, key=lambda e: e.pause_time_seconds)
    for event in top_pauses:
        heap_after_mb = event.heap_after_kb / 1024
        heap_total_mb = event.heap_total_kb / 1024
        heap_after_text = (
            f"{heap_after_mb:.0f}M/{heap_total_mb:.0f}M ({event.heap_used_percentage:.1f}%)"
        )
        old_text = (
            f"{event.old_used_percentage:.1f}%" if event.old_used_percentage is not None else "—"
        )
        rows.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "gc_kind": event.gc_kind,
                "pause": f"{event.pause_time_seconds:.3f}s",
                "heap_after": heap_after_text,
                "old_gen": old_text,
            }
        )
    return rows


def build_old_gen_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build old generation analysis rows."""
    rows = [
        ("Old gen at start", f"{summary.old_start:.1f}%"),
        ("Old gen at end", f"{summary.old_end:.1f}%"),
        ("Net old gen growth", f"{summary.old_growth:+.1f} pts"),
        ("Peak old gen usage", f"{summary.peak_old_pct:.1f}%"),
    ]
    if summary.full_gc_count > 0:
        avg_reclaimed_mb = summary.avg_reclaimed_kb / 1024
        rows.append(("Avg reclaimed per Full GC", f"{avg_reclaimed_mb:.1f} MB"))
        rows.append(
            ("Full GCs reducing old gen", f"{summary.full_gc_reducing}/{summary.full_gc_count}")
        )
        if summary.reclamation_declining:
            rows.append(("⚠ Reclamation declining over time", ""))
    return rows


def build_trend_rows(summary: GCSummary) -> list[tuple[str, str, str]]:
    """Build trend rows for early/mid/late buckets."""
    rows: list[tuple[str, str, str]] = []
    for period_name in ["early", "mid", "late"]:
        if period_name in summary.period_stats:
            period = summary.period_stats[period_name]
            rows.append(
                (
                    period_name,
                    f"{period.average_old_gen_percentage:.1f}%",
                    str(period.full_gc_count),
                )
            )
    return rows


def build_allocation_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build allocation pressure rows."""
    rows = [("Allocation rate", f"{summary.allocation_rate_mb_sec:.2f} MB/sec")]
    if summary.avg_promotion_kb:
        rows.append(("Average promotion", f"{summary.avg_promotion_kb:.0f} KB per Young GC"))
    return rows


def build_g1_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build G1 GC metrics rows."""
    g1 = summary.g1_metrics
    if not g1:
        return []
    return [
        ("Mixed GC count", f"{g1.mixed_gc_count}"),
        ("Mixed GCs per hour", f"{g1.mixed_gc_frequency_per_hour:.2f}"),
        ("Humongous activity (events)", f"{g1.humongous_allocation_count}"),
        ("Evacuation failures", f"{g1.evacuation_failure_count}"),
        ("To-space exhaustion events", f"{g1.to_space_exhausted_count}"),
    ]


def build_cms_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build CMS GC metrics rows."""
    cms = summary.cms_metrics
    if not cms:
        return []
    rows = [
        ("CMS cycles per hour", f"{cms.cms_cycle_frequency_per_hour:.2f}"),
        ("Average concurrent pause", f"{cms.average_concurrent_pause_seconds:.4f}s"),
        ("Concurrent mode failures", f"{cms.concurrent_mode_failures}"),
    ]
    if cms.concurrent_mode_failures > 0:
        rows.append(("⚠ Failures indicate old gen filled before cycle completed", ""))
    return rows


def build_metaspace_rows(summary: GCSummary) -> list[tuple[str, str]]:
    """Build metaspace analysis rows."""
    meta = summary.metaspace_analysis
    if not meta:
        return []
    return [
        ("Peak metaspace usage", f"{meta.peak_metaspace_used_percentage:.1f}%"),
        ("Metaspace growth", f"{meta.metaspace_growth_kb // 1024}MB"),
    ]


def build_executive_narrative(summary: GCSummary, gc_type: str) -> str:
    """Build a 3-5 sentence plain-English executive summary from existing GCSummary fields."""
    sentences: list[str] = []

    # Sentence 1: JVM identity
    heap_size_clause = ""
    if summary.jvm_heap_config and summary.jvm_heap_config.max_heap_size_bytes:
        heap_size_clause = (
            f" with {_format_bytes_human(summary.jvm_heap_config.max_heap_size_bytes)} max heap"
        )
    sentences.append(f"This JVM is running the {gc_type} collector{heap_size_clause}.")

    # Sentence 2: Event counts over observation window
    sentences.append(
        f"Over {summary.runtime_hours:.1f} hours, "
        f"{summary.total_event_count} GC events were observed "
        f"({summary.young_gc_count} young, {summary.full_gc_count} full)."
    )

    # Sentence 3: Old gen trend (if data available and growth > 5 pts)
    if summary.old_start is not None and summary.old_end is not None and summary.old_growth > 5:
        sentences.append(
            f"Old generation grew {summary.old_growth:+.1f} percentage points "
            f"(from {summary.old_start:.1f}% to {summary.old_end:.1f}%), "
            "suggesting increasing memory pressure over the observation window."
        )

    # Sentence 4: Full GC acceleration warning
    if summary.full_gc_accelerating:
        sentences.append(
            "Full GC frequency is accelerating over time, "
            "which often precedes an out-of-memory condition."
        )

    # Sentence 5: GC overhead assessment
    if summary.gc_overhead_pct > 20:
        sentences.append(
            f"GC overhead is {summary.gc_overhead_pct:.1f}%, "
            "meaning the application is spending a critical amount of time "
            "paused for garbage collection."
        )
    elif summary.gc_overhead_pct > 10:
        sentences.append(
            f"GC overhead is {summary.gc_overhead_pct:.1f}%, "
            "which is elevated and warrants investigation into allocation patterns."
        )
    elif summary.gc_overhead_pct > 5:
        sentences.append(f"GC overhead is {summary.gc_overhead_pct:.1f}%, which is moderate.")
    else:
        sentences.append(f"GC overhead is {summary.gc_overhead_pct:.1f}%, which is healthy.")

    return " ".join(sentences)


def determine_overall_status(summary: GCSummary) -> tuple[str, str]:
    """Determine overall status text and style token."""
    critical_count = sum(1 for w in summary.warnings if "CRITICAL" in w)
    warning_count = sum(1 for w in summary.warnings if "WARNING" in w)
    if critical_count >= 3:
        return "✖ CRITICAL - IMMEDIATE ACTION REQUIRED", "critical"
    if critical_count > 0:
        return "✖ UNSTABLE - INTERVENTION REQUIRED", "critical"
    if warning_count > 0:
        return "⚠ DEGRADED - MONITORING REQUIRED", "warning"
    return "✓ STABLE", "success"


def render_rich_output(
    summary: GCSummary,
    gc_type: str,
    *,
    events: list[GCEvent] | None = None,
    total_log_lines: int | None = None,
    raw_event_count: int | None = None,
    unparsed_full_gc_headers: int | None = None,
) -> None:
    """Render comprehensive analysis using Rich components."""
    # Header
    console.print()
    console.print(Panel(f"GC Type: {gc_type}", style="header", expand=True))
    console.print()

    # Executive Summary
    executive_narrative = build_executive_narrative(summary, gc_type)
    console.print(Panel(executive_narrative, title="Executive Summary", border_style="info"))
    console.print()

    # Data quality / parsing coverage (if provided)
    if total_log_lines is not None and raw_event_count is not None:
        console.print(
            create_parsing_coverage_table(
                total_log_lines=total_log_lines,
                raw_event_count=raw_event_count,
                usable_event_count=summary.total_event_count,
                unparsed_full_gc_headers=unparsed_full_gc_headers,
            )
        )
        console.print()

    # Explicit time window (if events are provided)
    if events:
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)
        console.print(
            create_time_window_table(
                first_timestamp=first_event.timestamp,
                last_timestamp=last_event.timestamp,
                first_uptime_seconds=first_event.uptime_seconds,
                last_uptime_seconds=last_event.uptime_seconds,
            )
        )
        console.print()

    # Data Notes (data quality or coverage limitations)
    if summary.warnings:
        data_warnings = [w for w in summary.warnings if classify_warning_for_health(w) == "data"]
        if data_warnings:
            notes_text = Text()
            for index, warning in enumerate(data_warnings):
                clean_warning = warning.removeprefix("⚠️ ").removeprefix("⚠ ")
                line_ending = "\n" if index < len(data_warnings) - 1 else ""
                notes_text.append("• ", style="info")
                notes_text.append(clean_warning + line_ending, style="info")
            console.print(Panel(notes_text, title="[info]Data Notes[/info]", border_style="cyan"))
            console.print()

    # JVM Heap Configuration (if available)
    if summary.jvm_heap_config:
        console.print(
            create_key_value_table(
                "JVM Heap & GC Tuning Configuration",
                build_jvm_config_rows(summary.jvm_heap_config),
            )
        )
        console.print()

    # Quick Diagnosis
    diagnosis = determine_quick_diagnosis(summary)
    diagnosis_style = (
        "critical"
        if "🔴" in diagnosis
        else "warning"
        if "🟠" in diagnosis or "🟡" in diagnosis
        else "success"
    )
    console.print(Panel(diagnosis, style=diagnosis_style, title="Quick Diagnosis"))
    console.print()

    # At-a-Glance
    console.print(create_at_a_glance_panel(summary))
    console.print()

    # Runtime overview
    console.print(create_runtime_overview_table(summary))
    console.print()

    # GC Frequency
    console.print(create_key_value_table("GC Frequency & Rate", build_frequency_rows(summary)))
    console.print()

    # GC Overhead
    console.print(create_key_value_table("GC Overhead & Throughput", build_overhead_rows(summary)))
    console.print()

    # Heap Utilization
    console.print(create_key_value_table("Heap Utilization", build_heap_rows(summary)))
    console.print()

    # Pause Time Analysis
    console.print(create_pause_distribution_table(summary))
    console.print()

    # Pause Violations
    pause_violation_rows = build_pause_violation_rows(summary)
    if pause_violation_rows:
        console.print(create_key_value_table("Pause Threshold Violations", pause_violation_rows))
        console.print()

    # Top pauses (fast triage)
    if events:
        top_n = 5
        top_rows = build_top_pauses_rows(events, top_n)
        if top_rows:
            top_table = Table(
                title=f"Top {min(top_n, len(top_rows))} Longest Pauses",
                show_header=True,
                header_style="header",
            )
            top_table.add_column("Timestamp", style="info")
            top_table.add_column("GC", style="info")
            top_table.add_column("Pause", justify="right", style="metric")
            top_table.add_column("Heap after", justify="right", style="metric")
            top_table.add_column("Old gen", justify="right", style="metric")

            for row in top_rows:
                top_table.add_row(
                    row["timestamp"],
                    row["gc_kind"],
                    row["pause"],
                    row["heap_after"],
                    row["old_gen"],
                )

            console.print(top_table)
            console.print()

    # Collector-Specific Diagnostics
    collector_blocks: list[Table] = []

    if summary.old_start is not None and summary.old_end is not None:
        old_table = Table(
            title="Old Generation Analysis", show_header=False, box=None, padding=(0, 2)
        )
        old_table.add_column("Metric", style="label")
        old_table.add_column("Value", style="metric")
        for label, value in build_old_gen_rows(summary):
            if label.startswith("⚠"):
                old_table.add_row(label, value, style="warning")
            else:
                old_table.add_row(label, value)

        collector_blocks.append(old_table)

        if summary.period_stats:
            trend_table = Table(
                title="Trend (Early / Mid / Late)",
                show_header=True,
                header_style="header",
                box=None,
                padding=(0, 2),
            )
            trend_table.add_column("Period", style="info")
            trend_table.add_column("Avg old gen %", justify="right", style="metric")
            trend_table.add_column("Full GC count", justify="right", style="metric")
            for period_name, avg_old, full_gc_count in build_trend_rows(summary):
                trend_table.add_row(period_name, avg_old, full_gc_count)
            collector_blocks.append(trend_table)

    if summary.allocation_rate_mb_sec and summary.allocation_rate_mb_sec > 0:
        collector_blocks.append(
            create_key_value_table("Allocation Pressure", build_allocation_rows(summary))
        )

    if summary.g1_metrics:
        collector_blocks.append(create_key_value_table("G1 GC Metrics", build_g1_rows(summary)))

    if summary.cms_metrics:
        cms_rows = build_cms_rows(summary)
        cms_table = Table(title="CMS GC Metrics", show_header=False, box=None, padding=(0, 2))
        cms_table.add_column("Metric", style="label")
        cms_table.add_column("Value", style="metric")
        for label, value in cms_rows:
            if label.startswith("⚠"):
                cms_table.add_row(label, value, style="warning")
            else:
                cms_table.add_row(label, value)
        collector_blocks.append(cms_table)

    if collector_blocks:
        console.print(
            Panel(f"Collector-Specific Diagnostics: {gc_type}", style="header", expand=True)
        )
        console.print()
        for table in collector_blocks:
            console.print(table)
            console.print()

    # Metaspace Analysis
    if summary.metaspace_analysis:
        console.print(create_key_value_table("Metaspace Analysis", build_metaspace_rows(summary)))
        console.print()

    # Memory Leak Assessment
    console.print(render_leak_severity_panel(summary.leak_severity))
    console.print()

    # Evidence-based Recommendations
    recommendations = build_evidence_based_recommendations(summary)
    if recommendations:
        rec_panel = Panel(
            "\n".join(f"• {rec}" for rec in recommendations),
            title="Evidence-Based Recommendations",
            border_style="info",
        )
        console.print(rec_panel)
        console.print()

    # Warning banner (health only)
    if summary.warnings:
        health_warnings = [
            w for w in summary.warnings if classify_warning_for_health(w) == "health"
        ]
        if health_warnings:
            console.print(render_warning_banner(health_warnings))
            console.print()

    # Overall Status
    status, status_color = determine_overall_status(summary)
    console.print(Panel(status, title="Overall Status", border_style=status_color))


# ============================================================
# MARKDOWN EXPORT
# ============================================================


def export_markdown_summary(
    summary: GCSummary,
    gc_type: str,
    output_path: Path,
    *,
    events: list[GCEvent] | None = None,
    total_log_lines: int | None = None,
    raw_event_count: int | None = None,
    unparsed_full_gc_headers: int | None = None,
) -> None:
    """Export analysis summary to Markdown format."""
    md_content: list[str] = []

    md_content.append("# GC Analysis Report\n\n")
    md_content.append(f"**Generated:** {datetime.now().isoformat()}\n\n")
    md_content.append(f"**GC Type:** {gc_type}\n\n")

    # Executive Summary
    md_content.append("## Executive Summary\n\n")
    md_content.append(f"{build_executive_narrative(summary, gc_type)}\n\n")

    # Parsing Coverage
    if total_log_lines is not None and raw_event_count is not None:
        md_content.append("## Parsing Coverage\n\n")
        for label, value in build_parsing_coverage_rows(
            total_log_lines=total_log_lines,
            raw_event_count=raw_event_count,
            usable_event_count=summary.total_event_count,
            unparsed_full_gc_headers=unparsed_full_gc_headers,
        ):
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    # Time Window
    if events:
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)
        md_content.append("## Time Window\n\n")
        for label, value in build_time_window_rows(
            first_timestamp=first_event.timestamp,
            last_timestamp=last_event.timestamp,
            first_uptime_seconds=first_event.uptime_seconds,
            last_uptime_seconds=last_event.uptime_seconds,
        ):
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    # Data Notes (data quality or coverage limitations)
    if summary.warnings:
        data_warnings = [w for w in summary.warnings if classify_warning_for_health(w) == "data"]
        if data_warnings:
            md_content.append("## Data Notes\n\n")
            for warning in data_warnings:
                clean_warning = warning.removeprefix("⚠️ ").removeprefix("⚠ ")
                md_content.append(f"- {clean_warning}\n")
            md_content.append("\n")

    # JVM Heap Configuration
    if summary.jvm_heap_config:
        md_content.append("## JVM Heap & GC Tuning Configuration\n\n")
        for label, value in build_jvm_config_rows(summary.jvm_heap_config):
            md_content.append(f"- **{label}**: {value}\n")
        md_content.append("\n")

    # Quick Diagnosis
    md_content.append("## Quick Diagnosis\n\n")
    md_content.append(f"{sanitize_text(determine_quick_diagnosis(summary))}\n\n")

    # At-a-Glance
    md_content.append("## At-a-Glance\n\n")
    for label, value in build_at_a_glance_rows(summary):
        md_content.append(f"- **{label}:** {value}\n")
    md_content.append("\n")

    # Overall Status
    # Runtime Overview
    md_content.append("## Runtime Overview\n\n")
    for label, value in build_runtime_overview_rows(summary):
        md_content.append(f"- **{label}:** {value}\n")
    md_content.append("\n")

    # GC Frequency & Rate
    md_content.append("## GC Frequency & Rate\n\n")
    for label, value in build_frequency_rows(summary):
        md_content.append(f"- **{label}:** {value}\n")
    md_content.append("\n")

    # GC Overhead & Throughput
    md_content.append("## GC Overhead & Throughput\n\n")
    for label, value in build_overhead_rows(summary):
        md_content.append(f"- **{label}:** {value}\n")
    md_content.append("\n")

    # Heap Utilization
    md_content.append("## Heap Utilization\n\n")
    for label, value in build_heap_rows(summary):
        md_content.append(f"- **{label}:** {value}\n")
    md_content.append("\n")

    # Pause Time Analysis
    md_content.append("## Pause Time Analysis\n\n")
    md_content.append("| GC Type | Count | Avg | Median | P95 | P99 | Max |\n")
    md_content.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    md_content.extend(
        f"| {row['type']} | {row['count']} | {row['avg']} | "
        f"{row['median']} | {row['p95']} | {row['p99']} | {row['max']} |\n"
        for row in build_pause_distribution_rows(summary)
    )
    md_content.append("\n")

    # Pause Violations
    pause_violation_rows = build_pause_violation_rows(summary)
    if pause_violation_rows:
        md_content.append("### Pause Threshold Violations\n\n")
        for label, value in pause_violation_rows:
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    # Top pauses (fast triage)
    if events:
        top_n = 5
        top_rows = build_top_pauses_rows(events, top_n)
        if top_rows:
            md_content.append(f"## Top {min(top_n, len(top_rows))} Longest Pauses\n\n")
            md_content.append("| Timestamp | GC | Pause | Heap after | Old gen |\n")
            md_content.append("| --- | --- | ---: | ---: | ---: |\n")
            md_content.extend(
                f"| {row['timestamp']} | {row['gc_kind']} | {row['pause']} | "
                f"{row['heap_after']} | {row['old_gen']} |\n"
                for row in top_rows
            )
            md_content.append("\n")

    # Collector-Specific Diagnostics
    if summary.old_start is not None and summary.old_end is not None:
        md_content.append(f"## Collector-Specific Diagnostics: {gc_type}\n\n")
        md_content.append("### Old Generation Analysis\n\n")
        for label, value in build_old_gen_rows(summary):
            if value:
                md_content.append(f"- **{label}:** {value}\n")
            else:
                md_content.append(f"- **{label}:** yes\n")
        md_content.append("\n")

        if summary.period_stats:
            md_content.append("### Trend (Early / Mid / Late)\n\n")
            md_content.append("| Period | Avg old gen % | Full GC count |\n")
            md_content.append("| --- | ---: | ---: |\n")
            for period_name, avg_old, full_gc_count in build_trend_rows(summary):
                md_content.append(f"| {period_name} | {avg_old} | {full_gc_count} |\n")
            md_content.append("\n")

    if summary.allocation_rate_mb_sec and summary.allocation_rate_mb_sec > 0:
        md_content.append("### Allocation Pressure\n\n")
        for label, value in build_allocation_rows(summary):
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    if summary.g1_metrics:
        md_content.append("### G1 GC Metrics\n\n")
        for label, value in build_g1_rows(summary):
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    if summary.cms_metrics:
        md_content.append("### CMS GC Metrics\n\n")
        for label, value in build_cms_rows(summary):
            if value:
                md_content.append(f"- **{label}:** {value}\n")
            else:
                md_content.append(f"- **{label}:** yes\n")
        md_content.append("\n")

    # Metaspace Analysis
    if summary.metaspace_analysis:
        md_content.append("## Metaspace Analysis\n\n")
        for label, value in build_metaspace_rows(summary):
            md_content.append(f"- **{label}:** {value}\n")
        md_content.append("\n")

    # Leak Assessment
    md_content.append("## Memory Leak Assessment\n\n")
    md_content.append(f"- **Severity Score:** {summary.leak_severity.score:.1f}/100\n")
    md_content.append(f"- **Level:** {summary.leak_severity.severity_level}\n\n")

    if any(
        "Insufficient old-gen data" in indicator for indicator in summary.leak_severity.indicators
    ):
        md_content.append("**Note:** Score not computed without old-gen data.\n\n")

    if summary.leak_severity.indicators:
        md_content.append("**Indicators:**\n\n")
        md_content.extend(f"- {indicator}\n" for indicator in summary.leak_severity.indicators)
        md_content.append("\n")

    # Evidence-based Recommendations
    recommendations = build_evidence_based_recommendations(summary)
    if recommendations:
        md_content.append("## Evidence-Based Recommendations\n\n")
        md_content.extend(f"- {rec}\n" for rec in recommendations)
        md_content.append("\n")

    # Health Warnings (health only)
    if summary.warnings:
        health_warnings = [
            w for w in summary.warnings if classify_warning_for_health(w) == "health"
        ]
        if health_warnings:
            md_content.append("## Health Warnings\n\n")
            md_content.extend(f"- {sanitize_text(warning)}\n" for warning in health_warnings)
            md_content.append("\n")

    # Overall Status
    status, _status_style = determine_overall_status(summary)
    status = sanitize_text(status)
    md_content.append("## Overall Status\n\n")
    md_content.append(f"{status}\n\n")

    # Actionable Recommendations (only when there are issues)
    has_leak_issues = summary.leak_severity.severity_level in ["High", "Critical"]
    has_full_gc_issues = summary.full_gc_per_hour > 1
    has_overhead_issues = summary.gc_overhead_pct > 10
    has_any_issues = has_leak_issues or has_full_gc_issues or has_overhead_issues

    if has_any_issues:
        md_content.append("## Actionable Recommendations\n\n")

        if has_leak_issues:
            md_content.append("### IMMEDIATE ACTION REQUIRED\n\n")
            md_content.append("1. Capture heap dump NOW for analysis\n")
            md_content.append("2. Use heap dump analysis tools (MAT, JProfiler)\n")
            md_content.append("3. Identify objects with unexpected retention\n\n")

        if has_full_gc_issues:
            md_content.append("### Full GC Issues\n\n")
            md_content.append("- Excessive Full GC frequency detected\n")
            md_content.append("- Capture heap dump at peak old-gen usage\n")
            md_content.append("- Analyze retained object graphs\n\n")

        if has_overhead_issues:
            md_content.append("### GC Overhead\n\n")
            md_content.append("- Review allocation hotspots\n")
            md_content.append("- Consider object pooling for high-churn objects\n\n")

        md_content.append("### Caution\n\n")
        md_content.append("- Do not increase heap size without root cause analysis\n")
        md_content.append("- Do not ignore warning signs - issues will escalate\n")

    # Write to file
    output_path.write_text("".join(md_content), encoding="utf-8")


# ============================================================
# TYPER CLI INTERFACE
# ============================================================

app = typer.Typer(
    name="gc-analyze",
    help="Advanced GC log analyzer supporting Parallel GC, G1 GC, and CMS GC",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def analyze(
    log_file: Annotated[
        Path,
        typer.Argument(
            help="Path to GC log file to analyze",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Export analysis report to Markdown file (e.g., report.md)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    heap_warning: Annotated[
        float,
        typer.Option(
            "--heap-warning",
            help="Heap utilization percentage to trigger warnings (default: 80.0)",
            min=0.0,
            max=100.0,
        ),
    ] = 80.0,
    heap_critical: Annotated[
        float,
        typer.Option(
            "--heap-critical",
            help="Heap utilization percentage to trigger critical alerts (default: 95.0)",
            min=0.0,
            max=100.0,
        ),
    ] = 95.0,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output with detailed parsing information",
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile",
            help="Enable performance profiling and display timing statistics",
        ),
    ] = False,
    profile_output: Annotated[
        Path | None,
        typer.Option(
            "--profile-output",
            help="Save detailed profiling data to file (e.g., profile.prof)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Analyze a JVM GC log file (Parallel, G1, or CMS).

    Exit codes: 0 = clean, 1 = warnings, 2 = critical issues.
    """
    # Initialize profiler if requested
    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # Read log file
        with log_file.open() as f:
            lines = f.readlines()

        if verbose:
            console.print(f"[info]Read {len(lines)} lines from {log_file}[/info]")

        # Detect GC type and create parser
        parser = detect_gc_type_and_create_parser(lines)

        if verbose:
            console.print(f"[info]Detected GC type: {parser.gc_type_name}[/info]")

        # Parse events
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            parse_task = progress.add_task(
                f"[cyan]Parsing {parser.gc_type_name} events...", total=None
            )
            raw_events = parser.parse(lines)
            progress.update(parse_task, completed=100)

        # Normalize events
        events: list[GCEvent] = []
        for raw_event in raw_events:
            normalized_event = normalize_event(raw_event)
            if normalized_event is not None:
                events.append(normalized_event)

        if not events:
            console.print("[critical]ERROR: No usable GC events found in log file[/critical]")
            sys.exit(1)

        events.sort(key=lambda event: event.timestamp)

        if verbose:
            console.print(f"[info]Successfully parsed {len(events)} GC events[/info]")

        # Build summary with custom thresholds
        thresholds = DiagnosticThresholds(
            heap_warning_percentage=heap_warning, heap_critical_percentage=heap_critical
        )
        summary = build_comprehensive_summary(events, thresholds, lines)

        # Render output
        possible_full_gc_headers: int | None = None
        if parser.gc_type_name == "Parallel GC":
            possible_full_gc_headers = sum(1 for line in lines if "Full GC" in line)

        unparsed_full_gc_headers: int | None = None
        if possible_full_gc_headers is not None:
            unparsed_full_gc_headers = max(0, possible_full_gc_headers - summary.full_gc_count)

        render_rich_output(
            summary,
            parser.gc_type_name,
            events=events,
            total_log_lines=len(lines),
            raw_event_count=len(raw_events),
            unparsed_full_gc_headers=unparsed_full_gc_headers,
        )

        # Export if requested
        if output:
            export_markdown_summary(
                summary,
                parser.gc_type_name,
                output,
                events=events,
                total_log_lines=len(lines),
                raw_event_count=len(raw_events),
                unparsed_full_gc_headers=unparsed_full_gc_headers,
            )
            console.print(f"\n[success]✓ Summary exported to {output}[/success]")

        # Finalize profiler and display stats
        if profiler:
            profiler.disable()

            # Save to file if requested
            if profile_output:
                profiler.dump_stats(str(profile_output))
                console.print(f"\n[info]Profiling data saved to {profile_output}[/info]")

            # Display top 20 functions by cumulative time
            console.print(
                "\n[bold cyan]═══ Performance Profile (Top 20 Functions) ═══[/bold cyan]\n"
            )
            # Capture stats output
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(20)

            console.print(stats_stream.getvalue())

        # Exit with appropriate code
        if summary.leak_severity.severity_level in ["High", "Critical"]:
            sys.exit(2)  # Critical issues found
        elif summary.warnings:
            sys.exit(1)  # Warnings found
        # Otherwise exit normally with 0

    except ValueError as e:
        console.print(f"[critical]ERROR: {e}[/critical]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[critical]ERROR: {e}[/critical]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def version() -> None:
    """Display version."""
    console.print("gc-analyze 3.0.0")


if __name__ == "__main__":
    app()
