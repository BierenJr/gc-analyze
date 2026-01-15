# gc-analyze

`gc-analyze` is a command-line utility that parses JVM garbage-collection (GC) logs and produces an operations-oriented summary: GC rate, pause distributions (P50/P95/P99), heap/old-gen pressure signals, and a memory-leak risk score with actionable recommendations.

## What it does

Given a GC log file, the tool:

- Detects the collector format from the beginning of the log (Parallel, G1, or CMS).
- Parses GC events and normalizes them into a common event model.
- Computes runtime-window statistics (rates, percentiles, utilization, trends).
- Emits a rich terminal report and (optionally) a Markdown report for sharing.

### Supported GC log formats

- **Parallel GC**: `PSYoungGen` / `ParOldGen`
- **G1 GC**:
  - Java 9+ unified logging (e.g., `-Xlog:gc*`)
  - Java 8-era “legacy” formats (e.g., `G1Young`, `G1Mixed`)
- **CMS GC**: `ParNew` + `CMS`

## Quick start

```bash
uv run gc-analyze analyze path/to/gc.log
```

## Installation

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd gc-analyze

# Install uv python
uv python install 3.14

# Create uv vitrual environment
uv venv 

# Install dependencies
uv sync
```

### Run the analyzer

```bash
uv run gc-analyze analyze path/to/gc.log
uv run gc-analyze analyze path/to/gc.log --output report.md
```

## Usage

### Analyze a log

```bash
uv run gc-analyze analyze [OPTIONS] LOG_FILE
```

Options:

- `--output`, `-o`: write a Markdown report to the given path
- `--heap-warning`: peak heap % threshold for warnings (default `80.0`)
- `--heap-critical`: peak heap % threshold for critical warnings (default `95.0`)
- `--verbose`, `-v`: print parsing details and exception tracebacks

## Understanding the terminal summary output

The terminal report is organized into a consistent core, plus a collector-specific block. Not every
section appears for every log; some require collector-specific fields.

### Quick Diagnosis

A two-line health classification derived from the computed metrics:

- **GC Health**: overall GC health based on warning thresholds (critical/degraded/stable)
- **Leak Risk**: leak risk level derived from old-gen trends (or “not assessed” if data is missing)

### Parsing Coverage

A data-quality summary for the input log:

- **Total log lines**, **Parsed events (raw)**, **Usable events**, **Dropped events**
- **Usable rate** as a percentage of parsed events
- **Unparsed Full GC headers** (Parallel GC logs with multi-line full GC entries)

### Time Window

Explicit first/last timestamps and uptimes to correlate GC behavior with incidents or deploys:

- **First event timestamp** / **Last event timestamp**
- **First event uptime** / **Last event uptime**

### Data Notes

Up-front data limitations (for example, old-gen not present in the log) so readers don’t
misinterpret leak scoring or trend indicators.

### At-a-Glance

A compact panel intended for “first look” triage:

- **Runtime**: time span covered by parsed events
- **GC Events**: number of parsed events
- **GC Overhead (pause time)**: % of runtime spent in GC pauses
- **P99 STW Pause**: 99th percentile pause (seconds) across all parsed events
- **Peak Heap**: maximum observed heap utilization %
- **Full GCs/hr**: Full GC rate normalized per hour
- **Leak Level**: `None` / `Low` / `Medium` / `High` / `Critical`

### JVM Heap & GC Tuning Configuration (when present)

If the log contains a `CommandLine flags:` line (or certain unified-logging headers), the report extracts and displays heap sizing and common GC tuning knobs (e.g., `-Xms`, `-Xmx`, `-XX:MaxGCPauseMillis`, selected G1/CMS flags).

### GC Frequency & Rate

How often collections occur over the observed runtime window:

- **Young GCs per hour**
- **Full GCs per hour**
- **Avg time between Full GCs** (if there are multiple Full GCs)

### GC Overhead & Throughput

- **Total time in GC**: sum of all parsed pause times
- **GC overhead**: `(total pause time / runtime window) * 100`
- **Application throughput**: `100 - GC overhead`

### Heap Utilization

- **Peak heap usage**: maximum observed heap usage %
- **Average heap usage**: average post-GC heap usage %
- **Heap headroom**: `100 - peak heap usage` (lower headroom means higher OOM risk)

### Old Generation Analysis (when old-gen data is available)

Old-gen pressure is one of the strongest early warning signals for leaks and long-tail latency:

- **Old gen at start / end**: old-gen utilization % at the beginning vs end of the log window
- **Net old gen growth**: end-start in percentage points (e.g., `+12.4 pts`)
- **Peak old gen usage**
- **Avg reclaimed per Full GC** and **Full GCs reducing old gen** (effectiveness indicators)

### Allocation Pressure (when calculable)

- **Allocation rate** (MB/sec), derived from young-gen activity and event cadence
- **Average promotion** (KB promoted per Young GC), when available

### Pause Time Analysis

A distribution table showing (by GC type) count, average, median, P95, P99, and maximum pause times. The “Overall (All Types)” row reports P50/P95/P99 across all parsed events.

### Top Longest Pauses

A short list of the longest pauses with timestamp, GC type, pause time, heap-after, and old-gen % (if available).

### Collector-Specific Diagnostics

A single block containing only the diagnostics that apply to the detected collector:

- **G1 GC Metrics**: mixed GC count/rate, humongous activity, evacuation/to-space failures
- **Parallel GC**: old-gen analysis, early/mid/late trend view, allocation pressure
- **CMS GC Metrics**: CMS cycles per hour, average concurrent pause, CMS-specific notes

### Memory Leak Assessment

Leak severity is a score from **0–100** plus a qualitative level:

- `None` (<20)
- `Low` (20–39)
- `Medium` (40–59)
- `High` (60–79)
- `Critical` (80+)

The score is computed from multiple indicators, including:

- Net old-gen growth over the log window (percentage points)
- Declining memory reclamation effectiveness over time
- Accelerating Full GC frequency
- High residual old-gen occupancy after Full GC
- Low heap headroom

The report lists the specific indicators that contributed to the score.

### Metaspace Analysis (when metaspace is present)

- **Peak metaspace usage %**
- **Metaspace growth** (MB) over the log window

### Evidence-Based Recommendations

A short, metric-linked recommendation list (e.g., reduce allocation hotspots, capture heap dumps when post-Full-GC old-gen stays high).

### Warnings and Overall Status

Warnings are emitted when thresholds are exceeded (for example: peak heap beyond `--heap-warning`/`--heap-critical`, high Full GC rate, high overhead, high P99 pause, consecutive Full GCs). The final “Overall Status” panel summarizes severity across all warnings.

## Markdown export (`--output`)

When you pass `--output report.md`, the tool writes a Markdown report containing:

- GC type and generation time
- Runtime overview and leak assessment (including indicators)
- Warnings and evidence-based recommendations
- Extracted JVM heap/GC configuration (when present)
