"""Run all tests and generate a visual HTML report with charts.

Usage:
    python tests/visualize_results.py

Outputs:
    tests/report/test_report.html   - Interactive HTML report
    tests/report/summary.png        - Test result summary chart
    tests/report/by_module.png      - Per-module breakdown chart
    tests/report/duration.png       - Test duration chart
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


REPORT_DIR = Path(__file__).parent / "report"
ROOT_DIR = Path(__file__).resolve().parent.parent


def run_tests() -> dict:
    """Run pytest and collect results as JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "results.json"

    cmd = [
        sys.executable, "-m", "pytest",
        str(ROOT_DIR / "tests"),
        f"--rootdir={ROOT_DIR}",
        "-v",
        "--tb=short",
        f"--json-report-file={json_path}",
        "--json-report",
        "-q",
    ]

    # Try with json-report plugin first; fall back to parsing output
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest-json-report", "-q"],
            capture_output=True,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
        stdout = result.stdout
        stderr = result.stderr

        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            return _normalize_json_report(raw, result.stdout, result.stderr)
    except Exception:
        pass

    # Fallback: parse verbose pytest output
    cmd_fallback = [
        sys.executable, "-m", "pytest",
        str(ROOT_DIR / "tests"),
        f"--rootdir={ROOT_DIR}",
        "-v",
        "--tb=short",
    ]
    result = subprocess.run(cmd_fallback, capture_output=True, text=True, cwd=str(ROOT_DIR))
    return _parse_pytest_output(result.stdout, result.stderr, result.returncode)


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Parse verbose pytest output into a structured dict."""
    tests = []
    for line in stdout.splitlines():
        # Match lines like: tests/test_config.py::TestModelConfig::test_defaults PASSED
        m = re.match(r"^(tests/\S+)::(\S+)::(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)", line)
        if m:
            file, cls, name, outcome = m.groups()
            tests.append({
                "nodeid": f"{file}::{cls}::{name}",
                "outcome": outcome.lower(),
                "duration": 0.0,
                "module": file.replace("tests/", "").replace(".py", ""),
                "class": cls,
                "name": name,
            })

    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    errors = sum(1 for t in tests if t["outcome"] == "error")
    skipped = sum(1 for t in tests if t["outcome"] == "skipped")

    # Extract duration from summary line if present
    duration_match = re.search(r"in ([\d.]+)s", stdout)
    total_duration = float(duration_match.group(1)) if duration_match else 0.0

    return {
        "summary": {
            "total": len(tests),
            "passed": passed,
            "failed": failed,
            "error": errors,
            "skipped": skipped,
            "duration": total_duration,
        },
        "tests": tests,
        "stdout": stdout,
        "stderr": stderr,
    }


def _normalize_json_report(raw: dict, stdout: str, stderr: str) -> dict:
    """Normalize pytest-json-report output to our standard format."""
    tests = []
    for t in raw.get("tests", []):
        nodeid = t.get("nodeid", "")
        parts = nodeid.split("::")
        module = parts[0].replace("tests/", "").replace(".py", "") if len(parts) > 0 else ""
        cls = parts[1] if len(parts) > 2 else ""
        name = parts[-1] if parts else ""

        tests.append({
            "nodeid": nodeid,
            "outcome": t.get("outcome", "unknown"),
            "duration": t.get("duration", 0.0),
            "module": module,
            "class": cls,
            "name": name,
        })

    s = raw.get("summary", {})
    return {
        "summary": {
            "total": s.get("total", len(tests)),
            "passed": s.get("passed", 0),
            "failed": s.get("failed", 0),
            "error": s.get("error", 0),
            "skipped": s.get("skipped", 0) + s.get("deselected", 0),
            "duration": raw.get("duration", 0.0),
        },
        "tests": tests,
        "stdout": stdout,
        "stderr": stderr,
    }


def plot_summary(data: dict):
    """Create a donut chart of pass/fail/skip/error counts."""
    s = data["summary"]
    labels = []
    sizes = []
    colors = []

    for key, color in [("passed", "#4CAF50"), ("failed", "#F44336"),
                        ("error", "#FF9800"), ("skipped", "#9E9E9E")]:
        val = s.get(key, 0)
        if val > 0:
            labels.append(f"{key.capitalize()} ({val})")
            sizes.append(val)
            colors.append(color)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    # Donut
    centre = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_patch(centre)
    ax.text(0, 0, f"{s['total']}\ntests", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#333")
    ax.set_title("Test Results Summary", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    path = REPORT_DIR / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart] {path}")
    return path


def plot_by_module(data: dict):
    """Stacked bar chart showing pass/fail per test module."""
    modules = defaultdict(lambda: {"passed": 0, "failed": 0, "error": 0, "skipped": 0})
    for t in data["tests"]:
        mod = t.get("module", t["nodeid"].split("::")[0])
        modules[mod][t["outcome"]] += 1

    mod_names = sorted(modules.keys())
    passed = [modules[m]["passed"] for m in mod_names]
    failed = [modules[m]["failed"] for m in mod_names]
    errors = [modules[m]["error"] for m in mod_names]
    skipped = [modules[m]["skipped"] for m in mod_names]

    # Shorten module names for display
    display_names = [m.replace("test_", "") for m in mod_names]

    x = np.arange(len(mod_names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(max(8, len(mod_names) * 1.5), 6))
    bars_passed = ax.bar(x, passed, width, label="Passed", color="#4CAF50")
    bars_failed = ax.bar(x, failed, width, bottom=passed, label="Failed", color="#F44336")
    bottom2 = [p + f for p, f in zip(passed, failed)]
    bars_error = ax.bar(x, errors, width, bottom=bottom2, label="Error", color="#FF9800")
    bottom3 = [b + e for b, e in zip(bottom2, errors)]
    bars_skipped = ax.bar(x, skipped, width, bottom=bottom3, label="Skipped", color="#9E9E9E")

    ax.set_ylabel("Number of Tests", fontsize=12)
    ax.set_title("Test Results by Module", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=11)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(p + f + e + s for p, f, e, s in zip(passed, failed, errors, skipped)), 1) * 1.15)

    # Add count labels on bars
    for bar_group in [bars_passed]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + h / 2,
                        str(int(h)), ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white")

    plt.tight_layout()
    path = REPORT_DIR / "by_module.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart] {path}")
    return path


def plot_test_status_grid(data: dict):
    """Create a grid/heatmap showing each test's status by module and class."""
    modules = defaultdict(list)
    for t in data["tests"]:
        mod = t.get("module", t["nodeid"].split("::")[0]).replace("test_", "")
        modules[mod].append(t)

    fig, axes = plt.subplots(
        len(modules), 1,
        figsize=(14, max(4, len(data["tests"]) * 0.35)),
        squeeze=False,
    )

    color_map = {"passed": "#4CAF50", "failed": "#F44336", "error": "#FF9800", "skipped": "#9E9E9E"}
    row_idx = 0

    for ax_idx, (mod_name, tests) in enumerate(sorted(modules.items())):
        ax = axes[ax_idx, 0]
        y_labels = []
        colors_list = []
        for t in tests:
            short_name = t["name"]
            cls = t.get("class", "")
            label = f"{cls}.{short_name}" if cls else short_name
            y_labels.append(label)
            colors_list.append(color_map.get(t["outcome"], "#9E9E9E"))

        y_pos = np.arange(len(tests))
        bars = ax.barh(y_pos, [1] * len(tests), color=colors_list, edgecolor="white", height=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlim(0, 1.3)
        ax.set_xticks([])
        ax.set_title(f"Module: {mod_name}", fontsize=11, fontweight="bold", loc="left")
        ax.invert_yaxis()

        # Status labels
        for i, t in enumerate(tests):
            ax.text(1.05, i, t["outcome"].upper(), va="center", fontsize=8,
                    fontweight="bold", color=color_map.get(t["outcome"], "#333"))

    # Legend
    patches = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in color_map.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.suptitle("Test Status Grid", fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = REPORT_DIR / "status_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart] {path}")
    return path


def generate_html_report(data: dict, chart_paths: dict):
    """Generate a self-contained HTML report with embedded charts."""
    import base64

    def embed_image(path: Path) -> str:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"

    s = data["summary"]
    pass_rate = (s["passed"] / s["total"] * 100) if s["total"] > 0 else 0

    # Build test table rows
    test_rows = ""
    for t in data["tests"]:
        outcome = t["outcome"]
        css_class = {"passed": "pass", "failed": "fail", "error": "error", "skipped": "skip"}.get(outcome, "")
        test_rows += f"""
        <tr class="{css_class}">
            <td>{t.get('module', '-')}</td>
            <td>{t.get('class', '-')}</td>
            <td>{t['name']}</td>
            <td class="status">{outcome.upper()}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Test Report - ViT + LoRA Image Classification</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #333; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ text-align: center; margin: 20px 0; font-size: 28px; color: #1a237e; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px; margin-bottom: 30px; }}
  .stat-card {{ background: white; border-radius: 12px; padding: 20px; text-align: center;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .stat-card .number {{ font-size: 36px; font-weight: bold; }}
  .stat-card .label {{ font-size: 13px; color: #888; margin-top: 5px; }}
  .stat-card.total .number {{ color: #1a237e; }}
  .stat-card.passed .number {{ color: #4CAF50; }}
  .stat-card.failed .number {{ color: #F44336; }}
  .stat-card.rate .number {{ color: #2196F3; }}
  .stat-card.time .number {{ color: #FF9800; font-size: 28px; }}

  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .chart-card img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  .chart-card.full {{ grid-column: 1 / -1; }}

  table {{ width: 100%; border-collapse: collapse; background: white;
          border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  th {{ background: #1a237e; color: white; padding: 12px 15px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px 15px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:hover {{ background: #f8f9ff; }}
  tr.pass .status {{ color: #4CAF50; font-weight: bold; }}
  tr.fail .status {{ color: #F44336; font-weight: bold; }}
  tr.error .status {{ color: #FF9800; font-weight: bold; }}
  tr.skip .status {{ color: #9E9E9E; font-weight: bold; }}

  .section-title {{ font-size: 20px; font-weight: bold; margin: 30px 0 15px; color: #1a237e; }}

  .stdout-box {{ background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 12px;
                font-family: 'Fira Code', 'Consolas', monospace; font-size: 12px;
                overflow-x: auto; white-space: pre-wrap; margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15); max-height: 500px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>Test Report</h1>
  <p class="subtitle">ViT + LoRA Image Classification Pipeline</p>

  <div class="stats">
    <div class="stat-card total"><div class="number">{s['total']}</div><div class="label">Total Tests</div></div>
    <div class="stat-card passed"><div class="number">{s['passed']}</div><div class="label">Passed</div></div>
    <div class="stat-card failed"><div class="number">{s.get('failed', 0)}</div><div class="label">Failed</div></div>
    <div class="stat-card rate"><div class="number">{pass_rate:.0f}%</div><div class="label">Pass Rate</div></div>
    <div class="stat-card time"><div class="number">{s.get('duration', 0):.2f}s</div><div class="label">Duration</div></div>
  </div>

  <div class="charts">
    <div class="chart-card"><img src="{embed_image(chart_paths['summary'])}" alt="Summary"></div>
    <div class="chart-card"><img src="{embed_image(chart_paths['by_module'])}" alt="By Module"></div>
    <div class="chart-card full"><img src="{embed_image(chart_paths['status_grid'])}" alt="Status Grid"></div>
  </div>

  <div class="section-title">Test Details</div>
  <table>
    <thead><tr><th>Module</th><th>Class</th><th>Test Name</th><th>Status</th></tr></thead>
    <tbody>{test_rows}</tbody>
  </table>

  <div class="section-title">Console Output</div>
  <div class="stdout-box">{data.get('stdout', 'N/A')}</div>
</div>
</body>
</html>"""

    path = REPORT_DIR / "test_report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [html]  {path}")
    return path


def main():
    print("=" * 60)
    print("  Running tests and generating visual report")
    print("=" * 60)

    print("\n[1/2] Running pytest...")
    data = run_tests()

    s = data["summary"]
    print(f"\n  Total: {s['total']}  |  Passed: {s['passed']}  |  "
          f"Failed: {s.get('failed', 0)}  |  Duration: {s.get('duration', 0):.2f}s")

    print("\n[2/2] Generating charts and report...")
    charts = {
        "summary": plot_summary(data),
        "by_module": plot_by_module(data),
        "status_grid": plot_test_status_grid(data),
    }

    html_path = generate_html_report(data, charts)

    print(f"\n{'=' * 60}")
    print(f"  Report ready: {html_path}")
    print(f"{'=' * 60}")

    return data


if __name__ == "__main__":
    main()
