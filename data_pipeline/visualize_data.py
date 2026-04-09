"""
Data Visualization — Embedding-based clustering and distribution analysis.
Uses sentence embeddings to visualize how diverse the training data is.
Generates an HTML report with interactive charts.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_texts(data: list) -> tuple[list[str], list[str]]:
    """Extract assistant responses and their scenario labels."""
    texts = []
    labels = []
    for convo in data:
        scenario = convo.get("scenario", "unknown")
        for msg in convo.get("conversations", []):
            if msg["role"] == "assistant":
                texts.append(msg["content"])
                labels.append(scenario)
    return texts, labels


def compute_stats(data: list) -> dict:
    """Compute basic dataset statistics."""
    scenarios = Counter(d.get("scenario", "unknown") for d in data)
    turn_counts = []
    assistant_lengths = []

    for convo in data:
        msgs = convo.get("conversations", [])
        turns = sum(1 for m in msgs if m["role"] in ("user", "assistant"))
        turn_counts.append(turns)
        for m in msgs:
            if m["role"] == "assistant":
                assistant_lengths.append(len(m["content"]))

    return {
        "total_conversations": len(data),
        "scenario_distribution": dict(scenarios.most_common()),
        "avg_turns": np.mean(turn_counts) if turn_counts else 0,
        "avg_response_length": np.mean(assistant_lengths) if assistant_lengths else 0,
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
    }


def generate_html_report(stats: dict, output_path: str):
    """Generate an HTML report with embedded charts using Chart.js."""
    scenarios = stats["scenario_distribution"]
    labels = list(scenarios.keys())
    values = list(scenarios.values())

    # Generate distinct colors for each scenario
    colors = [
        f"hsl({int(i * 360 / len(labels))}, 70%, 55%)"
        for i in range(len(labels))
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Emotional AI — Data Distribution Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; padding: 40px; }}
        h1 {{ color: #58a6ff; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-card .number {{ font-size: 2.5em; font-weight: bold; color: #58a6ff; }}
        .stat-card .label {{ color: #8b949e; margin-top: 5px; }}
        .chart-container {{ background: #161b22; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Emotional AI — Training Data Report</h1>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="number">{stats['total_conversations']}</div>
            <div class="label">Total Conversations</div>
        </div>
        <div class="stat-card">
            <div class="number">{stats['avg_turns']:.1f}</div>
            <div class="label">Avg Turns / Conversation</div>
        </div>
        <div class="stat-card">
            <div class="number">{stats['avg_response_length']:.0f}</div>
            <div class="label">Avg Response Length (chars)</div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="scenarioChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <canvas id="scenarioPie" height="80"></canvas>
    </div>

    <script>
        new Chart(document.getElementById('scenarioChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Conversations per Scenario',
                    data: {json.dumps(values)},
                    backgroundColor: {json.dumps(colors)},
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 45 }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('scenarioPie'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    data: {json.dumps(values)},
                    backgroundColor: {json.dumps(colors)}
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'right', labels: {{ color: '#e6edf3' }} }} }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {output_path}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "emotional_training_data_v2.json"
    report_path = "data_analysis_report.html"

    print("=" * 60)
    print("Data Distribution Analyzer")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = compute_stats(data)

    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Average turns: {stats['avg_turns']:.1f}")
    print(f"Average response length: {stats['avg_response_length']:.0f} chars")
    print(f"\nScenario distribution:")
    for scenario, count in stats["scenario_distribution"].items():
        bar = "█" * (count // 10)
        print(f"  {scenario:30s} {count:4d} {bar}")

    generate_html_report(stats, report_path)


if __name__ == "__main__":
    main()
