import os
import json
import matplotlib.pyplot as plt

# Defaults (used when no external JSON is provided, or fields are missing)
DEFAULTS = {
    "labels": [
        "3D position resolution [cm]",
        "Transverse position resolution [cm]",
        "Longitudinal position resolution [cm]",
        "Longitudinal position mean bias [cm]",
        "Direction resolution [°]",
        "Momentum resolution [%]",
        "Momentum mean bias [%]",
        "FC classification efficiency",
    ],
    "col1_name": "Muon",
    "col2_name": "Electron",
    # Use None for blanks so they render as "/"
    "col1_data": [6.46, 5.67, 2.69, 0.17, 2.17, 2.48, 0.21, None],
    "col2_data": [6.83, 4.96, 4.17, 0.63, 4.12, 8.80, -2.00, 0.93],
    "output_path": "/home/zhihao/WCTE_2024.png",
    "figsize": [6, 4],
    "row_height": 0.08,
}

# External data path: env var TABLE_PLOT_DATA takes precedence; otherwise a JSON next to this script
DATA_PATH = os.environ.get(
    "TABLE_PLOT_DATA",
    os.path.join(os.path.dirname(__file__), "table_plot_data.json"),
)

def load_config(path: str):
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise ValueError("JSON root must be an object/dict")
            # Shallow-merge over defaults so you can override only parts
            merged = {**DEFAULTS, **cfg}
            return merged
        except Exception as e:
            print(f"[table_plot] Failed to load {path}: {e}. Using defaults.")
    return DEFAULTS

cfg = load_config(DATA_PATH)

labels = cfg["labels"]
col1_name = cfg["col1_name"]
col2_name = cfg["col2_name"]
col1_data = cfg["col1_data"]
col2_data = cfg["col2_data"]
output_path = cfg["output_path"]
figsize = tuple(cfg.get("figsize", DEFAULTS["figsize"]))
row_height = float(cfg.get("row_height", DEFAULTS["row_height"]))

# Start plotting
fig, ax = plt.subplots(figsize=figsize)
ax.axis("off")

start_y = 1.0

# Headers
ax.text(0.70, start_y, col1_name, fontsize=10, fontweight='bold', ha='center')
ax.text(0.90, start_y, col2_name, fontsize=10, fontweight='bold', ha='center')

# Helper for formatting cells
def fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "/"

# Body rows
for i, label in enumerate(labels):
    y = start_y - (i + 1) * row_height
    ax.text(0.00, y, label, fontsize=10, ha='left')

    v1 = col1_data[i] if i < len(col1_data) else None
    v2 = col2_data[i] if i < len(col2_data) else None

    ax.text(0.70, y, fmt(v1), fontsize=10, ha='center')
    ax.text(0.90, y, fmt(v2), fontsize=10, ha='center')

# Save
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")