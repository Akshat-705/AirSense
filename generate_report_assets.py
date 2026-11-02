import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".venv")
os.makedirs("report_assets", exist_ok=True)

# 1. Model Performance Comparison Chart
models = ['Random Forest', 'Gradient Boosting', 'Linear Regression']
mae = [23.28, 24.40, 31.40]
rmse = [42.99, 44.16, 59.71]
r2 = [0.899, 0.894, 0.805]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(7, 4), dpi=300)
plt.bar(x - width, mae, width, label='MAE (Lower is Better)', color='#2b5c8f')
plt.bar(x, rmse, width, label='RMSE (Lower is Better)', color='#e06666')
plt.bar(x + width, [val * 50 for val in r2], width, label='R² (scaled x50, Higher is Better)', color='#6aa84f')

plt.xlabel('Regression Models', fontweight='bold', fontsize=11)
plt.ylabel('Error / Score Metric Value', fontweight='bold', fontsize=11)
plt.title('Figure 4.1: Model Performance Comparison (MAE, RMSE, R²)', fontweight='bold', fontsize=12)
plt.xticks(x, models, fontweight='bold')
plt.legend(frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("report_assets/fig_model_comparison.png")
plt.close()

# 2. Feature Importance Chart
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
importance = [52.4, 28.1, 11.2, 5.3, 3.0]
features.reverse()
importance.reverse()

plt.figure(figsize=(7, 3.8), dpi=300)
bars = plt.barh(features, importance, color='#3c78d8', edgecolor='navy', height=0.55)
plt.xlabel('Relative Feature Importance (%)', fontweight='bold', fontsize=11)
plt.ylabel('Pollutant Features', fontweight='bold', fontsize=11)
plt.title('Figure 3.2: Pollutant Feature Importance in AQI Estimation', fontweight='bold', fontsize=12)
plt.xlim(0, 60)
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2 - 0.08, f"{bar.get_width()}%", fontweight='bold', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("report_assets/fig_feature_importance.png")
plt.close()

# 3. Graphical Abstract Diagram
fig, ax = plt.subplots(figsize=(8.5, 2.5), dpi=300)
ax.axis('off')

boxes = [
    ("Client Request\n(Browser UI)", "#e8f0fe", "#1a73e8"),
    ("Flask Web App\n(/filter, /api/live)", "#fce8e6", "#d93025"),
    ("ML Engine\n(Random Forest)", "#e6f4ea", "#188038"),
    ("Data & APIs\n(city_data / OpenWeather)", "#fef7e0", "#f9ab00"),
    ("Response & Plots\n(AQI, Health Advice)", "#f3e8fd", "#9334e6")
]

for i, (text, bg, border) in enumerate(boxes):
    x_pos = 0.02 + i * 0.20
    rect = FancyBboxPatch((x_pos, 0.15), 0.16, 0.7, boxstyle="round,pad=0.02",
                          facecolor=bg, edgecolor=border, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x_pos + 0.08, 0.5, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#202124', transform=ax.transAxes)
    if i < len(boxes) - 1:
        ax.annotate('', xy=(x_pos + 0.198, 0.5), xytext=(x_pos + 0.162, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#5f6368'), transform=ax.transAxes)

plt.tight_layout()
plt.savefig("report_assets/fig_graphical_abstract.png")
plt.close()

# 4. Three-Tier Architecture
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
ax.axis('off')

tiers = [
    ("Presentation Layer", "HTML5 Dashboard UI, Jinja2 Templates, Chart.js Time-Series Plots, Dynamic Health Badges", "#e1f5fe", "#0288d1", 0.7),
    ("Application / Logic Layer", "Flask Web Server, REST API Routing, Scikit-Learn Inference Pipeline, SimpleImputer Preprocessing", "#ede7f6", "#512da8", 0.4),
    ("Data & Services Layer", "Indian Cities Historical CSV Dataset (PM2.5, PM10, NO2, CO, SO2), OpenWeather Live Air Pollution API", "#e8f5e9", "#2e7d32", 0.1)
]

for title, desc, bg, border, y_pos in tiers:
    rect = FancyBboxPatch((0.05, y_pos), 0.9, 0.22, boxstyle="round,pad=0.02",
                          facecolor=bg, edgecolor=border, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, y_pos + 0.14, title, ha='center', va='center', fontsize=11, fontweight='bold', color=border, transform=ax.transAxes)
    ax.text(0.5, y_pos + 0.06, desc, ha='center', va='center', fontsize=8.5, color='#333333', transform=ax.transAxes)
    if y_pos > 0.2:
        ax.annotate('', xy=(0.5, y_pos - 0.06), xytext=(0.5, y_pos),
                    arrowprops=dict(arrowstyle="<->", lw=2, color='#555555'), transform=ax.transAxes)

plt.tight_layout()
plt.savefig("report_assets/fig_architecture.png")
plt.close()

print("All asset figures generated successfully!")
