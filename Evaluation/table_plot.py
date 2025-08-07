import matplotlib.pyplot as plt

# 指标列表
labels = [
  "3D position resolution [cm]",
  "Transverse position resolution [cm]",
  "Longitudinal position resolution [cm]",
  "Longitudinal position mean bias [cm]",
  "Direction resolution [°]",
  "Momentum resolution [%]",
  "Momentum mean bias [%]",
  "FC classification efficiency",
]

# 两列数据
col1_name = "Muon"
col2_name = "Electron"

col1_data = [6.46, 5.67, 2.69, 0.17, 2.17, 2.48, 0.21, ""]
col2_data = [6.83, 4.96, 4.17, 0.63, 4.12, 8.80, -2.00, 0.93]

# 开始画图
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")

# 行高设置
row_height = 0.08
start_y = 1.0

# 表头
ax.text(0.70, start_y, col1_name, fontsize=10, fontweight='bold', ha='center')
ax.text(0.90, start_y, col2_name, fontsize=10, fontweight='bold', ha='center')

# 主体行
for i, label in enumerate(labels):
  y = start_y - (i + 1) * row_height
  ax.text(0.00, y, label, fontsize=10, ha='left')
  val1 = f"{col1_data[i]:.2f}" if isinstance(col1_data[i], (int, float)) else "/"
  val2 = f"{col2_data[i]:.2f}" if isinstance(col2_data[i], (int, float)) else "/"
  ax.text(0.70, y, val1, fontsize=10, ha='center')
  ax.text(0.90, y, val2, fontsize=10, ha='center')

# 保存图片
plt.tight_layout()
plt.savefig("/home/zhihao/WCTE_2024.png", dpi=300, bbox_inches="tight")