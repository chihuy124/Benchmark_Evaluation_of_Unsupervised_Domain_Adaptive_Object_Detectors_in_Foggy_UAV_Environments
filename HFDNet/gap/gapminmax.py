import json
import matplotlib.pyplot as plt
import numpy as np
import os

# JSON 文件路径映射
json_files = [
    "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/vis/city_to_foggy/loss_stats.json",
    "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/vis/voc_to_clipart1k/loss_stats.json",
    "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/vis/sim10k_to_city/loss_stats.json",
    "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/vis/privatepower_to_publicpower/loss_stats.json"
]

# 定义 loss 类型和简化标签
loss_types = ['mean_l2_loss', 'mean_gaussianmmd_loss', 'mean_dss_loss']
loss_types_simplified = ['l2', 'gmmd', 'dss']  # 简化标签

# 定义数据集对应的简化名称
legend_names = {
    "city_to_foggy": "c2f",
    "voc_to_clipart1k": "v2c",
    "sim10k_to_city": "s2c",
    "privatepower_to_publicpower": "pr2pu"
}

# 读取 JSON 数据
def read_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 遍历每种 loss 类型，分别绘制图表
for loss_idx, loss in enumerate(loss_types):
    plt.figure(figsize=(8, 6))
    
    # 设定柱状图宽度
    bar_width = 0.2
    x = np.arange(len(json_files))  # 不同数据集的 x 轴位置
    
    # 获取该 loss 的 min 和 max 值，并用透明度表示变化
    for i, file in enumerate(json_files):
        data = read_json_data(file)
        file_name = os.path.basename(os.path.dirname(file))  # 获取父目录名称
        legend_name = legend_names.get(file_name, file_name)  # 获取简化的图例名称
        
        # 获取当前 loss 的 min 和 max 值
        min_value = data[loss]['min']
        max_value = data[loss]['max']
        print(f'文件 {legend_name} | {loss}: min={min_value}, max={max_value}')
        
        # 绘制柱状图，透明度由浅到深表示值变化
        alpha_value = (max_value - min_value) / max_value  # 用最大值与最小值差值决定透明度
        plt.bar(x[i], min_value, width=bar_width, label=f'{legend_name} (min)', alpha=0.2 + alpha_value * 0.8)  # 透明度范围为0.2到1.0
        plt.bar(x[i], max_value - min_value, width=bar_width, bottom=min_value, label=f'{legend_name} (max)', alpha=0.2 + alpha_value * 0.8)
    
    # 设置图表标题
    plt.title(f'{loss_types_simplified[loss_idx]} Loss Comparison', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    
    # 设置横轴刻度和标签
    plt.xticks(x, [legend_names.get(os.path.basename(os.path.dirname(f)), os.path.basename(os.path.dirname(f))) for f in json_files], rotation=20, fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 显示网格
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(f"loss_comparison_{loss_types_simplified[loss_idx]}.png", dpi=300, bbox_inches="tight")
    plt.close()  # 关闭图表，避免多次显示
