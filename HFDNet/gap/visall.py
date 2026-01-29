import json
import os
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适用于远程服务器
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_losses(json_file, output_dir="vis"):
    """
    读取 JSON 文件中各实验的指标数据，将所有 loss 绘制在一张图上，
    每个指标生成一个子图，图表简单大方、易于直观对比。
    
    参数:
        json_file (str): 存储结果的 JSON 文件路径。
        output_dir (str): 保存图像的目录，默认为 "vis"。
    """
    # 读取 JSON 数据
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有实验/子文件夹的名称
    subs = list(results.keys())
    sub_indices = np.arange(len(subs))  # 用索引代替实验名称
    
    # 定义需要绘制的指标列表
    loss_types = [
        "mean_gram_loss", 
        "mean_l2_loss", 
        # "mean_gaussianmmd_loss", 
        "mean_linearmmd_loss", 
        "mean_dss_loss", 
        "mean_swd_loss"
    ]
    
    # 为每个指标提取对应的数值
    data = {loss_type: [] for loss_type in loss_types}
    for sub in subs:
        for loss_type in loss_types:
            # 如果某个指标缺失，则设置为 None 或者 0，这里采用 None
            data[loss_type].append(results[sub].get(loss_type, None))
    
    # 设置 Seaborn 主题
    sns.set_theme(style="whitegrid")
    
    # 动态设置图像宽度：每个实验占 0.8 英寸，最小宽度 8 英寸，最大不超过 20 英寸
    width_inches = min(max(8, len(subs) * 0.8), 20)
    
    # 创建一个大图，包含多个子图
    fig, axes = plt.subplots(len(loss_types), 1, figsize=(width_inches, 6 * len(loss_types)))
    fig.suptitle("Loss Comparison Across Sub-folders", fontsize=16, y=1.02)
    
    # 针对每个 loss 绘制一个子图
    for i, loss_type in enumerate(loss_types):
        ax = axes[i]
        # 绘制柱状图
        sns.barplot(x=subs, y=data[loss_type], hue=subs, palette="viridis", dodge=False, legend=False, ax=ax)
        ax.set_title(f"{loss_type.replace('_', ' ').title()}", fontsize=14)
        ax.set_xlabel("Sub-folder", fontsize=12)
        ax.set_ylabel("Loss Value", fontsize=12)
        # 仅显示部分索引，防止过于密集
        step = max(1, len(sub_indices) // 10)  # 每 10% 取一个刻度
        ax.set_xticks(ticks=sub_indices[::step])
        ax.set_xticklabels(sub_indices[::step])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    out_path = os.path.join(output_dir, "all_losses_summary.png")
    plt.savefig(out_path, dpi=500)
    plt.close()
    print(f"已保存所有 loss 图: {out_path}")

# 示例调用：
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/city_to_foggy_gap.json", output_dir="vis/city_to_foggy")
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/sim10k_to_city_gap.json", output_dir="vis/sim10k_to_city")
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/voc_to_clipart1k_gap.json", output_dir="vis/voc_to_clipart1k")
visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/privatepower_to_publicpower_gap.json", output_dir="vis/privatepower_to_publicpower")
