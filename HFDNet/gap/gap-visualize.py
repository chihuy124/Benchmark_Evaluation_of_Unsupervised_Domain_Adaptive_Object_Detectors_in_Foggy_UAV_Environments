import json
import os
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适用于远程服务器
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_losses(json_file, output_dir="vis", sample_size=20):
    """
    读取 JSON 文件中各实验的指标数据，随机选择 50 个实验（如果不足 50 个则选择全部），
    分别生成各个 loss 的柱状图，每个指标生成一张图，不在柱子上显示数值，图表简单大方、易于直观对比。
    
    参数:
        json_file (str): 存储结果的 JSON 文件路径。
        output_dir (str): 保存图像的目录，默认为 "vis"。
        sample_size (int): 随机采样的实验数量，默认为 50。
    """
    # 读取 JSON 数据
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 获取所有实验/子文件夹的名称
    all_subs = list(results.keys())
    viz_subs = all_subs.copy()  # 用于可视化的样本
    
    # 随机采样指定数量的实验用于可视化（如果不足则选择全部）
    if len(all_subs) > sample_size:
        viz_subs = random.sample(all_subs, sample_size)
    
    # 定义需要绘制的指标列表
    loss_types = [
        "mean_gram_loss", 
        "mean_l2_loss", 
        "mean_gaussianmmd_loss", 
        "mean_linearmmd_loss", 
        "mean_dss_loss", 
        "mean_swd_loss"
    ]
    
    # 为所有样本提取指标数据（用于统计）
    all_data = {loss_type: [] for loss_type in loss_types}
    for sub in all_subs:
        for loss_type in loss_types:
            all_data[loss_type].append(results[sub].get(loss_type, None))
    
    # 为可视化样本提取指标数据
    viz_data = {loss_type: [] for loss_type in loss_types}
    for sub in viz_subs:
        for loss_type in loss_types:
            viz_data[loss_type].append(results[sub].get(loss_type, None))
    
    # 统计所有样本的每个 loss 类型的最小值和最大值
    loss_stats = {}
    for loss_type in loss_types:
        valid_values = [v for v in all_data[loss_type] if v is not None]
        if valid_values:
            loss_stats[loss_type] = {
                "min": min(valid_values),
                "max": max(valid_values),
                "mean": np.mean(valid_values),
                "median": np.median(valid_values),
                "num_samples": len(valid_values)
            }
        else:
            loss_stats[loss_type] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "num_samples": 0
            }
    
    # 将统计结果保存到 JSON 文件
    stats_file = os.path.join(output_dir, "loss_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(loss_stats, f, indent=4, ensure_ascii=False)
    print(f"已保存 loss 统计结果 (基于全部 {len(all_subs)} 个样本): {stats_file}")

    
    # 设置 Seaborn 主题
    # sns.set_theme(style="whitegrid")
    sns.set_theme(style="white")
    
    # 动态设置图像宽度：每个实验占 0.8 英寸，最小宽度 8 英寸，最大不超过 20 英寸
    width_inches = min(max(8, len(viz_subs) * 0.8), 20)
    
    # 针对每个 loss 单独绘制一张图
    for loss_type in loss_types:
        plt.figure(figsize=(width_inches, 6))
        # 绘制柱状图
        sns.barplot(x=viz_subs, y=viz_data[loss_type], hue=viz_subs, palette="viridis", dodge=False, legend=False)
        
        # Simplified y-axis label mapping
        loss_name_map = {
            "mean_gram_loss": "Gram Gap",
            "mean_l2_loss": "L2 Gap",
            "mean_gaussianmmd_loss": "MMD Gap",
            "mean_linearmmd_loss": "L-MMD Gap",
            "mean_dss_loss": "DSS Gap",
            "mean_swd_loss": "SWD Gap"
        }
        # plt.title(f"{loss_type.replace('_', ' ').title()} Across Sub-sample", fontsize=14)
        plt.xlabel("Sub-sample", fontsize=12)
        plt.ylabel(loss_name_map[loss_type], fontsize=12)  # Use simplified name
        
        # 仅显示部分索引，防止过于密集
        step = max(1, len(viz_subs) // 10)  # 每 10% 取一个刻度
        plt.xticks(ticks=np.arange(len(viz_subs))[::step], labels=np.arange(len(viz_subs))[::step])

        plt.tight_layout()
        # 保存图像，每个指标保存为独立文件
        out_path = os.path.join(output_dir, f"{loss_type}_summary.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"已可视化 {len(viz_subs)}/{len(all_subs)} 个样本的 {loss_type} 图: {out_path}")

# 示例调用：
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/c2f_gap.json", output_dir="vis/city_to_foggy")
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/s2c_gap.json", output_dir="vis/sim10k_to_city")
# visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/v2c_gap.json", output_dir="vis/voc_to_clipart1k")
# # visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/pr2pu_gap.json", output_dir="vis/privatepower_to_publicpower")
visualize_losses("/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/pu2pr_gap.json", output_dir="vis/pupower_to_prpower")
