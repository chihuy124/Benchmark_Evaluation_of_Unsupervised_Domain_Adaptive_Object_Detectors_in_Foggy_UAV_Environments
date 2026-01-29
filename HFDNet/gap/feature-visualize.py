import json

def calculate_global_layer_strengths(json_path):
    """
    统计JSON文件中多个layer_comparison的层级特征全局均值
    返回: (global_shallow, global_middle, global_deep)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 初始化全局统计
    shallow_sum, middle_sum, deep_sum = 0.0, 0.0, 0.0
    shallow_count, middle_count, deep_count = 0, 0, 0
    
    # 遍历所有条目
    for entry in data.values():
        if "layer_comparison" in entry:
            lc = entry["layer_comparison"]
            
            if "shallow" in lc:
                shallow_sum += lc["shallow"]["mean_strength"]
                shallow_count += 1
            
            if "middle" in lc:
                middle_sum += lc["middle"]["mean_strength"]
                middle_count += 1
                
            if "deep" in lc:
                deep_sum += lc["deep"]["mean_strength"]
                deep_count += 1
    
    # 计算全局均值
    global_shallow = shallow_sum / shallow_count if shallow_count > 0 else 0.0
    global_middle = middle_sum / middle_count if middle_count > 0 else 0.0
    global_deep = deep_sum / deep_count if deep_count > 0 else 0.0
    
    print("\n=== 全局层级特征强度 ===")
    print(f"浅层全局均值: {global_shallow:.6f} (来自 {shallow_count} 组数据)")
    print(f"中层全局均值: {global_middle:.6f} (来自 {middle_count} 组数据)")
    print(f"深层全局均值: {global_deep:.6f} (来自 {deep_count} 组数据)")
    # print(f"浅/深比例: {global_shallow/global_deep:.2f}:1")
    
    return global_shallow, global_middle, global_deep

# calculate_global_layer_strengths('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/feature/c2f.json')
# === 全局层级特征强度 ===
# 浅层全局均值: 0.060454 (来自 500 组数据)
# 中层全局均值: 0.036881 (来自 500 组数据)
# 深层全局均值: 0.030127 (来自 500 组数据)

# calculate_global_layer_strengths('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/feature/s2c.json')
# === 全局层级特征强度 ===
# 浅层全局均值: 0.061881 (来自 500 组数据)
# 中层全局均值: 0.039876 (来自 500 组数据)
# 深层全局均值: 0.030893 (来自 500 组数据)

# calculate_global_layer_strengths('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/feature/v2c.json')
# === 全局层级特征强度 ===
# 浅层全局均值: 0.062205 (来自 500 组数据)
# 中层全局均值: 0.038175 (来自 500 组数据)
# 深层全局均值: 0.029958 (来自 500 组数据)
calculate_global_layer_strengths('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/gap/feature/pu2pr.json')
# === 全局层级特征强度 ===
# 浅层全局均值: 0.061800 (来自 102 组数据)
# 中层全局均值: 0.038754 (来自 102 组数据)
# 深层全局均值: 0.030847 (来自 102 组数据)