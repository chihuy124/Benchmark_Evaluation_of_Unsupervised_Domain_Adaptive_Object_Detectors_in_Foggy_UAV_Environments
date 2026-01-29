import torch
import ot
# ot 是 POT（Python Optimal Transport）库的一个常用的别名。POT 是一个 Python 库，
# 专门用于处理最优输运问题，其中包括计算 Earth Mover’s Distance（EMD）距离。
from torch import cdist
from ultralytics import YOLO
import yaml
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import random

start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 固定随机种子
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

class InputDataset(Dataset):
    def __init__(self, input_dir,  img_size=640):
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.img_size = img_size

    def __len__(self):
        return len(self.input_files)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        return torch.tensor(image).permute(2, 0, 1).float()  # [C, H, W]

    def __getitem__(self, idx):
        input = self.preprocess_image(self.input_files[idx])
        return input

class YOLOFeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers_to_extract, config_path=None):
        super(YOLOFeatureExtractor, self).__init__()
        self.model = model.model
        self.layers_to_extract = layers_to_extract
        self.extracted_features = {}
        self.concat_map, self.detect_map = self._parse_yaml(config_path)

    def _parse_yaml(self, config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        concat_map = {}
        detect_map = {}
        all_layers = cfg["backbone"] + cfg["head"]
        for idx, layer in enumerate(all_layers):
            if "Concat" in layer[2]:
                concat_map[idx] = layer[0]
            if "Detect" in layer[2]:
                detect_map[idx] = layer[0]
        return concat_map, detect_map

    def forward(self, x):
        outputs = {}
        self.extracted_features = {}
        # print('forward begin')
        for idx, layer in enumerate(self.model):
            # print(f"\nLayer {idx}: {layer.__class__.__name__}")
            # print(f"Input shape: {x.shape if isinstance(x, torch.Tensor) else [t.shape for t in x]}")
            # 动态处理 Concat 层
            if idx in self.concat_map:
                concat_inputs = []
                for i in self.concat_map[idx]:
                    if i == -1:
                        concat_inputs.append(x)  # x1 当前流的上一层输出
                    else:
                        concat_inputs.append(outputs[i])  # 特定层的输出
                x = layer(concat_inputs)
                # print(f"Concat input shapes: {[t.shape for t in concat_inputs]}")
            elif idx in self.detect_map:
                detect_inputs = [outputs[i] for i in self.detect_map[idx]]
                x = layer(detect_inputs)
                # print(f"Detect input shapes: {[t.shape for t in detect_inputs]}")
            else:
                x = layer(x)
                # print(f"Output shape: {x.shape if isinstance(x, torch.Tensor) else [t.shape for t in x]}")
            # 保存当前层输出
            outputs[idx] = x
            if idx in self.layers_to_extract:
                self.extracted_features[f"layer_{idx}"] = x.clone()
        # print('forward end')
        return x,  self.extracted_features

def gram_matrix(input):
    # 确保输入是 torch.Tensor
    print("Gram input size:", input.shape)  # 检查输入张量的形状 [B, C, H, W] = (300, 96, 160, 160) 
    a, b, c, d = input.shape
    features = input.view(a * b, c * d) # 展平特征
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d) # 归一化

def compute_style_loss(input_feature, target_feature):
    G_input = gram_matrix(input_feature.detach())  # detach 防止梯度计算
    G_target = gram_matrix(target_feature.detach())
    return F.mse_loss(G_input, G_target)

def compute_l2_loss(input_feature, target_feature):
    return F.mse_loss(input_feature.detach(), target_feature.detach())

def load_data(source, target, batch_size):
    # g = torch.Generator()
    # g.manual_seed(42)
    source_data_loader = DataLoader(
        InputDataset(source), batch_size=batch_size, shuffle=False, 
        # num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    target_data_loader = DataLoader(
        InputDataset(target), batch_size=batch_size, shuffle=False, 
        # num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    return source_data_loader, target_data_loader

def initialize_yolo_model(yolo_weights_path, config_path, layers_to_extract):
    yolo_model = YOLO(yolo_weights_path).model
    feature_extractor = YOLOFeatureExtractor(yolo_model, layers_to_extract, config_path).to(device)
    feature_extractor.eval()  # 确保 batchnorm 处于 eval 模式
    return feature_extractor

def extract_features(feature_extractor, data_loader, layers_to_extract, n_samples=-1):
    print(f"Start extracting features from {n_samples if n_samples != -1 else len(data_loader)} batches")
    # 如果未指定批次限制，使用数据加载器的长度
    total = len(data_loader) if n_samples == -1 else n_samples # 1000
    # 初始化存储特征的变量
    projected_features = {f"layer_{layer}": [] for layer in layers_to_extract} 
    for input, batch_idx in zip(data_loader,range(total)):
        if batch_idx % 10 == 1:
            print("Process image %d/%d" % (batch_idx, total))  
        # 将数据移动到 GPU 并提取特征
        input = input.to(device)
        _, features = feature_extractor(input)
        # 遍历所有目标层
        for layer in layers_to_extract:
            layer_key = f"layer_{layer}"
            # 提取特定层的特征
            input_features = features[layer_key]  # 直接取出特征
            projected_features[layer_key].append(input_features.detach().cpu())  # 保持为 tensor
    # **拼接 tensor 而不是 numpy**
    for layer_key in projected_features:
        projected_features[layer_key] = torch.cat(projected_features[layer_key], dim=0)  # 拼接保持 tensor 格式
    return projected_features

def compute_domain_gap(source_data, target_data, yolo_weights_path, config_path, style_layers, content_layers, n_samples=50):
    source_loader, target_loader = load_data(source_data, target_data, batch_size=6)
    feature_extractor = initialize_yolo_model(yolo_weights_path, config_path, style_layers + content_layers)
    source_features = extract_features(feature_extractor, source_loader, style_layers + content_layers, n_samples)
    target_features = extract_features(feature_extractor, target_loader, style_layers + content_layers, n_samples)
    gram_loss = sum(compute_style_loss(source_features[f"layer_{l}"], target_features[f"layer_{l}"]) for l in style_layers)
    print('gram_loss finish')
    l2_loss = sum(compute_l2_loss(source_features[f"layer_{l}"], target_features[f"layer_{l}"]) for l in content_layers)
    print('l2_loss finish')
    return gram_loss.item(), l2_loss.item()

# 数据路径

print('city_to_foggy')
source = "/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/train"
target = "/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format/images/train"
yolo_weights_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/sourcecity/weights/best.pt'
# Gram loss (style gap): 1.7628055048665714e-12, L2 loss (semantic gap): 0.5793765783309937

# print('sim10k_to_city')
# source = "/home/lenovo/data/liujiaji/DA-Datasets/Sim10k/images/train"
# target = "/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format_car_class/images/train"
# yolo_weights_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/sourcesim10k/weights/best.pt'
# # Gram loss (style gap): 4.66217255773449e-12, L2 loss (semantic gap): 0.6599050164222717

# print('private powerdata_to_public powerdata')
# source = "/home/lenovo/data/liujiaji/yolov8/privatepower/images/train"
# target = "/home/lenovo/data/liujiaji/Datasets/publicpower/images/train"
# yolo_weights_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/sourceprivate/weights/best.pt'
# # # Gram loss (style gap): 9.969383825414457e-12, L2 loss (semantic gap): 0.7197887301445007

config_path = "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/ultralytics/cfg/models/v8/yolov8.yaml"
style_layers = [2,4,6]
content_layers = [8,9]

style_gap, l2_gap = compute_domain_gap(source, target, yolo_weights_path, config_path, style_layers, content_layers)
print(f"Gram loss (style gap): {style_gap}" )
print(f"L2 loss (semantic gap): {l2_gap} " )

end_time = time.time()
print(f"程序耗时：{end_time - start_time}秒")



