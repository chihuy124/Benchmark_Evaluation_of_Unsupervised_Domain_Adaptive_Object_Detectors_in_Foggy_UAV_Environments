import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# import itertools 
# import random
# # 定义搜索空间
# gamma_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5 ]  # gamma_weight 的候选值
# alpha_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # alpha_weight 的候选值
# lambda_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # lambda_weight 的候选值

# # # 生成所有可能的组合 网格搜索
# # param_grid = list(itertools.product(gamma_weights, alpha_weights, lambda_weights ))
# # # 生成所有可能的组合，并过滤掉不满足 alpha_weight < lambda_weight 的组合
# # param_grid = [
# #     (gamma, alpha, lambda_ )
# #     for gamma, alpha, lambda_ in itertools.product(gamma_weights, alpha_weights, lambda_weights)
# #     if alpha < lambda_  # 确保 alpha_weight < lambda_weight
# # ]
# # 随机采样次数 随机搜索
# n_iter = 3  # 随机采样 20 次
# # 随机采样参数组合
# param_samples = [
#     (
#         random.choice(gamma_weights),
#         random.choice(alpha_weights),
#         random.choice(lambda_weights),
#     )
#     for _ in range(n_iter)
# ]

# best_mAP = 0  # 记录最佳验证集性能
# best_params = None  # 记录最佳参数组合
    
# for gamma_weight, alpha_weight, lambda_weight in param_samples:
    
#     model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
#     # model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp2/weights/last.pt') # 断点续训
#     # 伪标签使用的 源域 pre-trained weight
#     # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcecity/weights/best.pt') # loading pretrain weights
#     # COCO pre-trained weight
#     model.load('yolov8m.pt')
#     result = model.train(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/city_to_foggycity.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=1,
#                 batch=8, # 32
#                 close_mosaic=10, 
#                 workers=8,# 4
#                 # device='0',
#                 optimizer='SGD', # using SGD
#                 patience=0, # set 0 to close earlystop.
#                 resume=True, # 断点续训,YOLO初始化时选择last.pt
#                 amp=False, # close amp
#                 # half=False,
#                 # fraction=0.2,
#                 cos_lr = True,
#                 project='runs/debug',
#                 # project='runs/train/improve',
#                 name='sourcecity',
#                 # mixup = 1.0,
#                 # mosaic = 0.0,

#                 # 设置自定义损失权重
#                 gamma_weight = gamma_weight,
#                 alpha_weight = alpha_weight,
#                 lambda_weight = lambda_weight
#                 )
#     # # 验证集性能
#     if result.results_dict['metrics/mAP50(B)'] > best_mAP :
#         best_mAP = result.results_dict['metrics/mAP50(B)']
#         best_params = (gamma_weight, alpha_weight, lambda_weight)
#     print(f"Params: gamma={gamma_weight}, alpha={alpha_weight}, lambda={lambda_weight}, Val mAP50: {val_mAP}")

# # 输出最优结果
# print(f"Best mAP50: {best_mAP}")
# print(f"Best parameters (gamma, alpha, lambda): {best_params}")

        # city_to_foggycity.yaml sourcecity
        # foggycityscapes.yaml oraclefoggy
        
        # sim10k_to_cityscapes.yaml sourcesim10k
        # cityscapes.yaml oraclecity

        # voc_to_clipart1k.yaml sourcevoc
        # clipart1k.yaml oracleclipart1k
        
        # pupower_to_prpower.yaml sourcepu
        # prpower.yaml oraclepr

# # 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings



import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from torchvision import transforms
from PIL import Image

feature_maps = []

# Hook function to capture feature map
def hook_fn(module, input, output):
    global feature_maps
    feature_maps.append(output)

# Function to add hook to a specific layer
def add_hook(model):
    target_layer = model.model.model[0]  # Change this to the layer of interest
    target_layer.register_forward_hook(hook_fn)
    
    
    
    
    
if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    # model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp2/weights/last.pt') # 断点续训
    # 域适应会使用 源域 pre-trained weight
    # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcepu2/weights/best.pt') # loading pretrain weights
    # COCO pre-trained weight
    # model.load('yolov8m.pt')
    
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    
        
    print("model: ", model.model.model[0])
    add_hook(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    img_path = "/home/huytnc/LuuTru/HFDNet/fog_1.jpg"  # Specify the image path here
    net_input_size = (2048, 2048)  # Resize to fit input size of the model
    frame = Image.open(img_path).resize(net_input_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet50
    ])
    frame = transform(frame).unsqueeze(0)  # Add batch dimension

    frame = frame.to(device)
    
    
    # Run inference to capture feature maps
    with torch.no_grad():
        _ = model.model(frame)
    
    if feature_maps:
        for idx, fm in enumerate(feature_maps):
            fm = fm[0].cpu().detach().numpy()  
            print(f"Shape of fm[0] for layer {idx}: {fm.shape}")
    
            fig, ax = plt.subplots(figsize=(6, 6)) 
            ax.imshow(fm[0], cmap='jet')  
            ax.axis('off')  
            plt.title(f"Feature Map of First Filter from Layer {idx}")
            fig.savefig('/home/huytnc/LuuTru/HFDNet/feature_map.png')
            
            
            
            
            
            
    
    # model.load('yolov5mu.pt')
    model.load('/home/huytnc/LuuTru/HFDNet/yolov8s.pt') # loading pretrain weights
    
    result = model.train(data='/home/huytnc/LuuTru/HFDNet/dataset/uitdrone.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4, # 32
                close_mosaic=10, 
                workers=4,# 4
                # device='0',
                optimizer='SGD', # using SGD
                patience=0, # set 0 to close earlystop
                resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # half=False,
                # fraction=0.2,
                cos_lr = True,
                # project='runs/debug',
                project='runs/train/v8',
                name = 's2c',
                # mixup = 1.0,
                )
