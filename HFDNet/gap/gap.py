import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import json
import ot
import numpy as np
from torch import cdist

# 存储源域和目标域预测特征
# def feature():
#     # 输出特定层的 npy特征值
#     results_source, results_target = model.predict(source=(source_domain,target_domain), 
#                                                 imgsz=640,project='runs/detect',name='city_to_foggy',conf=0.6,
#                                                 # stream=True,
#                                                 agnostic_nms=True,
#                                                 visualize=True)
#     return results_source, results_target
# source_domain = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/val'
# target_domain = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format/images/val'
# weigth_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/sourcecity/weights/best.pt'
# model = YOLO(weigth_path) # select your model.pt path
# results_source, results_target = feature()
# print(results_source)
'''
sourcecity
city_to_foggy
/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/val
/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format/images/val

sourcesim10k
sim10k_to_city
/home/lenovo/data/liujiaji/DA-Datasets/Sim10k/images/val
/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format_car_class/images/val

sourcevoc
voc_to_clipart1k
/home/lenovo/data/liujiaji/DA-Datasets/VOC/train/VOCdevkit/VOC2012/yolov5_format/images/val
/home/lenovo/data/liujiaji/DA-Datasets/clipart/yolov5_format/images/val

sourceprivate
privatepower_to_publicpower
/home/lenovo/data/liujiaji/yolov8/privatepower/images/val
/home/lenovo/data/liujiaji/Datasets/publicpower/images/val
'''
## 1.获取源域和目标域特征值
weigth_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcecity/weights/best.pt'
model = YOLO(weigth_path) # select your model.pt path
# model.info(detailed=True)
# model.profile([640, 640])
model.fuse()

def source_feature():
    # 输出特定层的 npy特征值
    model.predict(source='/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/val', 
                imgsz=640,project='runs/detect/city_to_foggy',name='source',conf=0.6,
                # stream=True,
                agnostic_nms=True,
                visualize=True)
    
def target_feature():
    # 输出特定层的 npy特征值
    model.predict(source='/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format/images/val', 
                imgsz=640,project='runs/detect/city_to_foggy',name='target',conf=0.6,
                # stream=True,
                agnostic_nms=True,
                visualize=True)

# 2.计算差异
def gram_matrix(x):
    # x: (batch, channels, height, width)
    b, c, h, w = x.shape
    features = x.view(b, c, h*w)  # shape: (b, c, N)
    gram = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return gram

def compute_gram_loss(source_feat, target_feat):
    """
    Gram 矩阵差异（风格损失）
    """
    # print('gram',source_feat.shape,target_feat.shape)
    gram_s = gram_matrix(source_feat)
    gram_t = gram_matrix(target_feat)
    loss = F.mse_loss(gram_s, gram_t)
    return loss.detach()

def compute_l2_loss(source_feat, target_feat):
    """
    L2 损失（均方误差）
    """
    # print('l2',source_feat.shape,target_feat.shape)
    loss = F.mse_loss(source_feat, target_feat)
    return loss.detach()

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-6):
    """
    计算两个样本集合 x 和 y 之间的多尺度高斯核矩阵。
    
    Args:
        x (torch.Tensor): shape (n, feature_dim)
        y (torch.Tensor): shape (m, feature_dim)
        kernel_mul (float): 带宽倍数因子
        kernel_num (int): 高斯核个数
        fix_sigma (float): 固定带宽（可选）
        eps (float): 防止除零的小数值
    Returns:
        torch.Tensor: 高斯核矩阵，形状 (n, m)
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, feature_dim)
    L2_distance = torch.sum(diff ** 2, dim=2)  # (n, m)
    
    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.mean(L2_distance)
    # 防止带宽为0 ,导致后面 结果为nan
    bandwidth = max(bandwidth.item(), eps)
    # 调整带宽
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    # 构造多尺度带宽列表
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)

def compute_gaussianmmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    对每个通道分别计算 MMD，然后取所有通道的均值作为最终结果。
    
    输入特征的 shape 为 (batch, channels, height, width)。先对空间维度展平（或取均值），
    然后对每个通道分别计算 MMD。
    
    Returns:
        float: 每个通道 MMD 值的均值。
    """
    n = source_feat.size(0)
    m = target_feat.size(0)
    batch_size, n_channels, height, width = source_feat.shape
    
    # 将空间维度展平，保留通道信息： (batch, channels, height*width)
    feas_s = source_feat.view(n, n_channels, -1)
    feas_t = target_feat.view(m, n_channels, -1)
    
    mmd_val = 0.0
    for i in range(n_channels):
        # 对第 i 个通道：先对空间维度取均值，得到 (n, 1) 和 (m, 1)
        channel_s = feas_s[:, i, :].mean(dim=1, keepdim=True)  # (n, 1)
        channel_t = feas_t[:, i, :].mean(dim=1, keepdim=True)  # (m, 1)
        
        # # 计算高斯核矩阵
        K_xx = gaussian_kernel(channel_s, channel_s, kernel_mul, kernel_num, fix_sigma)
        K_yy = gaussian_kernel(channel_t, channel_t, kernel_mul, kernel_num, fix_sigma)
        K_xy = gaussian_kernel(channel_s, channel_t, kernel_mul, kernel_num, fix_sigma)

        # # 使用线性核，即内积。计算核矩阵：
        # K_xx = channel_s @ channel_s.t()  # (n, n)
        # K_yy = channel_t @ channel_t.t()  # (m, m)
        # K_xy = channel_s @ channel_t.t()  # (n, m)
        
        # 检查是否存在 nan
        if torch.isnan(K_xx).any() or torch.isnan(K_yy).any() or torch.isnan(K_xy).any():
            print(f"Warning: nan encountered in channel {i}")
            continue
        
        mmd_channel = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        mmd_val += mmd_channel
    # 取所有通道均值
    return (mmd_val / n_channels).item()

def compute_linearmmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    对每个通道分别计算 MMD，然后取所有通道的均值作为最终结果。
    
    输入特征的 shape 为 (batch, channels, height, width)。先对空间维度展平（或取均值），
    然后对每个通道分别计算 MMD。
    
    Returns:
        float: 每个通道 MMD 值的均值。
    """
    n = source_feat.size(0)
    m = target_feat.size(0)
    batch_size, n_channels, height, width = source_feat.shape
    
    # 将空间维度展平，保留通道信息： (batch, channels, height*width)
    feas_s = source_feat.view(n, n_channels, -1)
    feas_t = target_feat.view(m, n_channels, -1)
    
    mmd_val = 0.0
    for i in range(n_channels):
        # 对第 i 个通道：先对空间维度取均值，得到 (n, 1) 和 (m, 1)
        channel_s = feas_s[:, i, :].mean(dim=1, keepdim=True)  # (n, 1)
        channel_t = feas_t[:, i, :].mean(dim=1, keepdim=True)  # (m, 1)
        
        # # # 计算高斯核矩阵
        # K_xx = gaussian_kernel(channel_s, channel_s, kernel_mul, kernel_num, fix_sigma)
        # K_yy = gaussian_kernel(channel_t, channel_t, kernel_mul, kernel_num, fix_sigma)
        # K_xy = gaussian_kernel(channel_s, channel_t, kernel_mul, kernel_num, fix_sigma)

        # 使用线性核，即内积。计算核矩阵：
        K_xx = channel_s @ channel_s.t()  # (n, n)
        K_yy = channel_t @ channel_t.t()  # (m, m)
        K_xy = channel_s @ channel_t.t()  # (n, m)
        
        # 检查是否存在 nan
        if torch.isnan(K_xx).any() or torch.isnan(K_yy).any() or torch.isnan(K_xy).any():
            print(f"Warning: nan encountered in channel {i}")
            continue
        
        mmd_channel = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        mmd_val += mmd_channel
    # 取所有通道均值
    return (mmd_val / n_channels).item()


def compute_mmd_loss_allmean(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    方式1: 将所有通道展平为一维，得到 shape: (batch, channels*height*width)
         用于计算整体 MMD。  
    Returns:
        float: MMD 值（标量）。
    """
    # 获取样本数
    n = source_feat.size(0)
    m = target_feat.size(0)
    
    # 方式1：将所有通道展平
    source_flat = source_feat.view(n, -1)    # (n, n_channels*height*width)
    target_flat = target_feat.view(m, -1)      # (m, n_channels*height*width)

    # 计算高斯核矩阵
    kernels = gaussian_kernel(source_flat, target_flat, kernel_mul, kernel_num, fix_sigma)
    
    # 根据 MMD 定义计算均值差异：
    # MMD^2 = mean(K_xx) + mean(K_yy) - 2*mean(K_xy)
    # 其中 K_xx: 源域内部的核矩阵, K_yy: 目标域内部的核矩阵, K_xy: 跨域核矩阵。
    K_xx = gaussian_kernel(source_flat, source_flat, kernel_mul, kernel_num, fix_sigma)
    K_yy = gaussian_kernel(target_flat, target_flat, kernel_mul, kernel_num, fix_sigma)
    K_xy = kernels  # (n, m)
    
    mean_K_xx = K_xx.mean()
    mean_K_yy = K_yy.mean()
    mean_K_xy = K_xy.mean()
    
    mmd_loss = mean_K_xx + mean_K_yy - 2 * mean_K_xy
    return mmd_loss.item()

def compute_dss_loss(source_feat, target_feat):
    """
    Compute the average DSS (Domain Shift Score) difference between source and target domain features.
    Features are 4D tensors with shape (batch, channel, height, width).
    
    Args:
        source_feat (torch.Tensor): Source domain features (batch, channel, height, width)
        target_feat (torch.Tensor): Target domain features (batch, channel, height, width)
    
    Returns:
        float: Average DSS value across all channels
    """
    # Get dimensions
    batch_size, n_channels, height, width = source_feat.shape # [1,96,96,160]
    
    dss_val = 0
    
    # Flatten height and width dimensions for each channel
    feas_s_flat = source_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    feas_t_flat = target_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    for i in range(n_channels):
        # Extract features for current channel
        feas_s = feas_s_flat[:, i, :].unsqueeze(-1)  # (batch, height*width, 1)
        feas_t = feas_t_flat[:, i, :].unsqueeze(-1)  # (batch, height*width, 1)
        # Calculate dimensions
        ns = feas_s.shape[0]  # batch size 1
        d = feas_s.shape[1]   # height*width (feature dimension) 15360
        
        # Source covariance
        xm = torch.mean(feas_s, 1, keepdim=True) - feas_s
        xc = xm.transpose(1, 2) @ xm / ns  # (d, 1) @ (1, d)
        # Target covariance
        xmt = torch.mean(feas_t, 1, keepdim=True) - feas_t
        xct = xmt.transpose(1, 2) @ xmt / ns  # (d, 1) @ (1, d)
       
        # Frobenius norm between source and target covariances
        loss = torch.mul((xc - xct), (xc - xct))
        dss = torch.sum(loss) / (4 * d * d)
        dss_val += dss
    
    # Average across all channels
    return (dss_val / n_channels).item()

def compute_swd_loss(source_feat, target_feat):
    batch_size, n_channels, height, width = source_feat.shape # [1,96,96,160]
    wasserstein_distance = 0
    # Flatten height and width dimensions for each channel
    feas_s_flat = source_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    feas_t_flat = target_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    for i in range(n_channels):
        feas_s = feas_s_flat[:, i, :]  # (batch, height*width, 1)
        feas_t = feas_t_flat[:, i, :]  # (batch, height*width, 1)
        cost_matrix = cdist(feas_s, feas_t)**2
        cost_matrix = cost_matrix.detach().cpu().numpy()
        gamma = ot.emd(ot.unif(feas_s.shape[0]), ot.unif(feas_t.shape[0]), cost_matrix, numItermax=1e6)
        wasserstein_distance += np.sum(np.multiply(gamma, cost_matrix))
        
        del cost_matrix
 
    return (wasserstein_distance / n_channels).item()

'''
def process_features(source_dir, target_dir):
    """
    计算源域和目标域 **不同子文件夹下** 的相同 npy 文件的差异，并保存 JSON 结果。
    """
    source_subdirs = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    target_subdirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])

    results = {} # 存储单个 stage 的损失

    for src_sub, tgt_sub in zip(source_subdirs, target_subdirs):  # 逐个子文件夹匹配
        src_path = os.path.join(source_dir, src_sub)
        tgt_path = os.path.join(target_dir, tgt_sub)

        source_files = sorted(glob.glob(os.path.join(src_path, "*.npy")))
        target_files = sorted(glob.glob(os.path.join(tgt_path, "*.npy")))

        # 获取相同的文件名
        source_names = set(os.path.basename(f) for f in source_files)
        target_names = set(os.path.basename(f) for f in target_files)
        common_files = sorted(source_names & target_names)

        # 取最小文件数作为循环次数
        loop_count = min(len(source_files), len(target_files), len(common_files))

        if loop_count == 0:
            print(f"未找到 {src_sub} 和 {tgt_sub} 下的相同 npy 文件！")
            continue

        sub_results = {} # 计算当前子文件的所有损失
        gram_losses = [] # Gram
        l2_losses = [] # l2 
        gaussianmmd_losses = [] # gaussian-mmd
        linearmmd_losses = [] # gaussian-mmd
        dss_losses = [] # dss
        swd_losses = []

        for file_name in common_files[:loop_count]:  # 仅处理 loop_count 个文件
            # 解析 stage 信息 ，文件名格式为 "stage{stage}_{module_type}_features.npy"
            try:
                stage_part = file_name.split("stage")[1]
                stage_str = stage_part.split("_")[0]
            except Exception as e:
                print(f"解析 {file_name} 失败: {e}")
                continue

            src_file = os.path.join(src_path, file_name)
            tgt_file = os.path.join(tgt_path, file_name)

            # 读取 npy 文件，并转换为 torch.Tensor
            source_feat = torch.from_numpy(np.load(src_file))
            target_feat = torch.from_numpy(np.load(tgt_file))
            if source_feat.shape != target_feat.shape:
                # 调整 target_feat 的尺寸，使其匹配 source_feat
                target_feat = F.interpolate(target_feat, size=source_feat.shape[2:], mode="bilinear", align_corners=False)

            # 根据 stage 信息选择损失计算方式
            if stage_str in ['2', '4', '6']:
                loss = compute_gram_loss(source_feat, target_feat)
                gram_losses.append(loss)
                loss_type = 'Gram'

                gaussianmmd_loss = compute_gaussianmmd_loss(source_feat, target_feat)
                gaussianmmd_losses.append(gaussianmmd_loss)
                linearmmd_loss = compute_linearmmd_loss(source_feat, target_feat)
                linearmmd_losses.append(linearmmd_loss)

                dss_loss = compute_dss_loss(source_feat,target_feat)
                dss_losses.append(dss_loss)

                swd_loss = compute_swd_loss(source_feat,target_feat)
                swd_losses.append(swd_loss)

            elif stage_str in ['8', '9']:
                loss = compute_l2_loss(source_feat, target_feat)
                l2_losses.append(loss)
                loss_type = 'L2'

                gaussianmmd_loss = compute_gaussianmmd_loss(source_feat, target_feat)
                gaussianmmd_losses.append(gaussianmmd_loss)
                linearmmd_loss = compute_linearmmd_loss(source_feat, target_feat)
                linearmmd_losses.append(linearmmd_loss)

                dss_loss = compute_dss_loss(source_feat,target_feat)
                dss_losses.append(dss_loss)

                swd_loss = compute_swd_loss(source_feat,target_feat)
                swd_losses.append(swd_loss)
           
            else:
                loss = None
                loss_type = 'Unknown'

            key = f"{src_sub}->{tgt_sub}/{file_name}"
            sub_results[key] = {'loss_type': loss_type, 'loss': loss}
            print(f"{key}: {loss_type} difference = {loss}")

        # mean_gram_loss = sum(gram_losses) / len(gram_losses)
        # mean_l2_loss = sum(l2_losses) / len(l2_losses)
        # mean_gaussianmmd_loss = sum(gaussianmmd_losses) / len(gaussianmmd_losses)
        # mean_linearmmd_loss = sum(linearmmd_losses) / len(linearmmd_losses)
        # mean_dss_loss = sum(dss_losses) / len(dss_losses)
        # mean_swd_loss = sum(swd_losses) / len(swd_losses)

        mean_gram_loss = sum(gram_losses) * 100
        mean_l2_loss = sum(l2_losses) 
        mean_gaussianmmd_loss = sum(gaussianmmd_losses) 
        mean_linearmmd_loss = sum(linearmmd_losses)
        mean_dss_loss = sum(dss_losses) 
        mean_swd_loss = sum(swd_losses) 

        # 存储子文件夹的均值
        results[src_sub] = {
            "individual_results": sub_results,
            "mean_gram_loss": mean_gram_loss.item() if isinstance(mean_gram_loss, torch.Tensor) else mean_gram_loss,
            "mean_l2_loss": mean_l2_loss.item() if isinstance(mean_l2_loss, torch.Tensor) else mean_l2_loss,
            "mean_gaussianmmd_loss": mean_gaussianmmd_loss.item() if isinstance(mean_gaussianmmd_loss, torch.Tensor) else mean_gaussianmmd_loss,
            "mean_linearmmd_loss": mean_linearmmd_loss.item() if isinstance(mean_linearmmd_loss, torch.Tensor) else mean_linearmmd_loss,
            "mean_dss_loss": mean_dss_loss.item() if isinstance(mean_dss_loss, torch.Tensor) else mean_dss_loss,
            "mean_swd_loss": mean_swd_loss.item() if isinstance(mean_swd_loss, torch.Tensor) else mean_swd_loss
        }

    return results
'''

def process_features(source_dir, target_dir):
    """
    计算源域和目标域 **不同子文件夹下** 的相同 npy 文件的差异，并保存 JSON 结果。
    """
    source_subdirs = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    target_subdirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])

    results = {}  # 存储单个 stage 的损失

    for src_sub, tgt_sub in zip(source_subdirs, target_subdirs):  # 逐个子文件夹匹配
        src_path = os.path.join(source_dir, src_sub)
        tgt_path = os.path.join(target_dir, tgt_sub)

        source_files = sorted(glob.glob(os.path.join(src_path, "*.npy")))
        target_files = sorted(glob.glob(os.path.join(tgt_path, "*.npy")))

        # 获取相同的文件名
        source_names = set(os.path.basename(f) for f in source_files)
        target_names = set(os.path.basename(f) for f in target_files)
        common_files = sorted(source_names & target_names)

        # 取最小文件数作为循环次数
        loop_count = min(len(source_files), len(target_files), len(common_files))

        if loop_count == 0:
            print(f"未找到 {src_sub} 和 {tgt_sub} 下的相同 npy 文件！")
            continue

        sub_results = {}  # 计算当前子文件的所有损失
        gram_losses = []  # Gram
        l2_losses = []  # l2
        gaussianmmd_losses = []  # gaussian-mmd
        linearmmd_losses = []  # gaussian-mmd
        dss_losses = []  # dss
        swd_losses = []

        # 新增各层级归一化特征差异列表
        normalized_diffs_shallow = []
        normalized_diffs_middle = []
        normalized_diffs_deep = []

        for file_name in common_files[:loop_count]:  # 仅处理 loop_count 个文件
            # 解析 stage 信息 ，文件名格式为 "stage{stage}_{module_type}_features.npy"
            try:
                stage_part = file_name.split("stage")[1]
                stage_str = stage_part.split("_")[0]
            except Exception as e:
                print(f"解析 {file_name} 失败: {e}")
                continue

            src_file = os.path.join(src_path, file_name)
            tgt_file = os.path.join(tgt_path, file_name)

            # 读取 npy 文件，并转换为 torch.Tensor
            source_feat = torch.from_numpy(np.load(src_file))
            target_feat = torch.from_numpy(np.load(tgt_file))
            if source_feat.shape != target_feat.shape:
                # 调整 target_feat 的尺寸，使其匹配 source_feat
                target_feat = F.interpolate(target_feat, size=source_feat.shape[2:], mode="bilinear", align_corners=False)
            
            # 特征归一化（按通道）
            source_norm = F.normalize(source_feat, p=2, dim=1)
            target_norm = F.normalize(target_feat, p=2, dim=1)
            # 计算归一化特征之间的 L2 距离（可换成余弦距离等）
            norm_diff = F.mse_loss(source_norm, target_norm, reduction='mean').item()# 

            # 分类统计层级归一化差异
            if stage_str in ['2', '4']:
                normalized_diffs_shallow.append(norm_diff)
            elif stage_str in ['6']:
                normalized_diffs_middle.append(norm_diff)
            elif stage_str in ['8', '9']:
                normalized_diffs_deep.append(norm_diff)
            
                
            # 根据 stage 信息选择损失计算方式
            if stage_str in ['2', '4']:
                loss = compute_dss_loss(source_feat, target_feat)
                dss_losses.append(loss)
                loss_type = 'dss'

                gram_loss = compute_gram_loss(source_feat, target_feat)
                gram_losses.append(gram_loss)
                # loss_type = 'Gram'
                
                swd_loss = compute_swd_loss(source_feat, target_feat)
                swd_losses.append(swd_loss)
                # loss_type = 'swd'

            elif stage_str in ['6']:
                loss = compute_gaussianmmd_loss(source_feat, target_feat)
                gaussianmmd_losses.append(loss)
                loss_type = 'gmmd'
                
                linearmmd_loss = compute_linearmmd_loss(source_feat, target_feat)
                linearmmd_losses.append(linearmmd_loss)
                # loss_type = 'lmmd'

            elif stage_str in ['8', '9']:
                loss = compute_l2_loss(source_feat, target_feat)
                l2_losses.append(loss)
                loss_type = 'L2'

            else:
                loss = None
                loss_type = 'Unknown'

            key = f"{src_sub}->{tgt_sub}/{file_name}"
            sub_results[key] = {'loss_type': loss_type, 
                                'loss': loss.item() if isinstance(loss, torch.Tensor) else loss}
                                # 'norm_diff': norm_diff}
            print(f"{key}: {loss_type} loss = {loss}")

        # 计算均值
        mean_gram_loss = sum(gram_losses) / len(gram_losses)
        mean_l2_loss = sum(l2_losses) / len(l2_losses)
        mean_gaussianmmd_loss = sum(gaussianmmd_losses) / len(gaussianmmd_losses)
        mean_linearmmd_loss = sum(linearmmd_losses) / len(linearmmd_losses)
        mean_dss_loss = sum(dss_losses) / len(dss_losses)
        mean_swd_loss = sum(swd_losses)/  len(swd_losses)

        # 确保所有值为 float 类型
        results[src_sub] = {
            "individual_results": sub_results,
            "mean_gram_loss": mean_gram_loss.item() if isinstance(mean_gram_loss, torch.Tensor) else mean_gram_loss,
            "mean_l2_loss": mean_l2_loss.item() if isinstance(mean_l2_loss, torch.Tensor) else mean_l2_loss,
            "mean_gaussianmmd_loss": mean_gaussianmmd_loss.item() if isinstance(mean_gaussianmmd_loss, torch.Tensor) else mean_gaussianmmd_loss,
            "mean_linearmmd_loss": mean_linearmmd_loss.item() if isinstance(mean_linearmmd_loss, torch.Tensor) else mean_linearmmd_loss,
            "mean_dss_loss": mean_dss_loss.item() if isinstance(mean_dss_loss, torch.Tensor) else mean_dss_loss,
            "mean_swd_loss": mean_swd_loss.item() if isinstance(mean_swd_loss, torch.Tensor) else mean_swd_loss
        }


    return results, normalized_diffs_shallow,normalized_diffs_middle,normalized_diffs_deep
    # return normalized_diffs_shallow,normalized_diffs_middle,normalized_diffs_deep

import numpy as np
import json

# 统计函数并保存为字典
def summary(name, values):
    arr = np.array(values)
    stats = {
        "mean": float(arr.mean()),
        "std_dev": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max())
    }
    # print(f"{name}:")
    # print(f"  Mean      = {stats['mean']:.6f}")
    # print(f"  Std Dev   = {stats['std_dev']:.6f}")
    # print(f"  Min~Max   = [{stats['min']:.6f}, {stats['max']:.6f}]\n")
    return stats



# 示例调用
if __name__ == "__main__":
    ### 1.先计算特征值
    # source_feature()
    # target_feature()

    ###  2.计算对应层级特征分布差异
    # city_to_foggy，sim10k_to_city 、voc_to_clipart1k、  pupower_to_prpower
    source_directory = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/detect/pupower_to_prpower/source' 
    target_directory = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/detect/pupower_to_prpower/target'
    results,_,_,_ = process_features(source_directory, target_directory)
    # 保存结果
    gapoutput_path = "./gap/pu2pr_gap.json"
    with open(gapoutput_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"特征差异值 已保存结果到 {gapoutput_path}")

    _,normalized_diffs_shallow, normalized_diffs_middle, normalized_diffs_deep = process_features(source_directory, target_directory)
    results = {
        "shallow_stage2_4": summary("Shallow (stage2/4)", normalized_diffs_shallow),
        "middle_stage6": summary("Middle  (stage6)", normalized_diffs_middle),
        "deep_stage8_9": summary("Deep    (stage8/9)", normalized_diffs_deep)
    }
    # 保存为 JSON 文件
    feoutput_path = "./gap/feature/pu2pr.json"
    with open(feoutput_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"特征相对量 已保存结果到 {feoutput_path}")

