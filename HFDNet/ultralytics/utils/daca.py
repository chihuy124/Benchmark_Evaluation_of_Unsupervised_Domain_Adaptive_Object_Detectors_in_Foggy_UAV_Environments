import copy
import torch
import numpy as np
import ot
from ultralytics.utils.ops import  xywh2xyxy,xyxy2xywh
import torch.nn.functional as F



def cross_set_cutmix_pseudo(source_img, target_img, source_labels, target_labels, alpha=1.0, conf_threshold=0.5):
    """
    Cross-set CutMix for object detection tasks with pseudo-labels for the target domain.
    
    Args:
        source_img (torch.Tensor): Source domain images [B, C, H, W].
        target_img (torch.Tensor): Target domain images [B, C, H, W].
        source_labels (torch.Tensor): Source domain labels [N, 7] (batch_id, class_id, x, y, w, h, conf).
        target_labels (torch.Tensor): Target domain pseudo-labels [M, 7] (batch_id, class_id, x, y, w, h, conf).
        alpha (float): Beta distribution parameter for CutMix.
        conf_threshold (float): Confidence threshold for filtering pseudo-labels.
    
    Returns:
        mixed_img (torch.Tensor): Mixed image [B, C, H, W].
        mixed_labels (torch.Tensor): Mixed labels [K, 7] (batch_id, class_id, x, y, w, h, conf).
    """
    B, C, H, W = source_img.shape
    mixed_img = target_img.clone()
    
    # Filter out low-confidence pseudo-labels from the target domain
    target_labels = target_labels[target_labels[:, 6] >= conf_threshold]  # Filter by confidence
    
    # Combine source and target labels
    all_labels = torch.cat([source_labels, target_labels], dim=0) #[X,7]
    
    # Initialize mixed_labels with the same size as all_labels
    mixed_labels = torch.zeros_like(all_labels)  # Initialize with zeros #[X,7]
    mixed_labels[:,1] = torch.full_like(all_labels[:,1], -1)
    mixed_labels[:,2:6] = torch.zeros_like(all_labels[:,2:6])   # 初始化为0

    # Generate random bounding box for CutMix
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    
    # Copy the region from source image to target image
    mixed_img[:, :, y1:y2, x1:x2] = source_img[:, :, y1:y2, x1:x2]
    
    # Adjust bounding boxes and labels for the mixed region
    # valid_label_count = 0  # Counter for valid labels
    for i, label in enumerate(all_labels):
        batch_id, class_id, x, y, w, h, conf = label
        if batch_id >= B:  # Skip labels that do not belong to the current batch
            continue
        # Convert to absolute coordinates
        bbox_abs = torch.tensor([x * W, y * H, (x + w) * W, (y + h) * H])
        # Check if bbox intersects with CutMix region
        if (bbox_abs[0] < x2 and bbox_abs[2] > x1 and bbox_abs[1] < y2 and bbox_abs[3] > y1):
            # Clip bbox to CutMix region
            bbox_abs[0] = max(bbox_abs[0], x1)
            bbox_abs[1] = max(bbox_abs[1], y1)
            bbox_abs[2] = min(bbox_abs[2], x2)
            bbox_abs[3] = min(bbox_abs[3], y2)
            # Convert back to normalized coordinates
            bbox_norm = torch.tensor([
                bbox_abs[0] / W,  # x
                bbox_abs[1] / H,  # y
                (bbox_abs[2] - bbox_abs[0]) / W,  # w
                (bbox_abs[3] - bbox_abs[1]) / H,  # h
            ])
            # Fill the mixed_labels tensor
            mixed_labels = torch.tensor([
                batch_id, class_id, bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3], conf
            ])
            # valid_label_count += 1
    # Truncate mixed_labels to only include valid labels
    # mixed_labels = mixed_labels[:valid_label_count] 
    return mixed_img, mixed_labels

def cross_set_cutmix(source_img, target_img, source_cls, source_bbox, alpha=1.0):
    """
    Cross-set CutMix for object detection tasks.
    
    Args:
        source_img (torch.Tensor): Source domain images [B, C, H, W].
        target_img (torch.Tensor): Target domain images [B, C, H, W].
        source_cls (torch.Tensor): Source domain class labels [N, 1].
        source_bbox (torch.Tensor): Source domain bounding boxes [N, 4] (normalized xyxy format).
        alpha (float): Beta distribution parameter for CutMix.
    
    Returns:
        mixed_img (torch.Tensor): Mixed image [B, C, H, W].
        mixed_cls (torch.Tensor): Mixed class labels [N, 1].
        mixed_bbox (torch.Tensor): Mixed bounding boxes [N, 4].
    """
    B, C, H, W = source_img.shape
    mixed_img = target_img.clone()
    mixed_cls = torch.full_like(source_cls, -1)  # 初始化为 -1，表示无效标签
    mixed_bbox = torch.zeros_like(source_bbox)   # 初始化为0
    
    # Generate random bounding box for CutMix
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    
    # Copy the region from source image to target image
    mixed_img[:, :, y1:y2, x1:x2] = source_img[:, :, y1:y2, x1:x2]

    # Adjust bounding boxes and labels for the mixed region
    for idx, (bbox, cls) in enumerate(zip(source_bbox, source_cls)):
        bbox_abs = bbox.clone()
        # Convert to absolute coordinates
        bbox_abs[0] *= W
        bbox_abs[1] *= H
        bbox_abs[2] *= W
        bbox_abs[3] *= H
        
        # Check if bbox intersects with CutMix region
        if (bbox_abs[0] < x2 and bbox_abs[2] > x1 and bbox_abs[1] < y2 and bbox_abs[3] > y1):
            # Clip bbox to CutMix region
            bbox_abs[0] = max(bbox_abs[0], x1)
            bbox_abs[1] = max(bbox_abs[1], y1)
            bbox_abs[2] = min(bbox_abs[2], x2)
            bbox_abs[3] = min(bbox_abs[3], y2)
            
            # Normalize back
            bbox_abs[0] /= W
            bbox_abs[1] /= H
            bbox_abs[2] /= W
            bbox_abs[3] /= H
            
            mixed_bbox[idx] = bbox_abs
            mixed_cls[idx] = cls

    return mixed_img, mixed_cls, mixed_bbox


def adjust_alpha(epoch, max_epoch, initial_alpha=1.0, final_alpha=0.0):
    """
    Dynamically adjust alpha based on training progress.
    
    Args:
        epoch (int): Current epoch.
        max_epoch (int): Maximum number of epochs.
        initial_alpha (float): Initial value of alpha.
        final_alpha (float): Final value of alpha.
    
    Returns:
        alpha (float): Adjusted alpha value.
    """
    alpha = initial_alpha - (initial_alpha - final_alpha) * (epoch / max_epoch)
    return alpha




# def cutmix_detection(batch_s, batch_t, alpha):
#     # 解包源域和目标域的数据
#     source_img = batch_s['img']  # 源域图像 [4, 3, 640, 640]
#     source_cls = batch_s['cls']  # 源域分类标签 [203, 1]
#     source_bbox = batch_s['bboxes']  # 源域边界框坐标 [203, 4]
#     source_batchidx = batch_s['batch_idx'] # 源域的类别/边界框 索引 [203]

#     target_img = batch_t['img']  # 目标域图像 [4, 3, 640, 640]
#     target_batchidx = batch_t['batch_idx'] # 目标域的类别/边界框 索引 203]

#     # 生成相同的随机排列索引
#     indices = torch.randperm(source_img.size(0)) # tensor [2,1,0,3]
#     shuffled_source_img = source_img[indices] # [4, 3, 640, 640]
#     shuffled_target_img = target_img[indices] # [4, 3, 640, 640]

#     # 生成相同的混合比例 lam
#     lam = np.random.beta(alpha, alpha) # float值 0.44999

#     # 生成相同的裁剪区域
#     image_h, image_w = source_img.shape[2:] #[640,640]
#     cx = np.random.uniform(0, image_w) # 随机生成的  12.93977436
#     cy = np.random.uniform(0, image_h) # 随机生成的 532.8767
#     w = image_w * np.sqrt(1 - lam) # 475.01XXXXXX
#     h = image_h * np.sqrt(1 - lam) # 475.01XXXXXX
#     x0, x1 = int(np.round(max(cx - w / 2, 0))), int(np.round(min(cx + w / 2, image_w)))  # 0 ,250
#     y0, y1 = int(np.round(max(cy - h / 2, 0))), int(np.round(min(cy + h / 2, image_h))) # # 295 640 

#     # # 对源域和目标域进行相同的混合 进行CutMix
#     # source_img[:, :, y0:y1, x0:x1] = shuffled_source_img[:, :, y0:y1, x0:x1] # [4, 3, 640, 640]
#     # target_img[:, :, y0:y1, x0:x1] = shuffled_target_img[:, :, y0:y1, x0:x1] # [4, 3, 640, 640]
    
#     # 创建掩码，用于混合区域
#     mask = torch.zeros_like(source_img, dtype=torch.bool)
#     mask[:, :, y0:y1, x0:x1] = True

#     # 对源域和目标域进行相同的混合（避免原地操作）
#     mixed_source_img = torch.where(mask, shuffled_source_img, source_img)
#     mixed_target_img = torch.where(mask, shuffled_target_img, target_img)

#     # 遍历每张图像，调整其边界框
#     mixed_source_bbox = source_bbox.clone()  # 克隆以避免修改原始数据 [203,4]
#     for idx in torch.unique(source_batchidx):  # 遍历每张图像
#         # 获取当前图像的边界框索引
#         mask = source_batchidx == idx
#         if not mask.any():
#             continue  # 如果没有边界框，跳过
#         # 调整当前图像的边界框坐标
#         mixed_source_bbox[mask, 0] = torch.clamp(mixed_source_bbox[mask, 0], x0, x1)  # xmin
#         mixed_source_bbox[mask, 1] = torch.clamp(mixed_source_bbox[mask, 1], y0, y1)  # ymin
#         mixed_source_bbox[mask, 2] = torch.clamp(mixed_source_bbox[mask, 2], x0, x1)  # xmax
#         mixed_source_bbox[mask, 3] = torch.clamp(mixed_source_bbox[mask, 3], y0, y1)  # ymax

#     # # 混合后的源域标签
#     # # 由于 source_cls 和 source_bbox 是扁平化的，直接复制并添加 lam
#     # mixed_source_cls = torch.cat([source_cls, source_cls, torch.full_like(source_cls, lam)], dim=1)  # [203, 3]   3 表示原始标签、打乱标签和 lam
#     # mixed_source_bbox = torch.cat([source_bbox, source_bbox, torch.full_like(source_bbox, lam)], dim=1)  # [203, 9] # 9 表示原始边界框、打乱边界框和 lam
    

#     # 保持其他键值不变
#     mixed_batch_s = batch_s.copy() # mixed_batch_s['batch_idx'].shape [203]
#     mixed_batch_s['img'] = mixed_source_img # [4,3,640,640]
#     # mixed_batch_s['cls'] = mixed_source_cls # [203,3]
#     mixed_batch_s['bboxes'] = mixed_source_bbox # [203,12]

#     mixed_batch_t = batch_t.copy()
#     mixed_batch_t['img'] = mixed_target_img

#     return mixed_batch_s, mixed_batch_t

def gram_matrix(x):
    b, c, h, w = x.shape
    features = x.view(b, c, h*w)  # shape: (b, c, N)
    gram = torch.mm(features, features.transpose(1, 2)) / (c * h * w)
    return gram

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

def compute_mmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    对每个通道分别计算 MMD，然后取所有通道的均值作为最终结果。
    
    输入特征的 shape 为 (batch, channels, height, width)。先对空间维度展平（或取均值），
    然后对每个通道分别计算 MMD。
    
    Returns:
        float: 每个通道 MMD 值的均值。
    """
    # 确保输入张量在 GPU 上
    if not source_feat.is_cuda or not target_feat.is_cuda:
        raise ValueError("Input tensors must be on GPU (cuda).")
    n = source_feat.size(0)
    m = target_feat.size(0)
    batch_size, n_channels, height, width = source_feat.shape
    # 将空间维度展平，保留通道信息： (batch, channels, height*width)
    feas_s = source_feat.view(n, n_channels, -1)
    feas_t = target_feat.view(m, n_channels, -1)
    mmd_val = torch.tensor(0.0, device=source_feat.device)  # 初始化 mmd_val 在 GPU 上
    
    for i in range(n_channels):
        # 对第 i 个通道：先对空间维度取均值，得到 (n, 1) 和 (m, 1)
        channel_s = feas_s[:, i, :].mean(dim=1, keepdim=True)  # (n, 1)
        channel_t = feas_t[:, i, :].mean(dim=1, keepdim=True)  # (m, 1)
        # # # 计算高斯核矩阵
        K_xx = gaussian_kernel(channel_s, channel_s, kernel_mul, kernel_num, fix_sigma)
        K_yy = gaussian_kernel(channel_t, channel_t, kernel_mul, kernel_num, fix_sigma)
        K_xy = gaussian_kernel(channel_s, channel_t, kernel_mul, kernel_num, fix_sigma)

        # 使用线性核，即内积。计算核矩阵：
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
    return mmd_val / n_channels

def compute_swd_loss(source_feat, target_feat):
    batch_size, n_channels, height, width = source_feat.shape  # [1, 96, 96, 160]
    wasserstein_distance = 0
    # Flatten height and width dimensions for each channel 每个通道的特征图展平为一维向量，方便后续计算。
    feas_s_flat = source_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    feas_t_flat = target_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    
    for i in range(n_channels):
        feas_s = feas_s_flat[:, i, :]  # (batch, height*width) 
        feas_t = feas_t_flat[:, i, :]  # (batch, height*width)
        # Compute cost matrix on GPU，使用 cdist 函数计算 feas_s 和 feas_t 之间的欧氏距离平方，得到代价矩阵 cost_matrix：
        cost_matrix = torch.cdist(feas_s, feas_t, p=2)**2  # (batch, height*width, height*width)
        # # Move cost matrix to CPU for OT computation
        # cost_matrix_cpu = cost_matrix.detach().cpu().numpy()
        # # Compute optimal transport
        # '''
        # ot.unif(n) 生成一个均匀分布的概率向量，长度为 n，每个元素的值为 1/n。
        # 这里的作用是为 feas_s 和 feas_t 分别生成均匀分布的概率质量函数（PMF），表示每个样本的权重。
        # a=[ 1/n,..,1/n ],n=feas_s.shape[0]

        # ot.emd(a, b, cost_matrix, numItermax=1e6):是 POT 库中用于计算 Earth Mover's Distance (EMD) 的函数。
        # a: 第一个分布的权重向量（均匀分布）。
        # b: 第二个分布的权重向量（均匀分布）。
        # cost_matrix: 代价矩阵，表示从 a 的每个点到 b 的每个点的传输成本。
        # numItermax: 最大迭代次数，用于控制算法的收敛性。
        # 返回值 gamma 是一个最优传输矩阵，表示从分布 a 到分布 b 的最优传输计划。
        # '''
        # gamma = ot.emd(ot.unif(feas_s.shape[0]), ot.unif(feas_t.shape[0]), cost_matrix_cpu, numItermax=1e6)
        # # Compute Wasserstein distance
        # # np.multiply(gamma, cost_matrix_cpu): 计算传输矩阵 gamma 和代价矩阵 cost_matrix_cpu 的逐元素乘积。
        # # torch.sum(torch.tensor(...)): 将结果转换为 PyTorch 张量并求和，得到当前通道的 Wasserstein 距离。
        # wasserstein_distance += torch.sum(torch.tensor(np.multiply(gamma, cost_matrix_cpu), device=source_feat.device))
        # del cost_matrix_cpu

        gamma = ot.sinkhorn(ot.unif(feas_s.shape[0]), ot.unif(feas_t.shape[0]), cost_matrix.cpu().numpy(), reg=1e-1)
        wasserstein_distance += torch.sum(torch.tensor(np.multiply(gamma, cost_matrix.cpu().numpy()), device=source_feat.device))
    return wasserstein_distance / n_channels

def compute_dss_loss(source_feat, target_feat):
    """
    Compute the average DSS (Distance ofsecond-order statistics ) difference between source and target domain features.
    Features are 4D tensors with shape (batch, channel, height, width).
    
    Args:
        source_feat (torch.Tensor): Source domain features (batch, channel, height, width)
        target_feat (torch.Tensor): Target domain features (batch, channel, height, width)
    
    Returns:
        float: Average DSS value across all channels
    """
    # Get dimensions
    batch_size, n_channels, height, width = source_feat.shape # [1,96,96,160]
    dss_val = torch.tensor(0.0, device = source_feat.device)  # 在相同设备上初始化 dss_val
    # Flatten height and width dimensions for each channel
    feas_s_flat = source_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    feas_t_flat = target_feat.view(batch_size, n_channels, -1)  # (batch, channel, height*width)
    
    for i in range(n_channels):
        # Extract features for current channel
        feas_s = feas_s_flat[:, i, :].unsqueeze(-1)  # (batch, height*width, 1)
        feas_t = feas_t_flat[:, i, :].unsqueeze(-1)  # (batch, height*width, 1)
        # Calculate dimensions
        ns = feas_s.shape[0]  # batch size 1
        nt = feas_t.shape[0]
        d = feas_s.shape[1]   # height*width (feature dimension) 15360
        
        # Source covariance
        xm = torch.mean(feas_s, 1, keepdim=True) - feas_s
        xc = xm.transpose(1, 2) @ xm / ns  # (d, 1) @ (1, d)
        # Target covariance
        xmt = torch.mean(feas_t, 1, keepdim=True) - feas_t
        xct = xmt.transpose(1, 2) @ xmt / nt  # (d, 1) @ (1, d)
       
        # Frobenius norm between source and target covariances
        loss = torch.mul((xc - xct), (xc - xct)) 
        dss = torch.sum(loss) / (4 * d * d)
        dss_val += dss
    
    # Average across all channels
    return dss_val / n_channels


def get_features(x, module_type, stage):
    """
    获取特定层的输出特征。
    
    参数:
        x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]。
        module_type (str): 模块类型（如 "Detect", "Pose", "Segment"）。
        stage (int): 当前层数。
    
    返回:
        out_feas (torch.Tensor): 保存的特征图列表（如果层数属于 [2, 4, 6, 8, 9]）。
    """
    # 如果模块类型是 "Detect", "Pose", "Segment"，直接返回
    for m in ["Detect", "Pose", "Segment"]:
        if m in module_type:
            return None

    # 获取输入特征图的形状
    batch, channels, height, width = x.shape  # batch, channels, height, width

    # 初始化特征图列表
    out_feas_list = []
    # print(f"daca.py ⚠️ Computing features at stage {stage}") 
    
    # 如果层数属于 [2, 4, 6, 8, 9]，保存特征图
    if stage in [0]:
        print(f"Saving features at stage {stage}")  # 打印当前层数
        out_feas_list.append(x)

    # 将特征图列表转换为张量
    if out_feas_list:  # 如果列表不为空
        out_feas = torch.stack(out_feas_list)  # 将列表中的张量堆叠为一个张量
    else:
        out_feas = None  # 如果没有保存特征图，返回 None

    return out_feas

def get_best_region(out, imgs_t):
    # imgs_t.shape：torch.Size([4, 3, 640, 640])
    region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)] # initialize in case no bboxes are detected
    best_side = 'topleft'  # initialize in case no bboxes are detected 左上角
    if out.shape[0] > 0:
        bboxes_target = copy.deepcopy(out) # [batch_id, class_id, x, y, w, h, conf] (16,7)
        # 筛选左上角区域 (topleft)： numpy < float 320.0
        # bboxes_target[:, 2] < imgs_t.shape[2]/2 → 目标中心点 cx 在左半部分
        # bboxes_target[:, 3] < imgs_t.shape[3]/2 → 目标中心点 cy 在上半部分
        bboxes_target_topleft = bboxes_target[bboxes_target[:, 2] < imgs_t.shape[2]/2, :]
        bboxes_target_topleft = bboxes_target_topleft[bboxes_target_topleft[:, 3] < imgs_t.shape[3]/2, :]
        # 筛选左下角 (bottomleft)
        bboxes_target_bottomleft = bboxes_target[bboxes_target[:, 2] < imgs_t.shape[2]/2, :]
        bboxes_target_bottomleft = bboxes_target_bottomleft[bboxes_target_bottomleft[:, 3] > imgs_t.shape[3]/2, :]
        # 筛选右下角 (bottomright)
        bboxes_target_bottomright = bboxes_target[bboxes_target[:, 2] > imgs_t.shape[2]/2, :]
        bboxes_target_bottomright = bboxes_target_bottomright[bboxes_target_bottomright[:, 3] > imgs_t.shape[3]/2, :]
        # 筛选右上角 (topright)
        bboxes_target_topright = bboxes_target[bboxes_target[:, 2] > imgs_t.shape[2]/2, :]
        bboxes_target_topright = bboxes_target_topright[bboxes_target_topright[:, 3] < imgs_t.shape[3]/2, :]
        # 计算每个区域目标框的平均置信度 conf
        conf_topleft = np.mean(bboxes_target_topleft[:, -1]) if len(bboxes_target_topleft)>0 else 0
        conf_bottomleft = np.mean(bboxes_target_bottomleft[:, -1]) if len(bboxes_target_bottomleft)>0 else 0
        conf_bottomright = np.mean(bboxes_target_bottomright[:, -1]) if len(bboxes_target_bottomright)>0 else 0
        conf_topright = np.mean(bboxes_target_topright[:, -1]) if len(bboxes_target_topright)>0 else 0

        if bboxes_target.shape[0]>0:
            # 存储四个方向的名称
            side = ['topleft', 'bottomleft', 'bottomright', 'topright'] 
            # 存储四个方向的目标框数据 
            region_bboxes = [bboxes_target_topleft, bboxes_target_bottomleft, bboxes_target_bottomright, bboxes_target_topright]
            # 存储四个区域的置信度
            conf = [conf_topleft, conf_bottomleft, conf_bottomright, conf_topright]
            # 找到置信度最高的区域索引
            id_best = conf.index(max(conf))
            # 更新最佳区域名称
            best_side = side[id_best]
            # 获取该区域的目标框
            best_bboxes = region_bboxes[id_best]
        else:
            best_bboxes = []

        if best_bboxes.shape[0]>0:
            out = copy.deepcopy(best_bboxes)
            out = torch.from_numpy(out) # 转换为tensor

        # 对每个区域进行坐标裁剪，防止 bbox 超出选定区域
        if best_side == 'topleft' and best_bboxes.shape[0] > 0: 
            # 遍历 out 里的 bbox，如果超出该区域范围，则进行修正（调整 cx, cy, w, h）
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2 > imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2 > imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)
            
        elif best_side == 'bottomleft' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2 < imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2 > imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'bottomright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, int(imgs_t.shape[2]/2):]                       
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2 < imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2 < imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'topright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), int(imgs_t.shape[2]/2):]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2 > imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2 < imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

    else:
        out = torch.empty([0,7])  # 如果没有检测到目标，返回空张量
    
    # 最佳区域的图像部分,
    # 最佳区域中的目标框信息（格式 [batch_id, class, cx, cy, w, h, conf]）
    # 最佳区域的位置（topleft，bottomleft，bottomright，topright）
    return region_t, out, best_side 

def transform_img_bboxes(out, best_side, region_t, transform_):
    '''
    out 原始的检测框(bounding boxes) 形状为 (N, 7)，其中 N 是目标的数量。
    best_side:选定的最佳区域（'topleft', 'bottomleft', 'bottomright', 'topright'）。
    region_t:从原始图像裁剪出的最佳区域图像。 region_t.shape = torch.Size([4, 3, 320, 320])
    transform_:一个用于数据增强的变换函数。
    '''
    out_ = copy.deepcopy(out) # 作用是创建 out 的深拷贝，确保不会修改原始数据
    
    # fit the coordinates into the region-level reference instead of whole image
    # 目标框 out_ 是基于整张图片的坐标，而 region_t 只是原图的一部分，因此需要调整坐标：
    
    # bottomleft 区域（左下）：减去 region_t.shape[3] 以适应 region_t 的相对坐标。
    if best_side == 'bottomleft':
        out_[:, 3] -= region_t.shape[3]
    # bottomright 区域（右下）：既要调整 x 坐标（宽度），又要调整 y 坐标（高度）。
    if best_side == 'bottomright':
        out_[:, 2] -= region_t.shape[2]
        out_[:, 3] -= region_t.shape[3]
    # topright 区域（右上）：只需要调整 x 坐标。
    if best_side == 'topright':
        out_[:, 2] -= region_t.shape[2]  

    # convert to [0, 1]
    # out_[:, 2] 代表目标中心 x 坐标，out_[:, 3] 代表目标中心 y 坐标。
    # out_[:, 4] 代表目标宽度，out_[:, 5] 代表目标高度。
    # 避免目标框超出 region_t
    for jj in range(out_.shape[0]):
        if out_[jj, 2] - out_[jj, 4]/2 < 0:  
            out_[jj, 4] = 2*out_[jj, 2]
        if out_[jj, 2] + out_[jj, 4]/2 > region_t.shape[2]:
            out_[jj, 4] = 2*(region_t.shape[2] - out_[jj, 2])
        if out_[jj, 3] - out_[jj, 5]/2 < 0:  
            out_[jj, 5] = 2*out_[jj, 3]
        if out_[jj, 3] + out_[jj, 5]/2 > region_t.shape[3]:
            out_[jj, 5] = 2*(region_t.shape[3] - out_[jj, 3])

    bboxes_ = out_ # 2400,7
    # 目标框的 x, y, w, h 坐标归一化到 [0, 1] 之间，以适应归一化输入。
    bboxes_[:, 2:6] /= region_t.shape[2]
    # region_t_np = region_t.squeeze(0).cpu().numpy() # squeeze(0) 会移除第一个维度（如果它的大小为1）。(1, 3, 320, 320) -> (3, 320, 320)
    # 假设 region_t 维度为 (batch_size, channels, height, width)，这里取 batch 里的第一张图 (region_t[0])。
    region_t_np = region_t[0].cpu().numpy() # (8, 3, 320, 320) -> (3, 320, 320)
    #   原本的格式是 (C, H, W)（通道优先） 变成 (H, W, C)（通道最后），以适应 transform_ 函数的输入格式。
    region_t_np = np.transpose(region_t_np, (1, 2, 0))
    
    #  移除宽度或高度小于等于 0 的目标框，防止后续计算出错
    bboxes_ = bboxes_[bboxes_[:, 4] > 0]
    bboxes_ = bboxes_[bboxes_[:, 5] > 0]
    # bboxes_.shape = torch.Size([263, 7])

    if bboxes_.shape[0]: # 如果有目标框
        category_ids = [0] * bboxes_.shape[0] # 创建类别 ID（这里默认都为 0）
        transformed = transform_(image=region_t_np, bboxes= bboxes_[:, 2:6], category_ids=category_ids)
        transformed_img =  np.transpose(transformed['image'], (2,0,1)) # 变回 PyTorch 的格式 (C, H, W)，以便继续训练
        bboxes_transformed = transformed['bboxes'] # 增强后的坐标 len():263
        
        # 检查增强后的目标框数量
        if len(bboxes_transformed) != bboxes_.shape[0]:
            print(f"Warning: Number of transformed bboxes ({len(bboxes_transformed)}) does not match original bboxes ({bboxes_.shape[0]}).")
            # 过滤掉无效的目标框
            valid_indices = [i for i, bb in enumerate(bboxes_transformed) if bb is not None]
            bboxes_ = bboxes_[valid_indices]  # 更新 bboxes_

        bboxes_t = [list(bb) for bb in bboxes_transformed] 
        bboxes_t = torch.FloatTensor(bboxes_t)    #  把 list 转换回 PyTorch Tensor [263,4]
        # 确保形状匹配
        if bboxes_t.shape[0] == bboxes_.shape[0]:
            bboxes_t[:, [0, 2]] *= transformed_img.shape[2] # x, w 乘回图片的 宽度，还原真实坐标
            bboxes_t[:, [1, 3]] *= transformed_img.shape[1] # y, h 乘回图片的 高度，还原真实坐标
            bboxes_[:, 2:6] = bboxes_t
        else:
            raise ValueError(f"Shape mismatch: bboxes_t.shape={bboxes_t.shape}, bboxes_.shape={bboxes_.shape}")
        
        # print("bboxes_.shape:", bboxes_.shape) # (,7)
        # print("bboxes_t.shape:", bboxes_t.shape)  # Debugging (,4)
        
        return transformed_img, bboxes_
    else: 
        return region_t.squeeze(0).cpu().numpy(), np.ones((1, 7)) # 如果没有目标框，返回默认值


def clip_coords_target(target, w0, w1, h0, h1):
    # Clip bounding xywh bounding boxes to image shape (height, width)
    temp_coords = xywh2xyxy(target[:,2:6])
    temp_coords[:, 0].clamp_(w0, w1)  # x1
    temp_coords[:, 1].clamp_(h0, h1)  # y1
    temp_coords[:, 2].clamp_(w0, w1)  # x2
    temp_coords[:, 3].clamp_(h0, h1)  # y2
    return xyxy2xywh(temp_coords)
