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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InputDataset(Dataset):
    def __init__(self, input_dir, transform=None, img_size=640):

        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        # self.input2_files = sorted([os.path.join(input2_dir, f) for f in os.listdir(input2_dir)])
        self.transform = transform
        self.img_size = img_size

        # assert len(self.input_files) == len(self.input2_files), "两个输入路径下的文件数量不匹配"

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
        # input2 = self.preprocess_image(self.input2_files[idx])
        if self.transform:
            input = self.transform(input)
            # input2 = self.transform(input2)
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


def load_data(source, target, batch_size):
    """
    加载数据集并返回 DataLoader 对象。
    """
    source_dataset = InputDataset(source)
    target_dataset = InputDataset(target)

    source_data_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_data_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    return source_data_loader,target_data_loader


def initialize_yolo_model(yolo_weights_path, config_path, layers_to_extract):
    # 加载 YOLO 模型
    yolo_model = YOLO(yolo_weights_path).model
    # yolo_model.eval()
    # 初始化特征提取器
    feature_extractor = YOLOFeatureExtractor(yolo_model, layers_to_extract, config_path).to(device)
    return feature_extractor


def do_feature_extraction(feature_extractor, data_loader, layers_to_extract, n_samples=-1):
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

        # 遍历每个需要提取的层
        for layer in layers_to_extract:
            layer_key = f"layer_{layer}"
            # 提取特定层的特征
            input_features = features[layer_key].view(len(input), -1)  # 展平特征
            # 转换为 NumPy 格式并保存
            projected_features[layer_key].append(input_features.detach().cpu().numpy())
    # # # 将特征拼接为单个数组
    for layer_key in projected_features:
        projected_features[layer_key] = np.concatenate(projected_features[layer_key], axis=0)
    
    return projected_features

# project random n_samples in current_data and new_data
def project_features(source,target,yolo_weights_path,config_path,layers_to_extract,n_samples):

    source_data_loader,target_data_loader = load_data(source,target,batch_size)
    feature_extractor = initialize_yolo_model(yolo_weights_path, config_path, layers_to_extract)
    
    source_projected_features = do_feature_extraction(feature_extractor,source_data_loader,layers_to_extract,n_samples)
    target_projected_features = do_feature_extraction(feature_extractor,target_data_loader,layers_to_extract,n_samples)
    
    source_project_data = {}
    target_project_data = {}
    for layer in layers_to_extract:
        layer_key = f"layer_{layer}"
        source_project_data[layer_key] = torch.tensor(source_projected_features[layer_key])
        target_project_data[layer_key] = torch.tensor(target_projected_features[layer_key])

    return source_project_data,target_project_data

# compute gap on projected data then average them
def compute_dss(features_1, features_2):
    n_proj = features_1.shape[1]
    print('刚进来的source ',features_1.shape) # 60, 2457600
    dss_val = 0
    for i in range(n_proj):
        feas_1 = features_1[:, i]
        feas_2 = features_2[:, i]
        
        feas_1 = feas_1[:, None]
        feas_2 = feas_2[:, None]

        d = DSS(feas_1, feas_2)
        dss_val += d

    dss_val /= n_proj
    return dss_val

# compute DSS
def DSS(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    # print('变化后的source ',source.shape) #[60,1]
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1) # 矩阵相乘

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)
    return loss

'''
(1) ot.unif(feas_1.shape[0]) 和 ot.unif(feas_2.shape[0])
    ot.unif(n) 生成了一个均匀分布向量，表示两个分布的权重。
    对于 feas_1.shape[0] 和 feas_2.shape[0]：
    它们分别是两个特征集 feas_1 和 feas_2 的样本数量。
    均匀分布假设每个样本的权重是均等的（即权重为 $1/n$）。
(2) cost_matrix
    成本矩阵，用于量化两个分布中样本点之间的相似度（或距离）。
    一般由样本点的特征向量计算，例如欧几里得距离或余弦距离：
    cost_matrix[i][j]=distance(feas_1[i],feas_2[j])
    矩阵大小为 (feas_1.shape[0], feas_2.shape[0])。
(3) numItermax=1e6
    最大迭代次数，控制 EMD 求解算法的计算时间。
    如果问题规模较大，增加 numItermax 的值可以确保收敛。
'''

def compute_swd(features_1, features_2):
    n_proj = features_1.shape[1]

    wasserstein_distance = 0
    for i in range(n_proj):
        feas_1 = features_1[:, i]
        feas_2 = features_2[:, i]
        
        feas_1 = feas_1[:, None]
        feas_2 = feas_2[:, None]

        cost_matrix = cdist(feas_1, feas_2)**2
        # cost_matrix = ot.dist(feas_1, feas_2, metric='euclidean')
        cost_matrix = cost_matrix.detach().cpu().numpy()
        
        gamma = ot.emd(ot.unif(feas_1.shape[0]), ot.unif(feas_2.shape[0]), cost_matrix, numItermax=1e6)
        wasserstein_distance += np.sum(np.multiply(gamma, cost_matrix)).to(device)
        
        del cost_matrix
 
    wasserstein_distance /= n_proj
    return wasserstein_distance


def compute_swd_2(features_1, features_2):

    n_proj = features_1.shape[1]

    # Precompute uniform distributions for OT
    unif_1 = torch.ones(features_1.shape[0], dtype=torch.float32) / features_1.shape[0]
    unif_2 = torch.ones(features_2.shape[0], dtype=torch.float32) / features_2.shape[0]

    # Reshape features for batch processing
    feas_1 = features_1.unsqueeze(-1)  # (n_samples, n_proj, 1)
    feas_2 = features_2.unsqueeze(-1)  # (n_samples, n_proj, 1)

    # Batch OT computation
    # wasserstein_distances = []
    wasserstein_distances = 0

    for i in range(n_proj):
        # Select the i-th projection for both feature sets
        proj_feas_1 = feas_1[:, i, :]
        proj_feas_2 = feas_2[:, i, :]

        # Compute the pairwise cost matrix
        cost_matrix = torch.cdist(proj_feas_1, proj_feas_2, p=2)  # Squared L2 distance

        # Convert to NumPy for OT computation (if necessary)
        cost_matrix_np = cost_matrix.cpu().numpy()
        unif_1_np = unif_1.cpu().numpy()
        unif_2_np = unif_2.cpu().numpy()

        # Compute the OT plan using POT
        gamma = ot.emd(unif_1_np, unif_2_np, cost_matrix_np, numItermax=1e6)

        # Compute Wasserstein distance for the current projection
        # wasserstein_distances.append(np.sum(gamma * cost_matrix_np))
        wasserstein_distances += np.sum(gamma * cost_matrix_np)

    # Compute the mean Wasserstein distance across all projections
    # return np.mean(wasserstein_distances)
    return wasserstein_distances / n_proj


def compute_mmd(features_1, features_2):
    delta = features_1 - features_2
    return delta.dot(delta.T)


# compute gaps in res3, res4 and res5 and average them
def compute_avg_gap(features_1, features_2, gap_func):
    gap = 0
    length = 1
    for layer in layers_to_extract:
        layer_key = f"layer_{layer}"
        fea_1 = features_1[layer_key].to(device)
        fea_2 = features_2[layer_key].to(device)
        gap += gap_func(fea_1, fea_2)
    print('总 Gap :',gap)
    length = len(layers_to_extract)
    return gap / length


def do_mean_features_extraction(feature_extractor, data_loader, layers_to_extract, total_samples):
    print(f"Mean -- Start extracting features from {total_samples} batches")
    # # 如果未指定批次限制，使用数据加载器的长度
    # total = len(data_loader) if n_samples == -1 else n_samples
    # 初始化存储特征和均值的变量
    sum_features = {f"layer_{layer}": 0 for layer in layers_to_extract}
    for input, batch_idx in zip(data_loader,range(total_samples)):
        if batch_idx % 10 == 1:
            print("Mean --Process image %d/%d" % (batch_idx, total_samples))
        
        # 将数据移动到 GPU 并提取特征
        input = input.to(device)
        _, features = feature_extractor(input)  

        # 遍历每个需要提取的层
        for layer in layers_to_extract:
            layer_key = f"layer_{layer}"        
            # 提取特定层的特征并累加
            input_features = features[layer_key].view(-1)  # 展平成二维
            sum_features[layer_key] += input_features.detach().cpu()
    # 计算每层的均值特征
    mean_features = {f"layer_{layer}": [] for layer in layers_to_extract}
    for layer_key, summed_feature in sum_features.items():
        mean_features[layer_key] = summed_feature / total_samples

    return mean_features


def compute_mean_features(source, target, yolo_weights_path, config_path, layers_to_extract,n_samples):
    source_data_loader,target_data_loader = load_data(source,target,batch_size)
    feature_extractor = initialize_yolo_model(yolo_weights_path, config_path, layers_to_extract)
    # 计算当前数据的均值特征
    mean_current_features = do_mean_features_extraction(feature_extractor,source_data_loader,layers_to_extract,n_samples)
    # 计算新数据的均值特征
    mean_new_features = do_mean_features_extraction(feature_extractor,target_data_loader, layers_to_extract,n_samples)
    return mean_current_features, mean_new_features


def compute_domain_gap(source_data, target_data, metric, yolo_weights_path,config_path,layers_to_extract,n_samples = 10):
    gap = 0
    if metric == "DSS": 
        # project data
        projected_current_data, projected_new_data = project_features(source_data, target_data, yolo_weights_path,config_path,layers_to_extract,n_samples)
        # compute average gap 
        print('\n===============> Compute DSS ...')
        gap = compute_avg_gap(projected_current_data, projected_new_data, compute_dss)
        print('===============> DSS done.')
    if metric == "SWD":
        # project data
        projected_current_data, projected_new_data = project_features(source_data, target_data, yolo_weights_path,config_path,layers_to_extract,n_samples)
        # compute average gap 
        print('\n###############> Compute SWD  ...')
        gap = compute_avg_gap(projected_current_data, projected_new_data, compute_swd)
        print('###############> SWD done .')
    if metric == "MMD":
        # compute mean
        mean_current_data, mean_new_data = compute_mean_features(source_data, target_data,yolo_weights_path,config_path, layers_to_extract,n_samples)
        # compute average gap
        print('\n--------------> Compute MMD  ...')
        gap = compute_avg_gap(mean_current_data, mean_new_data, compute_mmd)
        print('--------------> MMD done .')
    return gap

'''
在实验中，我们从源域和目标域随机选取了 50 幅图像。
为便于进一步处理，我们对所有特征图逐个通道进行了平均池化处理。
这种池化操作可以更简洁地表示特征，便于后续分析和可视化。
'''

# 数据路径
source = "/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/train"
target = "/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format/images/train"
# target = "/home/lenovo/data/liujiaji/Datasets/VPMBGI/img"

batch_size = 6
# 加载 YOLO 模型
yolo_weights_path = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/yolov8m.pt'
config_path = "/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/ultralytics/cfg/models/v8/yolov8.yaml"
layers_to_extract = [2,4,6]
high_layers = [8,9]
dss_domain_gap = compute_domain_gap(source,target,'DSS',yolo_weights_path,config_path,layers_to_extract,n_samples = 10)
# mmd_domain_gap = compute_domain_gap(source,target,'MMD',yolo_weights_path,config_path,layers_to_extract,n_samples = 100)
# swd_domain_gap = compute_domain_gap(source,target,'SWD',yolo_weights_path,config_path,layers_to_extract,n_samples = 100)
# print('DSS gap:',dss_domain_gap.item())
# print('MMD gap:',mmd_domain_gap.item())
# print('SWD gap:',swd_domain_gap.item())


'''
city-to-foggy
===============> Compute DSS ...
总 Gap : tensor(0.0016, device='cuda:0')
DSS gap: 0.00032728680525906384

--------------> Compute MMD  ...
总 Gap : tensor(106856.1250, device='cuda:0')
MMD gap: 21371.224609375

'''

'''
===============> Compute DSS ...
Gap : tensor(0.0008)
===============> DSS done.
DSS gap: 0.00016965848044492304

--------------> Compute MMD  ...
Gap : tensor(6546.4980, device='cuda:0')
--------------> MMD done .
MMD gap: 1309.2996826171875

###############> Compute SWD  ...
Gap : 0.06380485921465757
###############> SWD done .
SWD gap: 0.012760971842931516


'''