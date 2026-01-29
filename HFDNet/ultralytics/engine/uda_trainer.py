# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
# from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.nn.extra_modules.kernel_warehouse import get_temperature
##########################################################
from ultralytics.nn.uda_tasks import attempt_load_one_weight, attempt_load_weights
import torch.nn.functional as F
from ultralytics.utils.daca import  get_best_region, transform_img_bboxes, cross_set_cutmix_pseudo, cross_set_cutmix,adjust_alpha, gram_matrix,compute_mmd_loss, compute_swd_loss,clip_coords_target,compute_dss_loss
from ultralytics.utils.ops import  non_max_suppression
from ultralytics.utils.plotting import output_to_target, plot_images
import copy
import albumentations as A


class UDABaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        with torch_distributed_zero_first(RANK):  # avoid auto-downloading dataset multiple times
            self.trainset, self.targetset , self.testset = self.uda_get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            # elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
            #     LOGGER.info(
            #         f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
            #         "See ultralytics.engine.trainer for customization of frozen layers."
            #     )
            #     v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,
                amp=self.amp,
                batch=self.batch_size,
            )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        # self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
        print(" ************************ uda_trainer/get_dataloader ")
        self.train_loader , self.target_loader = self.uda_get_dataloader(self.trainset,self.targetset,batch_size=batch_size, rank=RANK, mode_S="train", mode_T="target")
       
        
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = min(len(self.train_loader),len(self.target_loader))  # number of batches
        # print('最小的批次大小 ',nb)

        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f'Using {self.target_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        
        max_iterations = nb * ( self.epochs - epoch)

        while True:
        # for epoch in range(self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                self.target_loader.sampler.set_epoch(epoch)

            # pbar = enumerate(self.train_loader)
            pbar = enumerate(zip(self.train_loader,self.target_loader))

            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                # pbar = TQDM(enumerate(self.train_loader), total=nb)
                pbar = TQDM(pbar, total=nb)

            self.tloss = None

            for i, (batch_S,batch_T) in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                if hasattr(self.model, 'net_update_temperature'):
                    temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                    self.model.net_update_temperature(temp)

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    # ----------------------------------------------------- 
                    # 原来的 仅源域的检测损失 Source-only / 仅目标域的 Oracle
                    # batch = self.preprocess_batch(batch)
                    # self.loss, self.loss_items = self.model(batch)
                    # if RANK != -1:
                    #     self.loss *= world_size
                    # self.tloss = (
                    #     (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    # )
                    # ----------------------------------------------------- 

                    batch_s = self.preprocess_batch(batch_S)
                    batch_t = self.preprocess_batch(batch_T)

                    # # 1.原始源域的监督损失
                    # self.loss, self.loss_items = self.model(batch_s)
                    # if RANK != -1:
                    #     self.loss *= world_size
                    # self.tloss = (
                    #     (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    # )
                    
                    # -------- 方法一 --------------------------------------------- 
                    #  源域 目标域的特征图 差异，作为第二个损失 最终优化目标为 与源域的检测损失 ，权重相加
                    '''
                    
                    # self.model(batch) 其中，batch是字典就计算loss,不是字典就计算 预测值
                    # 源域的检测损失
                    self.source_loss, self.source_loss_items = self.model(batch_s) # loss*batch,[cls,bbox,dfl]

                    # 仅 源域和目标域图像 的前向传播，返回特征图值
                    self.source_feature_dict = self.model(batch_s['img'],layers=True)  
                    self.target_feature_dict = self.model(batch_t['img'],layers=True)
                    
                    gram_losses = [] # Gram
                    mmd_losses = []
                    swd_losses = []
                    mse_losses = [] # l2

                    
                    for layer in [2, 4, 6, 8, 9]:
                        source_feas = self.source_feature_dict[layer]
                        target_feas = self.target_feature_dict[layer]
                        # # 检查批次大小
                        min_batch_size = min(source_feas.size(0), target_feas.size(0))
                        source_fea = source_feas[:min_batch_size]
                        target_fea = target_feas[:min_batch_size]
                        if source_fea is not None and target_fea is not None:
                            # 检查源域和目标域特征的形状是否一致
                            if source_fea.shape != target_fea.shape: 
                                # 调整 target_feature 的尺寸，使其匹配 source_feature
                                target_fea = F.interpolate(
                                    target_fea, 
                                    size=target_fea.shape[2:],  # 调整为目标特征图的高度和宽度
                                    mode="bilinear", 
                                    align_corners=False
                                )
                            # 计算 特征 损失
                            if layer in [2, 4, 6]: 
                                # gram值太小，对结果影响很小
                                # gram_s = gram_matrix(source_fea)
                                # gram_t = gram_matrix(target_fea)
                                # gram_loss = F.mse_loss(gram_s, gram_t).to(self.device)
                                # gram_losses.append(gram_loss)
                            # mean_gram_loss = sum(gram_losses) / 3
                                
                                # mmd_linear 在50epoch还行，100epoch就变很小值了！
                                mmd_loss = torch.tensor(compute_linearmmd_loss(source_fea,target_fea))
                                mmd_losses.append(mmd_loss)
                            mean_mmd_loss = sum(mmd_losses) / 3  

                            #     # swd 不是很好，但比gram好一些
                            #     swd_loss = torch.tensor(compute_swd_loss(source_fea,target_fea))
                            #     swd_losses.append(swd_loss)
                            # mean_swd_loss = sum(swd_losses) / 3  


                            if layer in [8, 9]: # [2,4,6,8,9]
                                mse_loss = F.mse_loss(source_fea, target_fea)
                                mse_losses.append(mse_loss)
                            mean_mse_loss = sum(mse_losses) / 2
    
                            # else:# 如果源域或目标域特征为空，跳过计算
                            #     # print('WARNING  source target features is None!!!')
                            #     mse_loss = torch.Tensor(0)
                            #     gram_loss = torch.Tensor(0)  


                    # 计算最终损失
                    alpha_weight = 0.05 # 超参数，用于平衡 gram、mmd、swd
                    lambda_weight = 0.1  # 超参数，用于平衡 MSE损失              
                    self.loss = self.source_loss + lambda_weight * mean_mse_loss + alpha_weight * mean_mmd_loss
                    # 将 mean_mse_loss 和 mean_gram_loss 加入 loss_items
                    self.loss_items = torch.cat([
                        self.source_loss_items,  # 原有的 cls、bbox、dfl 损失
                        mean_mse_loss.detach().unsqueeze(0),   # 加入 mse 损失
                        mean_mmd_loss.detach().unsqueeze(0),  # 加入 gram\mmd\swd 损失
                    ])

                    # 多GPU训练时的损失调整
                    if RANK != -1:
                        self.loss *= world_size
                    # 更新平均损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    ) # i 是batch索引
                    '''
                    # ----------------------------------------------------- 

                    # ----------方法二 ------------------------------------------- 
                    # 基于伪标签的合成域，二次训练
                    '''
                    # # supervised detector loss term on the labelled source samples
                    # 1.源域的检测损失
                    self.source_loss, self.source_loss_items = self.model(batch_s) # pred_s 
                    
                    
                    ## 伪标签
                    r = ni / max_iterations
                    delta = 2 / (1 + math.exp(-5. * r)) - 1
                    pred_s = self.model(batch_s['img'], pseudo=True, delta=delta)  # forward          
                    pseudo_s, pred_s = pred_s # 源域 的 检测结果，特征图
                
                    pred_t = self.model(batch_t['img'], pseudo=True, delta=delta)  # forward
                    pseudo_t, _ = pred_t # 目标域的 伪标签 和 特征图 pseudo_t.shape(4,5,8400)
                   
                    
                    ##  2.1 简单的 基于目标域伪标签的训练
                    # # # filter pseudo detections on target images applying NMS
                    # out = non_max_suppression(pseudo_t.detach(), conf_thres=0.25, iou_thres=0.25, multi_label=False)
                    # out = torch.tensor(output_to_target(out))  # [batch_id, class_id, x, y, w, h, conf] (16,7)
                    # out_original = copy.deepcopy(out)
                    # b, c, h, w = batch_t['img'].shape
                    # out[:, [2, 4]] /= w
                    # out[:, [3, 5]] /= h  
                
                    # psbatch_t = batch_t.copy()
                    # psbatch_t['cls'] = out[:,1].unsqueeze(-1) # [32] -> [32,1]
                    # psbatch_t['bboxes'] = out[:,2:6] # [32,4]
                    # psbatch_t['batch_idx'] = out[:,0] # [32]
                    # self.daca_loss, self.daca_loss_items = self.model(psbatch_t)

                    # c_gamma_thres = 0.5
                    # #  .sum() 对布尔张量求和。True=1，False=0。 返回的是满足条件（大于 0.5）的元素个数。
                    # #  .nelement() 返回张量中元素的总数。对于 targets_confmix[:, 6]，它是一个形状为 [N] 的一维张量，因此 返回 N。
                    # gamma = (out[:,6] > c_gamma_thres).sum() / \
                    #                 (out[:,6]).nelement()
                    
                    
                    # 2.2 DACA
                    # 创建一个与源图像 imgs_s 形状相同的全 1 张量，并将其乘以 imgs_s 的均值。
                    # 目的是生成一个与 imgs_s 大小相同的空白图像，用于后续拼接增强后的图像。
                    imgs_concat = torch.ones_like(batch_s['img']) * torch.mean(batch_s['img']) #  初始化合成图像，进行再次训练
                    if out.shape[0] > 0: #（16，4） 如果 out 的行数大于 0，说明有目标框需要处理。
                        # get best region from target 从目标域中选 最好的区域
                        region_t1_original, out1_original, best_side = get_best_region(out, batch_t['img']) 
                        # torch.Size([4, 3, 320, 320]),(16,7),''topleft''  

                        transform = A.Compose([
                                            A.BBoxSafeRandomCrop(erosion_rate=0.1, always_apply=False, p=0.2),
                                            A.HorizontalFlip(p=0.5),
                                            A.Blur(blur_limit=1, always_apply=True, p=0.5), 
                                            A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                                            A.Downscale (scale_min=0.5, scale_max=0.99, interpolation=None, always_apply=False, p=0.5),
                                            A.RandomBrightnessContrast (brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
                                            ], 
                                            bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),)              
                        
                        # 对最佳区域进行增强
                        region_t1, out1 = transform_img_bboxes(out1_original, best_side, region_t1_original, transform)
                        region_t2, out2 = transform_img_bboxes(out1_original, best_side, region_t1_original, transform)
                        region_t3, out3 = transform_img_bboxes(out1_original, best_side, region_t1_original, transform)
                        region_t4, out4 = transform_img_bboxes(out1_original, best_side, region_t1_original, transform)

                        # fill up the concat image
                        # 将增强后的 4 个区域 region_t1 到 region_t4 拼接到 imgs_concat 的不同位置，形成一张新的拼接图像。
                        imgs_concat[:, :, 0:int(region_t1.shape[1]), 0:int(region_t1.shape[2])] = torch.from_numpy(region_t1).unsqueeze(0)
                        imgs_concat[:, :, int(batch_s['img'].shape[3]/2):int(batch_s['img'].shape[3]/2) + int(region_t2.shape[1]), 0:int(region_t2.shape[2])] = torch.from_numpy(region_t2).unsqueeze(0)
                        imgs_concat[:, :, int(batch_s['img'].shape[3]/2):int(batch_s['img'].shape[3]/2) + int(region_t3.shape[1]),  int(batch_s['img'].shape[3]/2):int(batch_s['img'].shape[3]/2) + int(region_t3.shape[2])] = torch.from_numpy(region_t3).unsqueeze(0)
                        imgs_concat[:, :, 0:int(region_t4.shape[1]), int(batch_s['img'].shape[3]/2):int(batch_s['img'].shape[3]/2) + int(region_t4.shape[2])] = torch.from_numpy(region_t4).unsqueeze(0)

                        # Adjust region-level bboxes of the image-level coordinates
                        # 调整目标框坐标
                        # convert to bottomleft
                        out2[:, 3] += batch_t['img'].shape[3]/2
                        # convert to bottomright
                        out3[:, 2] += batch_t['img'].shape[2]/2
                        out3[:, 3] += batch_t['img'].shape[3]/2
                        # convert to topright
                        out4[:, 2] += batch_t['img'].shape[2]/2
                        
                        # 将目标框转换为张量
                        if not torch.is_tensor(out1):
                            out1 = torch.from_numpy(out1)
                        if not torch.is_tensor(out2):
                            out2 = torch.from_numpy(out2)
                        if not torch.is_tensor(out3):
                            out3 = torch.from_numpy(out3)                                                        
                        if not torch.is_tensor(out4):
                            out4 = torch.from_numpy(out4)       
                        out = torch.cat((out1, out2, out3, out4), dim=0) # shape (32,7)
                    else:
                        out = torch.empty([0,7]) 

                    imgs_daca = imgs_concat # 合成域的图像
                    # out_s = torch.from_numpy(out_s) if out_s.size else torch.empty([0,7])
                    b, c, h, w = imgs_daca.shape # [4,3,640,640]
                    
                    # create daca targets 
                    # targets_daca_s = out_s
                    targets_daca_t = out # (32,7) 合成域的 GT
                    targets_daca =  targets_daca_t # (32,7)
                    
                    targets_daca = targets_daca[:,:6] # remove confidence values [32,6]
                    # normalize
                    targets_daca[:, [2, 4]] /= w
                    targets_daca[:, [3, 5]] /= h
                    
                
                    # batch_s['img'].shape [4,3,640,640]
                    # batch_s['cls'].shape [109,1]
                    # batch_s['bboxes'].shape [109,4]
                    # batch_s['batch_idx'].shape [109]

                    # self-supervised consistency loss term on the mixed samples
                    # 2. 合成域的 二次检测
                    batch_daca = {}
                    batch_daca['ori_shape'] = batch_s['ori_shape']
                    batch_daca['resized_shape'] = [[640,640],[640,640],[640,640],[640,640]]
                    batch_daca['img'] = imgs_daca #[4,3,640,640]
                    batch_daca['cls'] = targets_daca[:,1].unsqueeze(-1) # [32] -> [32,1]
                    batch_daca['bboxes'] = targets_daca[:,2:] # [32,4]
                    batch_daca['batch_idx'] = targets_daca[:,0] # [32]
                    self.daca_loss, self.daca_loss_items = self.model(batch_daca)
                    
             
                    # 仅 源域和目标域图像 的前向传播，返回特征图值
                    self.source_feature_dict = self.model(batch_s['img'],layers=True)  
                    self.target_feature_dict = self.model(batch_t['img'],layers=True)
                    gram_losses = [] # 值太小
                    swd_losses = []
                    dss_losses = []
                    mmd_losses = []
                    mse_losses = [] # l2
                    smmd_losses = []
                    mmmd_losses = []
                    hmmd_losses = []
                    # for layer in [2, 4, 6, 8, 9, 12, 15, 18, 21, 22]:
                    for layer in [2, 4, 6, 8, 9]:
                        source_feas = self.source_feature_dict[layer]
                        target_feas = self.target_feature_dict[layer]
                        if isinstance(source_feas, torch.Tensor) and isinstance(target_feas, torch.Tensor):
                            min_batch_size = min(source_feas.shape[0], target_feas.shape[0])   # # 检查批次大小
                            source_fea = source_feas[:min_batch_size]
                            target_fea = target_feas[:min_batch_size]
                            if source_fea is not None and target_fea is not None:
                                # 检查源域和目标域特征的形状是否一致
                                if source_fea.shape != target_fea.shape: 
                                    # 调整 target_feature 的尺寸，使其匹配 source_feature
                                    target_fea = F.interpolate(
                                        target_fea, 
                                        size=target_fea.shape[2:],  # 调整为目标特征图的高度和宽度
                                        mode="bilinear", 
                                        align_corners=False
                                    )
                                
                                # 3.计算源域和目标域的 特定层特征差异损失   ，缩小域间差异
                                # if layer in [2, 4, 6, 8, 9]:  # backbone
                                #     # gram值太小，对结果影响很小
                                #     gram_s = gram_matrix(source_fea)
                                #     gram_t = gram_matrix(target_fea)
                                #     gram_loss = F.mse_loss(gram_s, gram_t).to(self.device)
                                #     # gram_loss = (1000 / source_fea.shape[1]**2) * (F.mse_loss(gram_s, gram_t).to(self.device))
                                #     gram_losses.append(gram_loss)

                                #     # swd_loss = torch.tensor(compute_swd_loss(source_fea,target_fea)) # 太慢了
                                #     # swd_losses.append(swd_loss)

                                #     # dss_loss = torch.tensor(compute_dss_loss(source_fea,target_fea)) # 太慢了
                                #     # dss_losses.append(dss_loss)

                                # if layer in [12,15,18,21]:  # neck 高斯核
                                #     mmd_loss = torch.tensor(compute_mmd_loss(source_fea,target_fea))
                                #     mmd_losses.append(mmd_loss)
                                 

                                if layer in [2,4]:  # 
                                    dss_loss = torch.tensor(compute_dss_loss(source_fea,target_fea)) # 太慢了
                                    dss_losses.append(dss_loss)
                                if layer in [6]:  # 
                                    mmmd_loss = torch.tensor(compute_mmd_loss(source_fea,target_fea))
                                    mmmd_losses.append(mmmd_loss)
                                if layer in [8,9]:  # 
                                    mse_loss = F.mse_loss(source_fea, target_fea)
                                    mse_losses.append(mse_loss)
                        
                        # # head layer = [22] 层,多尺度【80，40，20】
                        # if isinstance(source_feas, list) and isinstance(target_feas, list): 
                        #     for i in range(len(source_feas)):
                        #         min_batch_size = min(source_feas[i].shape[0], target_feas[i].shape[0])
                        #         source_fea = source_feas[i][:min_batch_size]
                        #         target_fea = target_feas[i][:min_batch_size]
                        #         mse_loss = F.mse_loss(source_fea, target_fea)
                        #         mse_losses.append(mse_loss)
                        
                    # mean_gram_loss = torch.mean(torch.stack(gram_losses))  # 计算平均值
                    # mean_swd_loss = torch.mean(torch.stack(swd_losses))
                    mean_dss_loss = torch.mean(torch.stack(dss_losses))
                    # mean_mmd_loss = torch.mean(torch.stack(mmd_losses))  # 计算平均值
                    # mean_smmd_loss = torch.mean(torch.stack(smmd_losses))
                    mean_mmmd_loss = torch.mean(torch.stack(mmmd_losses)) 
                    # mean_hmmd_loss = torch.mean(torch.stack(hmmd_losses)) 
                    mean_mse_loss = torch.mean(torch.stack(mse_losses)) 
                    

                    # self.loss = self.source_loss + self.args.daca_weight * self.daca_loss
                    self.loss = self.source_loss + torch.nan_to_num(gamma) * self.confmix_loss + self.args.shallow_weight * mean_dss_loss + self.args.middle_weight * mean_mmmd_loss + self.args.high_weight * mean_mse_loss  
                    # self.loss = self.source_loss + self.args.shallow_weight * mean_dss_loss + self.args.middle_weight * mean_mmmd_loss + self.args.high_weight * mean_mse_loss 
                    self.loss_items = torch.cat([
                        self.source_loss_items,  # 原有的 cls、bbox、dfl 损失
                        # self.daca_loss_items,
                        self.confmix_loss_items,
                        mean_dss_loss.detach().unsqueeze(0), # 加入 gram 损失
                        # mean_swd_loss.detach().unsqueeze(0), # 加入 gram 损失
                        # mean_dss_loss.detach().unsqueeze(0), # 加入 gram 损失
                        # mean_mmd_loss.detach().unsqueeze(0),  # 加入 mmd 损失
                        # mean_smmd_loss.detach().unsqueeze(0),
                        mean_mmmd_loss.detach().unsqueeze(0),
                        # mean_hmmd_loss.detach().unsqueeze(0)
                        mean_mse_loss.detach().unsqueeze(0),   # 加入 mse 损失

                    ])
                    
                    # print('最终实际的loss_items',self.loss_items)
                    # 多GPU训练时的损失调整
                    if RANK != -1:
                        self.loss *= world_size
                    # 更新平均损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    '''
                    

                    if batch_s['img'].shape[0] != batch_t['img'].shape[0]:
                        min_batch_size = min(batch_s['img'].shape[0], batch_t['img'].shape[0])
                        batch_s['img'] = batch_s['img'][:min_batch_size]
                        batch_t['img'] = batch_t['img'][:min_batch_size]

                    r = ni / max_iterations
                    delta = 2 / (1 + math.exp(-5. * r)) - 1
                    
                    pred_s = self.model(batch_s['img'], pseudo=True, delta=delta)  # forward          
                    pseudo_s, pred_s = pred_s # 源域 的 检测结果，特征图
                    pred_t = self.model(batch_t['img'], pseudo=True, delta=delta)  # forward
                    pseudo_t, _ = pred_t # 目标域的 伪标签 和 特征图 pseudo_t.shape(4,5,8400)
                    
                    # filter pseudo detections on source images applying NMS
                    out_s = non_max_suppression(pseudo_s.detach(), conf_thres=0.25, iou_thres=0.5, multi_label=False)
                    out_s = output_to_target(out_s)  # [batch_id, class_id, x, y, w, h, conf] (16,7)
                    # filter pseudo detections on target images applying NMS
                    out_t = non_max_suppression(pseudo_t.detach(), conf_thres=0.25, iou_thres=0.5, multi_label=False)
                    out_t = output_to_target(out_t)  # [batch_id, class_id, x, y, w, h, conf] (16,7)

                    # if out_t[:,6] > out_s[:,6]: 
                    # confmix
                    b, c, h, w = batch_s['img'].shape
                    out_s = torch.from_numpy(out_s) if out_s.size else torch.empty([0,7])
                    out_t = torch.from_numpy(out_t) if out_t.size else torch.empty([0,7]) 

                    # divide the pseudo detections on the target into 4 regions ([0,0] is top-left)
                    tar_lb = out_t[(out_t[:,2] < w//2) & (out_t[:,3] >= h//2), :]
                    tar_lt = out_t[(out_t[:,2] < w//2) & (out_t[:,3] < h//2), :]
                    tar_rb = out_t[(out_t[:,2] >= w//2) & (out_t[:,3] >= h//2), :]
                    tar_rt = out_t[(out_t[:,2] >= w//2) & (out_t[:,3] < h//2), :]
                    target_regions = [tar_lb, tar_lt, tar_rb, tar_rt]

                    # select the most confident region
                    mean_confidences = torch.nan_to_num(torch.as_tensor([torch.mean(i[:,6]) for i in target_regions]))  # Column 6 includes the confidence of the predictions
                    index = torch.max(mean_confidences, 0)[1]

                    # create binary mask for the confmix image and filter the source pseudo detections based on the selected region
                    mask = torch.zeros((b, c, h, w)).to(self.device)
                    if index == 0:
                        # tar_lb
                        tar_lb[:,2:6] = clip_coords_target(tar_lb, 0, w//2, h//2, h)
                        out_s = out_s[(out_s[:,2] >= w//2) | (out_s[:,3] < h//2), :]
                        out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), :], w//2, w, 0, h)
                        out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), :], 0, w, 0, h)
                        out_s[out_s[:,2] < w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] < w//2, :], 0, w, 0, h//2)

                        mask[:, :, h//2:h+1, 0:w//2] = 1.
                    elif index == 1:
                        # tar_lt
                        tar_lt[:,2:6] = clip_coords_target(tar_lt, 0, w//2, 0, h//2)
                        out_s = out_s[(out_s[:,2] >= w//2) | (out_s[:,3] >= h//2), :]
                        out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] < h//2), :], w//2, w, 0, h)
                        out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] >= w//2) & (out_s[:,3] >= h//2), :], 0, w, 0, h)
                        out_s[out_s[:,2] < w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] < w//2, :], 0, w, h//2, h)

                        mask[:, :, 0:h//2, 0:w//2] = 1.
                    elif index == 2:
                        # tar_rb
                        tar_rb[:,2:6] = clip_coords_target(tar_rb, w//2, w, h//2, h)
                        out_s = out_s[(out_s[:,2] < w//2) | (out_s[:,3] < h//2), :]
                        out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), :], w//2, w, 0, h)
                        out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), :], 0, w, 0, h)
                        out_s[out_s[:,2] >= w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] >= w//2, :], 0, w, 0, h//2)

                        mask[:, :, h//2:h+1, w//2:w+1] = 1.
                    elif index == 3:
                        # tar_rt
                        tar_rt[:,2:6] = clip_coords_target(tar_rt, w//2, w, 0, h//2)
                        out_s = out_s[(out_s[:,2] < w//2) | (out_s[:,3] >= h//2), :]
                        out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] < h//2), :], w//2, w, 0, h)
                        out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), 2:6] = clip_coords_target(out_s[(out_s[:,2] < w//2) & (out_s[:,3] >= h//2), :], 0, w, 0, h)
                        out_s[out_s[:,2] >= w//2, 2:6] = clip_coords_target(out_s[out_s[:,2] >= w//2, :], 0, w, h//2, h)

                        mask[:, :, 0:h//2, w//2:w+1] = 1.

                    # create confmix targets and compute confmix weight
                    targets_confmix_s = out_s #  [batch_id, class_id, x, y, w, h, conf]
                    targets_confmix_t = target_regions[index]
                    targets_confmix = torch.cat((targets_confmix_s, targets_confmix_t))

                    c_gamma_thres = 0.5
                    #  .sum() 对布尔张量求和。True=1，False=0。 返回的是满足条件（大于 0.5）的元素个数。
                    #  .nelement() 返回张量中元素的总数。对于 targets_confmix[:, 6]，它是一个形状为 [N] 的一维张量，因此 返回 N。
                    gamma = (targets_confmix[:,6] > c_gamma_thres).sum() / \
                                    (targets_confmix[:,6]).nelement()

                    targets_confmix = targets_confmix[:,:6] # remove confidence values
                    # normalize
                    targets_confmix[:, [2, 4]] /= w
                    targets_confmix[:, [3, 5]] /= h

                    # create confmix image
                    # print(f"batch_s['img'] shape: {batch_s['img'].shape}")
                    # print(f"batch_t['img'] shape: {batch_t['img'].shape}")
                    # print(f"mask shape: {mask.shape}")
                    imgs_confmix = batch_s['img'] * (1-mask) + batch_t['img'] * mask
                    imgs_confmix = imgs_confmix.to(self.device, non_blocking=True).float() / 255.0

                    # self-supervised consistency loss term on the mixed samples
                    # 2. 合成域的 二次检测 损失
                    batch_confmix = {}
                    batch_confmix['ori_shape'] = batch_s['ori_shape']
                    batch_confmix['resized_shape'] = [[640,640],[640,640],[640,640],[640,640]]
                    batch_confmix['img'] = imgs_confmix #[4,3,640,640]
                    batch_confmix['cls'] = targets_confmix[:,1].unsqueeze(-1) # [32] -> [32,1]
                    batch_confmix['bboxes'] = targets_confmix[:,2:] # [32,4]
                    batch_confmix['batch_idx'] = targets_confmix[:,0] # [32]
                    self.confmix_loss, self.confmix_loss_items = self.model(batch_confmix)
            
                    # supervised detector loss term on the labelled source samples
                    # 1. 源域的检测损失
                    self.source_loss, self.source_loss_items = self.model(batch_s) # pred_s 
                    # batch_s['img'].shape [4,3,640,640]
                    # batch_s['cls'].shape [109,1]
                    # batch_s['bboxes'].shape [109,4]
                    # batch_s['batch_idx'].shape [109]

                    # 仅 源域和目标域图像 的前向传播，返回特征图值
                    self.source_feature_dict = self.model(batch_s['img'],layers=True)  
                    self.target_feature_dict = self.model(batch_t['img'],layers=True)
                    gram_losses = []
                    mmd_losses = []
                    dss_losses = []
                    mmmd_losses = []
                    mse_losses = [] # l2
                    for layer in [2, 4, 6, 8, 9]:
                        source_feas = self.source_feature_dict[layer]
                        target_feas = self.target_feature_dict[layer]
                        # # 检查批次大小
                        min_batch_size = min(source_feas.size(0), target_feas.size(0))
                        source_fea = source_feas[:min_batch_size]
                        target_fea = target_feas[:min_batch_size]
                        if source_fea is not None and target_fea is not None:
                            # 检查源域和目标域特征的形状是否一致
                            if source_fea.shape != target_fea.shape: 
                                # 调整 target_feature 的尺寸，使其匹配 source_feature
                                target_fea = F.interpolate(
                                    target_fea, 
                                    size=target_fea.shape[2:],  # 调整为目标特征图的高度和宽度
                                    mode="bilinear", 
                                    align_corners=False
                                )
                             # 3.计算源域和目标域的 特定层特征差异损失   ，缩小域间差异
                            if layer in [2, 4]: 
                                # gram值太小，对结果影响很小
                                # gram_s = gram_matrix(source_fea)
                                # gram_t = gram_matrix(target_fea)
                                # gram_loss = F.mse_loss(gram_s, gram_t).to(self.device)
                                # gram_losses.append(gram_loss)
                                dss_loss = torch.tensor(compute_dss_loss(source_fea,target_fea)) # 太慢了
                                dss_losses.append(dss_loss)
                            if layer in [6]: 
                                # mmd_linear 在50epoch还行，100epoch就变很小值了！
                                mmmd_loss = torch.tensor(compute_mmd_loss(source_fea,target_fea))
                                mmmd_losses.append(mmmd_loss)
                            if layer in [8, 9]: # [2,4,6,8,9]
                                mse_loss = F.mse_loss(source_fea, target_fea)
                                mse_losses.append(mse_loss)
                    
                    mean_dss_loss = torch.mean(torch.stack(dss_losses))
                    mean_mmmd_loss = torch.mean(torch.stack(mmmd_losses)) 
                    mean_mse_loss = torch.mean(torch.stack(mse_losses)) 
                    
                    # self.loss = self.source_loss +  self.confmix_loss * torch.nan_to_num(gamma)
                    # self.loss = self.source_loss + self.args.shallow_weight * mean_dss_loss + self.args.middle_weight * mean_mmmd_loss + self.args.high_weight * mean_mse_loss 
                    self.loss = self.source_loss +  self.confmix_loss * torch.nan_to_num(gamma) + self.args.shallow_weight * mean_dss_loss + self.args.middle_weight * mean_mmmd_loss + self.args.high_weight * mean_mse_loss 
                    self.loss_items = torch.cat([
                        self.source_loss_items,  # 原有的 cls、bbox、dfl 损失
                        self.confmix_loss_items,
                        mean_dss_loss.detach().unsqueeze(0), # 加入 gram 损失
                        mean_mmmd_loss.detach().unsqueeze(0),  # 加入 mmd 损失
                        mean_mse_loss.detach().unsqueeze(0),   # 加入 mse 损失
                    ])
                    
                    # print('最终实际的loss_items',self.loss_items)
                    # 多GPU训练时的损失调整
                    if RANK != -1:
                        self.loss *= world_size
                    # 更新平均损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                   
                    # ----------------------------------------------------- 
                    
                    # -----------方法三---------------------------------------- 
                    # 对源域和目标域的合成增强
                    '''
                    # 1.原始源域的监督损失
                    self.source_loss, self.source_loss_items = self.model(batch_s)
                    
                    # 2.1 合成源域的监督损失 (源域+目标域的 cutmix)
                    alpha = adjust_alpha(epoch, self.epochs, initial_alpha=1.0, final_alpha=0.0)
                    mixed_img, mixed_cls, mixed_bbox = cross_set_cutmix(batch_s['img'], batch_t['img'], batch_s['cls'],batch_s['bboxes'], alpha)
                    if torch.all(mixed_cls == -1):
                        if epoch < self.epochs // 2:  # 训练前半段
                            self.mix_loss = torch.tensor(1, dtype=torch.float32).to(self.device)
                            self.mix_loss_items = torch.tensor([1, 1, 1], dtype=torch.float32).to(self.device)
                        else:  # 训练后半段
                            self.mix_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
                            self.mix_loss_items = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
                    else:
                        mixed_batch_st = batch_s.copy() # mixed_batch_s['batch_idx'].shape [203]
                        mixed_batch_st['img'] = mixed_img # [4,3,640,640]
                        mixed_batch_st['cls'] = mixed_cls # [203,3]
                        mixed_batch_st['bboxes'] = mixed_bbox # [203,12]
                        self.mix_loss, self.mix_loss_items = self.model(mixed_batch_st)
                #-------------
                    
                        
                    # 2.2  基于伪标签 重新进行cutmix
                    # r = ni / max_iterations
                    # delta = 2 / (1 + math.exp(-5. * r)) - 1
                    
                    # pred_s = self.model(batch_s['img'], pseudo=True, delta=delta)  # forward          
                    # pseudo_s, pred_s = pred_s # 源域 的 检测结果，特征图
                    # pred_t = self.model(batch_t['img'], pseudo=True, delta=delta)  # forward
                    # pseudo_t, _ = pred_t # 目标域的 伪标签 和 特征图 pseudo_t.shape(4,5,8400)
                    # out_s = non_max_suppression(pseudo_s.detach(), conf_thres=0.1, iou_thres=0.5, multi_label=False)
                    # out_s = output_to_target(out_s)  # [batch_id, class_id, x, y, w, h, conf] (16,7)
                    # out_t = non_max_suppression(pseudo_t.detach(), conf_thres=0.1, iou_thres=0.5, multi_label=False)
                    # out_t = output_to_target(out_t)  # [batch_id, class_id, x, y, w, h, conf] (16,7)
                    # out_s = torch.from_numpy(out_s) if out_s.size else torch.empty([0,7]) 
                    # out_t = torch.from_numpy(out_t) if out_t.size else torch.empty([0,7]) 

                    # alpha = adjust_alpha(epoch, self.epochs, initial_alpha=1.0, final_alpha=0.0)
                    # mixed_img, mixed_label = cross_set_cutmix_pseudo(batch_s['img'], batch_t['img'],out_s,out_t,alpha,conf_threshold=0.5)

                    # if torch.all(mixed_label[:,1] == -1):
                    #     if epoch < self.epochs // 2:  # 训练前半段
                    #         self.mix_loss = torch.tensor(1, dtype=torch.float32).to(self.device)
                    #         self.mix_loss_items = torch.tensor([1, 1, 1], dtype=torch.float32).to(self.device)
                    #     else:  # 训练后半段
                    #         self.mix_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
                    #         self.mix_loss_items = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
                    # else:
                    #     mixed_batch_st = batch_s.copy() # mixed_batch_s['batch_idx'].shape [203]
                    #     mixed_batch_st['img'] = mixed_img #[4,3,640,640]
                    #     mixed_batch_st['cls'] = mixed_label[:,1].unsqueeze(-1) # [32] -> [32,1]
                    #     mixed_batch_st['bboxes'] = mixed_label[:,2:6] # [32,4]
                    #     self.mix_loss, self.mix_loss_items = self.model(mixed_batch_st)
                
                #-------------
                
                    # 仅 源域和目标域图像 的前向传播，返回特征图值
                    # self.source_feature_dict = self.model(batch_s['img'],layers=True)  
                    # self.target_feature_dict = self.model(batch_t['img'],layers=True)  
                    # gram_losses = []
                    # mmd_losses = []
                    # mse_losses = []
                    # for layer in [2, 4, 6, 8, 9]:
                    #     source_feas = self.source_feature_dict[layer]
                    #     target_feas = self.target_feature_dict[layer]
                    #     # mix_target_feas = self.mixed_target_feature_dict[layer]
                    #     # # 检查批次大小
                    #     min_batch_size = min(source_feas.size(0), target_feas.size(0))
                    #     source_fea = source_feas[:min_batch_size]
                    #     target_fea = target_feas[:min_batch_size]
                    #     # mix_target_fea = mix_target_feas[:min_batch_size]
                    #     if source_fea is not None and target_fea is not None:
                    #         # 检查源域和目标域特征的形状是否一致
                    #         if source_fea.shape != target_fea.shape: 
                    #             # 调整 target_feature 的尺寸，使其匹配 source_feature
                    #             target_fea = F.interpolate(
                    #                 target_fea, 
                    #                 size=target_fea.shape[2:],  # 调整为目标特征图的高度和宽度
                    #                 mode="bilinear", 
                    #                 align_corners=False
                    #             )
                    #         # 3.计算源域和目标域的 特定层特征的   ，缩小域间差异
                    #         if layer in [2, 4]: 
                    #             # gram值太小，对结果影响很小
                    #             gram_s = gram_matrix(source_fea)
                    #             gram_t = gram_matrix(target_fea)
                    #             gram_loss = F.mse_loss(gram_s, gram_t).to(self.device)
                    #             gram_losses.append(gram_loss)
                    #         mean_gram_loss = sum(gram_losses) / 2

                    #         if layer in [6]: 
                    #             # mmd_linear 在50epoch还行，100epoch就变很小值了！
                    #             mmd_loss = torch.tensor(compute_linearmmd_loss(source_fea,target_fea))
                    #             mmd_losses.append(mmd_loss)
                    #         mean_mmd_loss = sum(mmd_losses)

                    #         if layer in [8, 9]: # [2,4,6,8,9]
                    #             mse_loss = F.mse_loss(source_fea, target_fea)
                    #             mse_losses.append(mse_loss)
                    #         mean_mse_loss = sum(mse_losses) / 2
                                

                    ## 4.计算 源域 和 合成域 的一致性损失,肯定很大啊，标签都复制了，又不是直接改变风格
                    ## loss_cons = torch.abs(self.source_loss - self.mix_loss)  # L1 loss
                    ## loss_cons = torch.abs(self.source_loss - self.mix_loss)**2   # L2 loss
                    
                    # 计算最终损失
                    self.loss = self.source_loss + self.args.mix_weight * self.mix_loss
                    # self.loss = self.source_loss +  self.args.gram_weight * mean_gram_loss + self.args.mmd_weight * mean_mmd_loss + self.args.mse_weight * mean_mse_loss 
                    # self.loss = self.source_loss + self.args.mix_weight * self.mix_loss + self.args.gram_weight * mean_gram_loss + self.args.mmd_weight * mean_mmd_loss + self.args.mse_weight * mean_mse_loss 
                    self.loss_items = torch.cat([
                        self.source_loss_items,  # 原有的 cls、bbox、dfl 损失
                        self.mix_loss_items, # 合成域
                        # mean_gram_loss.detach().unsqueeze(0),
                        # mean_mmd_loss.detach().unsqueeze(0),  # 加入 gram\mmd\swd 损失
                        # mean_mse_loss.detach().unsqueeze(0)   # 加入 mse 损失
                    ])

                    # 多GPU训练时的损失调整
                    if RANK != -1:
                        self.loss *= world_size
                    # 更新平均损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    '''
                    # ----------------------------------------------------- 
                
            
                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                # 确保 losses 是一个张量
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s%11s%11.4g%11.4g%11.4g%11.4g" % (
                            f"{epoch + 1}/{self.epochs}", mem,
                            losses[0],  # box_loss
                            losses[1],  # cls_loss
                            losses[2],  # dfl_loss
                            losses[3],  # mse_loss
                        ))
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        # self.plot_training_samples(batch_s, ni)
                        self.plot_training_samples(batch_t, ni) # 绘制的就是目标域
                        # self.plot_uda_samples(batch_daca,ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                
                # # 验证集性能
                # if self.metrics['metrics/mAP50(B)'] > best_mAP :
                #     best_mAP = self.metrics['metrics/mAP50(B)']
                #     best_params = (gamma_weight, alpha_weight, lambda_weight)

                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            gc.collect()
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1



        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
                # f"Best parameters (gamma, alpha, lambda): {best_params}, Best mAP: {best_mAP}" 
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")


    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        # buffer = io.BytesIO()
        # torch.save(
        #     {
        #         "epoch": self.epoch,
        #         "best_fitness": self.best_fitness,
        #         "model": None,  # resume and final checkpoints derive from EMA
        #         "ema": deepcopy(self.ema.ema).half(),
        #         "updates": self.ema.updates,
        #         "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
        #         "train_args": vars(self.args),  # save as dict
        #         "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
        #         "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
        #         "date": datetime.now().isoformat(),
        #         "version": __version__,
        #         "license": "AGPL-3.0 (https://ultralytics.com/license)",
        #         "docs": "https://docs.ultralytics.com",
        #     },
        #     # buffer,
        # )
        # serialized_ckpt = buffer.getvalue()  # get the serialized content to save
        
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,  # resume and final checkpoints derive from EMA
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        # Save checkpoints
        # self.last.write_bytes(serialized_ckpt)  # save last.pt
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            # self.best.write_bytes(serialized_ckpt)  # save best.pt
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            # (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")
    
    def uda_get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data["target"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")
    
    def uda_get_dataloader(self, dataset_path_S,dataset_path_T, batch_size=16, rank=0, mode_S="train", mode_T="target"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("uda_get_dataloader function not implemented in trainer")


    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    # def plot_uda_samples(self, batch, ni):
    #     """Plots uda_training samples during YOLO training."""
    #     pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in "imgsz", "batch", "device":  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer


