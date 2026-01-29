# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device
from util.utils import renorm

import torch
import numpy as np

import util.misc as utils
from util.misc import NestedTensor
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, nestedTensor_to_cuda

target_domain_flag=1
from util.vis_utils import plot_dual_img, plot_dual_img
from util.visualizer import COCOVisualizer
import copy
import cv2

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, teacher=None,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, postprocessors=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    visualizer = COCOVisualizer()
    visualized_dir = "vis_out"
    vis = True
    for batch in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # ? Unpack batch
        all_samples_wa_wa = batch[0]  # NestedTensor
        all_samples_wa_sa = batch[1]
        all_samples_sa_wa = batch[2]
        all_samples_sa_sa = batch[3]
        wa_1src_gts       = batch[4]  # Ground truth from source domain
    
        # ? Prepare input
        samples = all_samples_wa_wa.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in wa_1src_gts]
        domain_flag = torch.cat([t["domain"] for t in wa_1src_gts]).to(device)
    
        use_pseudo_label = False
    
        if args.training_phase != "wa_src_tgt":
            print("Other methods are not available yet, waiting for internal approval.")
            exit(0)
    
        # ========================= Begin training =========================
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, mask_dict = model(
                    samples,
                    dn_args=(targets, args.scalar, args.label_noise_scale,
                             args.box_noise_scale, args.num_patterns),
                    domain_flag=domain_flag,
                    use_pseudo_label=use_pseudo_label
                )
                loss_dict = criterion(outputs, targets, mask_dict)
            else:
                outputs, _ = model(
                    samples,
                    domain_flag=domain_flag,
                    use_pseudo_label=use_pseudo_label
                )
                loss_dict = criterion(outputs, targets, mask_dict={})
    
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    
        # ========================= Backward + Optim =========================
        loss_dict_reduced = utils.reduce_dict(loss_dict)
    
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
    
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
    
        if not math.isfinite(loss_value):
            print(f"? Loss is {loss_value}, stopping training.")
            print(loss_dict_reduced)
            sys.exit(1)
    
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
    
        # ========================= Log =========================
        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        lr_scheduler.step()
    
        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("?? BREAK TRAINING DEBUG MODE ??")
            break     
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None, teacher=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    print("len of data_loader = {} ".format(len(data_loader)))
    
    visualizer = COCOVisualizer()  # Initialize the COCOVisualizer
    visualized_dir = os.path.join(output_dir, 'visualizations') 
    
    for samples, targets in metric_logger.log_every(data_loader, 20, header, logger=logger):
        
        domain_flag = torch.cat([ele["domain"] for ele in targets]).to(device)       # bs
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, _ = model(samples, dn_args=args.num_patterns, domain_flag=domain_flag)
            else:
                outputs, _ = model(samples, domain_flag=domain_flag)
            
        weight_dict = criterion.weight_dict
        

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        # print("----in engine.py, done with eval, begin to write results")
        if args.save_results:
            """
            saving results of eval.
            """
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                if _res_prob.size(0) < _res_bbox.size(0):
                    _res_bbox = _res_bbox[:_res_prob.size(0)]  

                
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()
                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())
                




        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
                
    visualizer = COCOVisualizer()
    for i, (sample, target, result) in enumerate(zip(samples.tensors, targets, results)):
        img = sample.cpu()
        tgt = {
            'image_id': target['image_id'],
            'boxes': result['boxes'],
            'size': target['size'],
            'box_label': result['labels']
        }
        visualizer.visualize(img, tgt, caption=f"eval_sample{i}", savedir=output_dir, show_in_console=False)
                         
    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        precisions = coco_evaluator.coco_eval['bbox'].eval['precision']
        iou_thr_index = list(coco_evaluator.coco_eval['bbox'].params.iouThrs).index(0.5)

        precisions_iou50 = precisions[iou_thr_index, :, :, 0, -1]

        print("Per-class AP @ IoU=0.50:")
        ap50_per_class = []
        class_names = [c['name'] for _, c in coco_evaluator.coco_gt.cats.items()]
        for cls_id, cls_name in enumerate(class_names):
            prec = precisions_iou50[:, cls_id]
            prec = prec[prec > -1]
            ap50 = np.mean(prec) if prec.size else 0.0
            ap50_per_class.append(ap50)
            print(f"  {cls_name:20s}: {ap50:.3f}")
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]


    return stats, coco_evaluator

