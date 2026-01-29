# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Check a model's accuracy on a test or val split of a dataset

Usage:
    $ yolo mode=val model=yolov8n.pt data=coco128.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlmodel            # CoreML (macOS-only)
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
"""
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
import cv2
import numpy as np
import math
from ultralytics.yolo.utils import dip
class BaseValidator:
    """
    BaseValidator

    A base class for creating validators.

    Attributes:
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        args (SimpleNamespace): Configuration for the validator.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        speed (float): Batch processing speed in seconds.
        jdict (dict): Dictionary to store validation results.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
        """
        self.dataloader = dataloader
        self.pbar = pbar
        self.args = args or get_cfg(DEFAULT_CFG)
        self.model = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        self.jdict = None

        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = save_dir or increment_path(Path(project) / name,
                                                   exist_ok=self.args.exist_ok if RANK in (-1, 0) else True)
        (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
    def DarkChannel(self,im):
                #     dtype:
                # dtype('float32')
                # max:
                # 1.0
                # min:
                # 0.15294118
                # shape:
                # (304, 608, 3)
                # size:
                # 554496
            im = im.permute(1,2,0)
            im1 = im.cpu().detach().numpy()
            b, g, r = cv2.split(im1)
            dc = cv2.min(cv2.min(r, g), b)
            return dc

    def DarkChannel_add(self,im):
            b, g, r = cv2.split(im)
            dc = cv2.min(cv2.min(r, g), b)
            return dc

    def AtmLight(self,im, dark):
            im = im.permute(1,2,0)
            im = im.cpu().numpy()
            [h, w] = im.shape[:2]
            imsz = h * w
            numpx = int(max(math.floor(imsz / 1000), 1))
            darkvec = dark.reshape(imsz, 1)
            imvec = im.reshape(imsz, 3)

            indices = darkvec.argsort(0)
            indices = indices[(imsz - numpx):imsz]

            atmsum = np.zeros([1, 3])
            for ind in range(1, numpx):
                atmsum = atmsum + imvec[indices[ind]]
            A = atmsum / numpx
            return A

    def DarkIcA(self,im, A):
        im = im.permute(1,2,0)
        im = im.cpu().numpy()
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        return self.DarkChannel_add(im3)
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None,CNNPP = None, Fog_engine_path=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != 'cpu'
            model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
        if CNNPP or Fog_engine_path:
            Fog_engine =True
        else:
            Fog_engine = False
        if CNNPP :
            CNNPP.half() if self.args.half  else CNNPP.float()
            CNNPP.eval()
        else:
            CNNPP = dip.DIP()
            if Fog_engine_path:
                CNNPP.load_state_dict(torch.load(Fog_engine_path))
            CNNPP.to(self.device)
            CNNPP.eval()
        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
           
            with dt[0]:
                batch = self.preprocess(batch)
            if Fog_engine:
                h = batch['resized_shape'][0][0]
                w = batch['resized_shape'][0][1]
                # img_i = batch['img'][0][:,16:(h-16),16:(w-16)]  # img_ori_fog 640*640 shape:torch.Size([4, 3, 640, 640]) 3, 256, 480 224 448 [256, 480]
                batch['img'] = batch['img'].float()
                img_mosaic_64_304608 = batch['img'][:,:,16:(h-16),16:(w-16)]  # img_ori_fog 640*640 shape:torch.Size([4, 3, 640, 640]) 3, 256, 480 224 448 [256, 480]
                train_data = img_mosaic_64_304608
                dark = np.zeros((train_data.shape[0], train_data.shape[2], train_data.shape[3]))
                defog_A = np.zeros((train_data.shape[0], train_data.shape[1]))
                IcA = np.zeros((train_data.shape[0], train_data.shape[2], train_data.shape[3]))
                for i in range(train_data.shape[0]):
                    dark_i = self.DarkChannel(train_data[i])
                    defog_A_i = self.AtmLight(train_data[i], dark_i)
                    IcA_i = self.DarkIcA(train_data[i], defog_A_i)
                    dark[i, ...] = dark_i
                    defog_A[i, ...] = defog_A_i
                    IcA[i, ...] = IcA_i

                IcA = np.expand_dims(IcA, axis=-1)
                defog_A = torch.tensor(defog_A).to(self.device)
                IcA = torch.tensor(IcA).to(self.device)
                img_mosaic_64_304608 = img_mosaic_64_304608.half() if self.args.half  else img_mosaic_64_304608.float()
                imgs_tar_weak_defog, ci_map, Pr, filter_parameters = CNNPP(img_mosaic_64_304608, defog_A, IcA)
                imgs_tar_weak_defog = imgs_tar_weak_defog.to(self.device, non_blocking=True).float()
                batch['img'][:,:,16:(h-16),16:(w-16)]  = imgs_tar_weak_defog
                batch['img'] = batch['img'].half() if self.args.half  else batch['img'].float()
            # Inference
            with dt[1]:
                preds = model(batch['img'], augment=self.args.augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds=preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError('get_dataloader function not implemented for this validator')

    def build_dataset(self, img_path):
        """Build dataset"""
        raise NotImplementedError('build_dataset function not implemented in validator')

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {'data': data, 'timestamp': time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
