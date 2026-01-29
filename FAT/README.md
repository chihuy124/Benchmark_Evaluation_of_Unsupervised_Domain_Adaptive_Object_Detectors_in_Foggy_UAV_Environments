#  Foggy-Aware Teacher: An Unsupervised Domain Adaptive Learning Framework for Object Detection in Foggy Scenes

![Framework](./data/FAT.png)

## Install dependencies
```shell
# create conda env
conda create -n Foggy_Aware_Teacher python=3.8
# activate the enviorment
conda activate Foggy_Aware_Teacher
# install 
pip install -r requirements.txt
```

## Prepare the Dataset
Download and prepare the dataset required for the experiments. Update the dataset path in the configuration file.
### Dataset
* **Cityscape and FoggyCityscape:**  Download Cityscapes dataset and Foggy-Cityscapes [shere](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data), and convert format from PASCAL_VOC to YOLO.
* **RTTS:** Please follow the [instruction](https://sites.google.com/site/boyilics/website-builder/reside) to prepare dataset.
* **VOC_fog:** Please follow the [instruction](https://drive.google.com/drive/folders/1P0leuiGHH69kVxyNVFuiCdCYXyYquPqM) to prepare dataset.

The datasets should be organized as follows:
```shell
datasets/
├── Cityscapes/
│   ├── images/
│   │   ├── Normal_train/
│   │   ├── Foggy_train/
│   │   └── Foggy_val/
│   ├── labels/
│   │   ├── Normal_train/
│   │   └── Foggy_val/
├── RTTS/
........

```

## Training the Model

1. Ensure the dataset paths are correctly configured in 
configuration file `ultralytics/datasets/VOC.yaml`.
2. Run the following command to start Brun-in stage :

```bash
python train_brun_in.py
```

3. Replace the paths in `train_foggy.py` with the model path trained on normal weather data during the Burn-in stage.

3. Run the following command to start TS mutual learning stage:

```bash
python train_foggy.py
```

### 3. Evaluating the Model
Replace the paths in `val.py` with the trained detector model path and the fog engine path, and run the following command:
```bash
python val.py
```
### 4. Citation
```bibtex
@article{qin2025foggy,
  title={Foggy-Aware Teacher: An Unsupervised Domain Adaptive Learning Framework for Object Detection in Foggy Scenes},
  author={Qin, Hongda and Lu, Xiao and Wang, Lucai and Wang, Yaonan},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
