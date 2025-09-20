# FR3 Plane Grasping with GR-CNN

This is a demonstration of a planar grasping system using a Franka FR3 robot arm. It leverages a **trained GR-ConvNet model** for grasp pose detection from RGB-D data and controls the robot via [franky](https://github.com/TimSchneider42/franky).

<p align="left">
  <a href="">
    <img src="./assets/grasp_demo1.gif" alt="Demo" width="100%">
  </a>
</p>

## Key Components

* **Robot**: Franka FR3
* **Camera**: Intel RealSense D435i
* **Grasping Network**: [GR-CNN](https://github.com/skumra/robotic-grasping)
* **Control Interface**: [franky](https://github.com/TimSchneider42/franky) (`libfranka`: 0.15.0)

## Tested Environment

* **OS**: Ubuntu 22.04
* **Python**: 3.10
* **CUDA**: 12.4
* **PyTorch**: 2.6.0

## Installation
```bash
conda create -n grcnn python==3.10
conda activate grcnn

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# libfranka 0.15.0
pip install franky-control
```

## Inference Usage

#### Offline Testing with a Single RGB-D Frame
```bash
python run_offline.py
```

#### Real-time Grasp Detection with D435i
```bash
python run_realtime.py
```

#### Full Grasping Pipeline: Detection + Robot Control
> **Note**: This pipeline uses an eye-in-hand setup. Before use, please replace the calibration parameters in the code with your own.

```bash
python run_franky_grcnn_realtime.py
```


-----------------------------
# Antipodal Robotic Grasping
We present a novel generative residual convolutional neural network based model architecture which detects objects in the camera’s field of view and predicts a suitable antipodal grasp configuration for the objects in the image.

This repository contains the implementation of the Generative Residual Convolutional Neural Network (GR-ConvNet) from the paper:

#### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/antipodal-robotic-grasping-using-generative/robotic-grasping-on-cornell-grasp-dataset)](https://paperswithcode.com/sota/robotic-grasping-on-cornell-grasp-dataset?p=antipodal-robotic-grasping-using-generative)

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:

```
@inproceedings{kumra2020antipodal,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}}
}
```

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/skumra/robotic-grasping.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd robotic-grasping
$ pip install -r requirements.txt
```

## Datasets

This repository supports both the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

Example for Jacquard dataset:

```bash
python train_network.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard --use-dropout 0 --input-size 300
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

Example for Jacquard dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --iou-eval --use-dropout 0 --input-size 300
```

## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python run_grasp_generator.py
```

## Run on a Robot
To run the grasp generator with a robot, please use our ROS implementation for Baxter robot. It is available at: https://github.com/skumra/baxter-pnp
