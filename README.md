# DeepExpr

**Facial Expression and Pose Generation via Self-Supervised Disentangled Embeddings Fusion in Text-to-Image Diffusion Models**
## 🔍 Overview

**DeepExpr** addresses the limitations of existing text-to-image diffusion models in human-centric generation tasks, specifically facial expression and head pose control. While traditional models can render subjects in various textual contexts, they lack precision in manipulating key human attributes like **facial expression** and **pose** without degrading identity.
To overcome these limitations, DeepExpr introduces a two-stage solution:
- **Semantic and Identity Disentanglement (SID)** module  
- **Multi-Embedding Fusion** module

This repository includes the full implementation of the **SID module**.

---
## 🧠 Key Modules

### 1. Semantic and Identity Disentanglement (SID) Module

> 📦 `./modules/sid/`

A self-supervised module that disentangles identity and semantic attributes (such as expressions and pose) from single or paired image frames.  
It:
- Utilizes frame-to-frame supervision from static or video data
- Separates identity and expression into two independent latent spaces
- Enables composable expression and pose conditioning with identity-preserving fidelity

**Training SID:**
```bash
python train_sid.py --config configs/sid_config.yaml
## 🖼️ Example Scenarios

### 1. **Driving Video (Frame Sequence)**

In this scenario, the system takes:
- **Identity image**: A single static image of the subject (first column)
- **Driving video**: A sequence of frames capturing target expressions and poses (second column)
- **Generated sequence**: SID produces a series of synthesized images (third column) with the subject’s identity, but mimicking the expressions and poses from the video frames.

📷 **Left to Right:**
1. Identity reference image  
2. Frame-by-frame driving video sequence  
3. Output: Generated image sequence with matching expressions/poses

---
## 🎬 Driving Video Example

[![Watch the video](images/video_thumbnail.png)](images/driving_video.mp4)

➡️ Click the thumbnail above to watch the 10-second video showing identity preservation and expression transfer using DeepExpr.


### 2. **Static Image Expression and pose Transfer**

This setup uses:
- **Identity image**: Static image of the subject
- **Driving image**: A single static image with a target expression and pose
- **Output**: A generated image with the subject’s identity and the driving image's expression/pose

📷 **Left to Right:**
1. Identity image  
2. Reference expression image  
3. Output: Expression-aligned synthesis

## 🎬 Static-Image as reference Example
![Generated Images](images/Readme_Static.png)
*Generated Image with Modified Facial Expression and Head Pose*

Both scenarios leverage the **SID module** for disentangling identity from expression/pose, ensuring identity preservation across dynamic or static conditioning inputs.

### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. See ```config/taichi-256.yaml``` to get description of each parameter.


### Pre-trained checkpoint
Checkpoints can be found under folder named SID_Checkpoints: as sid-adv-cpk.pth and sid-cpk.pth.

### Inference
To run a Inference, download checkpoint and run the following command:
```
## 🎬 Running Inference

You can run inference using either a driving **video sequence** or a **static image** as the driving reference.

```bash
python Inference.py \
  --config config/dataset_name.yaml \
  --driving_video path/to/driving_video_or_image \
  --source_image path/to/source_image \
  --checkpoint path/to/checkpoint \
  --relative \
  --adapt_scale
```

- `--driving_video` can be a **video file** (e.g., `videos/driving_video.mp4`) for driving with a frame sequence,  
  or a **single image file** (e.g., `images/driving_image.png`) for driving with a static image reference.

- `--source_image` is the identity image to preserve during generation.

---

### Example usage

- Using a **video** as driving input:

```bash
python Inference.py --config config/dataset_name.yaml --driving_video videos/driving_video.mp4 --source_image images/source.png --checkpoint checkpoints/model.ckpt --relative --adapt_scale
```

- Using a **static image** as driving input:

```bash
python Inference.py --config config/dataset_name.yaml --driving_video images/driving_image.png --source_image images/source.png --checkpoint checkpoints/model.ckpt --relative --adapt_scale
```

---

### Output

- For **video-driven inference**, the generated result will be saved as:

  ```text
  result.mp4
  ```

- For **static image-driven inference**, the generated result will be saved as:

  ```text
  result.png
  ```

The driving videos, driving static images, and source identity images should be cropped before they can be used in our method. You can find example inputs in the `Inputs/` directory:

```bash
~/Deep_Expr_SID_Module/Inputs/
├── driving static images/
├── driving videos/
└── Identites/

or you can run:

```bash
python crop-video.py --inp some_youtube_video.mp4

git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
```

### Inference with Docker

If you are having trouble getting the demo to work because of library compatibility issues,
and you're running Linux, you might try running it inside a Docker container, which would
give you better control over the execution environment.

Requirements: Docker 19.03+ and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
installed and able to successfully run the `nvidia-docker` usage tests.

We'll first build the container.

```
docker build -t DeepExpr_SID .
```

And now that we have the container available locally, we can use it to run the demo.

```
docker run -it --rm --gpus all \
       -v $HOME/DeepExpr:/app DeepExpr_SID_Module \
       python3 Inference.py --config config/vox-256.yaml \
           --driving_video driving.mp4 \
           --Reference_image reference.png  \ 
           --source_image Identity.png  \ 
           --checkpoint sid-cpk.pth.tar \ 
           --result_video result.mp4 \
           --result_image result.png \
           --relative --adapt_scale
```
### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/dataset_name.yaml --device_ids 0,1,2,3
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
By default the batch size is tuned to run on 2 or 4 Titan-X gpu (apart from speed it does not make much difference). You can change the batch size in the train_params in corresponding ```.yaml``` file.

### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```reconstruction``` subfolder will be created in the checkpoint folder.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
Instructions for computing metrics from the paper can be found: https://github.com/AliaksandrSiarohin/pose-evaluation.

### Image animation

In order to animate videos run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode animate --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```animation``` subfolder will be created in the same folder as the checkpoint.
You can find the generated video there and its loss-less version in the ```png``` subfolder.
By default video from test set will be randomly paired, but you can specify the "source,driving" pairs in the corresponding ```.csv``` files. The path to this file should be specified in corresponding ```.yaml``` file in pairs_list setting.

There are 2 different ways of performing animation:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) <i>Animation using absolute coordinates:</i> the animation is performed using the absolute positions of the driving video and appearance of the source image.
In this way there are no specific requirements for the driving video and source appearance that is used.
However this usually leads to poor performance since irrelevant details such as shape is transferred.
Check animate parameters in ```taichi-256.yaml``` to enable this mode.

<img src="sup-mat/absolute-demo.gif" width="512"> 

2) <i>Animation using relative coordinates:</i> from the driving video we first estimate the relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for animation. This usually leads to better performance, however this requires
that the object in the first frame of the video and in the source image have the same pose

<img src="sup-mat/relative-demo.gif" width="512"> 


### Datasets

1) **Bair**. This dataset can be directly [downloaded](https://yadi.sk/d/Rr-fjn-PdmmqeA).

2) **Mgif**. This dataset can be directly [downloaded](https://yadi.sk/d/5VdqLARizmnj3Q).

3) **Fashion**. Follow the instruction on dataset downloading [from](https://vision.cs.ubc.ca/datasets/fashion/).

4) **Taichi**. Follow the instructions in [data/taichi-loading](data/taichi-loading/README.md) or instructions from https://github.com/AliaksandrSiarohin/video-preprocessing. 

5) **Nemo**. Please follow the [instructions](https://www.uva-nemo.org/) on how to download the dataset. Then the dataset should be preprocessed using scripts from https://github.com/AliaksandrSiarohin/video-preprocessing.
 
6) **VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.


### Training on your own dataset
1) Resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the later, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params.
