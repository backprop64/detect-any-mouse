# Detect Any Mouse
A system for single/multi mouse tracking in complex enviornemnts.

## Notebooks + Video Tutorial
### Zero-Shot Tracking Notebook [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qiTIqScLwH7kfp_o5Z1t7UBHNykMptE9?usp=sharing)

**Description:**

In this notebook, you can track mice in your own videos using our DAM system. To use it, you need to specify the maximum number of mice visible in any frame of the video. The inputs for this notebook are your videos, and the output is tracking data.

**Inputs:**
- Your videos

**Outputs:**
- Tracking data

---

### Create Your Own Dataset Notebook [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tVG6HvkxVKCKRzauVEhld3Jp7WZM8QK0?usp=sharing)

**Description:**

In this notebook, you can create a custom dataset to fine-tune a mouse detector. The entire process can be done in Google Colab using a graphical user interface (GUI). You can sample a folder of images from a video or a directory of videos. Additionally, you can add your own images or delete images from this folder to prepare for annotation in Colab. You can then annotate these images to create a dataset suitable for fine-tuning a mouse detector. Remarkably, you can achieve impressive results with as few as 20 images, which can be annotated in just a few minutes.

**Inputs:**
- Video or folder of videos
- Optionally, your own images

**Outputs:**
- Annotated dataset for detector fine-tuning

---

### Fine-Tune A Model Notebook [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dBdoQYvQSOWLwfVQ4k9Fs250RwDQQdcV?usp=sharing) 
**Description:**

In this notebook, you can fine-tune a mouse detector using the dataset you created in the previous notebook. Fine-tuning the detector will enhance tracking performance in your experimental setup. The input for this notebook is a dataset, and the output is a new detector along with its configuration, which you can use to track mice in videos.

**Inputs:**
- Annotated dataset

**Outputs:**
- New detector and configuration for tracking
