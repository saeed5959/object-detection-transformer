# An End-to-End Object Detection with Vision Transformation [Link](article.pdf)

Official implementation of "Place of Attention Matters!" in pytorch : Article is [Here](article.pdf) .
<br/>

This work has been inspired by [vision transformer](https://arxiv.org/abs/2010.11929) and [Detr](https://arxiv.org/abs/2005.12872).
<br/>

---
GOOD NEWS!!<br/>
The pretrained model for 56 epoch is in [Drive](https://drive.google.com/file/d/1fNFAW1WeSJEpe4U-tlyZadDuo2jpqRay/view?usp=sharing)

### Model Architecture
<img src="/images/result.png" width="900" height="300" border="20" title="model">


---
Preprocess

    1-download coco dataset("annotations", "train2017") and unzip and put in dataset folder
    2-python3 preprocess.py
    3-it will make dataset_file_out.txt

Train or Fine-tune
    
    python3 train.py --train_file_path ./dataset/dataset_file_out.txt --model_path ./m.pth --pretrained ./x_56.pth

Inference

    python3 inference.py --img_path ./dataset/train2017/000000580197.jpg --model_path ./x_56.pth --out_path ./out.jpg
    
---
### Model Architecture
<img src="/images/model.png" width="900" height="300" border="20" title="model">

---

<br/>

# Place of Attention Matters! 
<br/>

An End-to-End Object Detection with Vision Transformation!


In the object detection task, the purpose is to find the class of object and a bounding box
around it. Most works have focused on just finding the class of object without considering
bounding box features properly. We present a new method that focuses on relationships
between patches of the image as a feature for bounding box detector.
Also, we combine convolutional neural network as a local feature detector and
Transformer network as a long-distance feature detector. We were also inspired by the
method that has been used in Transformer as a relationship between patches in the image.
Our implementation can perform in real-time and improve the accuracy of previous works.

---
<br/>

### Sample output
<img src="/images/sample.jpg" width="900" height="300" border="20" title="sample">

---

<br/>

### LOSS
<img src="/images/loss_box.png" width="900" height="200" border="10" title="loss_box">

<img src="/images/loss_class.png" width="900" height="200" border="10" title="loss_class">

<img src="/images/loss_obj.png" width="900" height="200" border="10" title="loss_obj">

<img src="/images/loss_poa.png" width="900" height="200" border="10" title="loss_poa">

---