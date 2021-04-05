# Meet Heli, the fully automated captcha breaker

![alt text](https://github.com/benji2264/heliade/blob/main/heli_demo.gif "Heli Demo")
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Among bot detection technologies, Google's image captcha service, called reCAPTCHAv2, is the most widely used today. 
Its goal : propose a challenge that is easy to solve for a human being but almost impossible for a computer. 
But are these challenges really difficult enough for a robot? 
State of the art deep learning techniques have proven to be incredibly powerful for image recognition, and today, we are going to use some of them to build a proof-of-concept reCAPTCHA solver : **Heli**.
 
**NB : A lot of the ideas presented here are derived from this paper by Hossen et al. :**
 
[An Object Detection based solver for Google's reCAPTCHA v2](https://www.usenix.org/system/files/raid20-hossen.pdf)
 
## Captchas as an image classification problem
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 - Pretrained Models and Adversarial Examples

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The simplest way to look at captchas is as a classification problem. 
You are given a label (e.g. bus), and you have to select all the bus pictures among the proposed images. 
Simple, right ? ConvNets are amazingly good at image classification. 
But there is a trick, captcha images are not ordinary images.
To see this, we can use a ResNet50 pre-trained on ImageNet, and ask it to classify the 9 images of a given challenge :
 
<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/pretrained_resnet50_preds.jpeg" alt="drawing" width="700"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first thing you notice, apart from the fact that the model performs really poorly, is that in 3 images out of 9, the model recognizes a skunk. 
And it is quite confident too (up to 63.71%). 
If we try again several times on other images, we find that some classes appear much more than others, especially : skunk, assault rifle, missile, ...
Then you also notice that most of the images are very blurry, noisy and distorted.
But when they are not, this ResNet actually does quite well, and it manages to recognize a minibus in the top right image.
These two observations often indicate the presence of **adversarial examples** : images that have been intentionally modified to fool classification algorithms.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A very good example of these adversarial images is this :
<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/adversarial_bus.PNG" alt="drawing" width="700"/>

As you can see, the first image (on the left) which is not noisy, is very easily recognized by the ResNet50 as a school bus, while the other two, apparently adversarial examples, are classified as skunk and sunglass. 

*Note : Keras allows you to use a pretrained DenseNet201 instead of ResNet50 with a single line of code. I tested both on about a hundred images and got the same results overall.*

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 - Adversarial Training and new model from scratch
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Obviously, we will not be able to use a pre-trained model to solve this problem. 
To overcome adversarial examples, we can use a technique called **adversarial training**, which is a very simple idea : train the model on adversarial examples. 
So I downloaded the 11,000+ captcha images available on deathlyface's GitHub right [here](https://github.com/deathlyface/recaptcha-dataset),
then I downloaded and labeled another 9,000+ images myself. The complete dataset is available [here](https://www.kaggle.com/benjaminmissaoui/recaptcha-data) on kaggle.
I implemented a simple ResNet50 in Keras and trained it on this new dataset (with lr = 0.0001, batch size = 128, Data Augmentation, Adam Optimizer, Batch Normalization): 
<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/training_resnet50.png" alt="drawing" width="700"/>

Even with data augmentation, there's still severe **overfitting** here. 
The model achieves **76%** accuracy on the test set (note that I didn't use a dev set to keep as many examples as possible for the training set).

Even if the accuracy is not that high, our model still performs very well (and much better than the pretrained model). For the sake of comparison, here are the predictions for the same images as before :

<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/custom_resnet50_preds.jpeg" alt="drawing" width="700"/>

Here are a few things I tried to improve performance or reduce overfitting but none of them showed significant effects :
* Mish Activation function
* Gradient clipping
* Dropout

**Areas of Improvement :**

* Collect and label more data (captcha images or images found on Google)
* Try an architecture that has shown better performance on smaller datasets (e.g. DenseNet)
* Try other types of data augmentation (e. g. with GANs)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3 - EfficientNet and Transfer Learning
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this last section, I tried to combine the best of both worlds: the high performances of pre-trained networks and the robustness of adversarial training. 
I chose here a state-of-the-art model for image classification, **EfficientNetB0**, pre-trained on ImageNet (to learn a few lower level features), and I fine-tuned on our own dataset. This is the idea of **transfer learning**. 
The implementation in Keras is, again, very straightforward. But this turned out to be a failure, the features in the ImageNet dataset and in our own dataset are apparently too different (maybe because of the adversarial examples) :
<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/training_efficientnetb0.png" alt="drawing" width="700"/>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4 - Problems and limitations of the "image classification" approach

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Unfortunately, as good as our model is, there are some obvious limitations to the image classification approach. 
The main issue is that the network can only predict one class per image, which is a severe problem with images containing multiple objects :

<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/multiple_class_preds.jpeg" alt="drawing" width="1150"/>
 
Here a human operator would have clicked on the middle image whether the challenge was to recognize cars, crosswalks or traffic lights.
But Heli would only have clicked if the challenge was to recognize crosswalks (Top 1 prediction).
One way to partially address this issue is to allow Heli to click on the image if it is confident enough (probability above a certain threshold, e.g. 40%).
This way, if the network output for a given image is Bridge (45%), Crosswalk (55%), Heli would click on that image in both cases.
Unfortunately, this causes more problems than it solves, especially since the network sometimes has trouble telling cars and buses apart.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The other problem is that sometimes reCAPTCHA presents to the user large images divided into a 4x4 grid. 
In this case, since we process each cell independently, the network does not see the whole image, and has trouble guessing what is in there:

<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/image_classification/4x4_preds.PNG" alt="drawing" width="700"/>

## Captchas as an object detection problem

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fortunately, both of these problems can be solved using another approach: an object detection solver. 
Object detection algorithms have made huge progress in recent years, and the one we are going to use here is YOLOv3, which is very well explained in this post by Ethan Yanjia Li if you need a refresher:

[Dive Really Deep into YOLO v3: A Beginner’s Guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The idea here is that we will give the entire image to YOLO and let it predict all the objects in it (with their positions),
and then we'll click on the cells that contain at least part of the object. I followed this tutorial to implement YOLOv3 in pytorch :

[How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

Once this is done, we can now check the performance of YOLO on the 4x4 grids :

<img src="https://github.com/benji2264/heli_captcha_solver/blob/main/object_detection/yolov3_4x4preds.png" alt="drawing" width="650"/>

All we have to do now (which I have not had time to implement yet) is to let Heli click on every cell that contains at least part of the requested object. An object-detection solver therefore solves both the 4x4 grid problem and the problem of multiple objects in the same image.

**TODO :**
Add the predictions of YOLO in the browser automation module (notebook 02 in the image classification folder).

**Areas of Improvement :**

* Collect and label custom dataset, in order for Heli to detect objects missing in the COCO dataset (like bridges)
* Try an even better object detection algorithmn (scaled-YOLOv4, EfficientDet)
