
Imagique app is an open source GUI application that is used for appliying augmentations to images which are used for train a model. There are different augmentation methods such as flip, brigthness adjustment, contrast adjustments and so on. 

By usign range bars, you can select an interval that augmentation value is going to be selected on it. For example, if you select 0 and 0.1 for shear X, program will select 
a random value between 0 and 0.1 and shear operation will be applied with the ratio of this value in X direction from the center of image. 

Every augmentation has a possibility value that can be adjusted with slide bars, these are the probabilities of augmentation. For example, if you choice 35% for flip operation,
flip operation is applied to image or images with 35% probability.

By using these range bars and slide bars for selection of random augmentation values and probabilities, it can be possible to produce new different images with it. Some
metadata of original image and applied augmentations with random values can be seen in the right down side of the window.

Imagique application can allows to apply the same augmentation to bounding boxes of images. However, a csv or excel file that contains image names column, region attributes 
column which is label name or index, and bounding boxes column that contains bounding boxes as center x, center y, width and height value of bounding boxes as absolute value 
(between 0 and 1) are needed to apply augmentations to bounding boxes. 

To better understand the what type of data Imagique app wonders, there are a folder named example data contaions example images and their bounding boxes csv file.

You can apply augmentations to one image and see new augmented image in right side of the window, and also compare with original image by using buttons on top of the image 
or you can apply augmentation to all images by using start button after select save folders and other desired paramaters. 
