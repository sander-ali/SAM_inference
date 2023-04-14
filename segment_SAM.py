#Step by Step Guide

#First install segment anything throup pip using following command

#pip install git+https://github.com/facebookresearch/segment-anything.git

#Make sure to install relevant packages such as 

# pip install opencv-python pycocotools matplotlib onnxruntime onnx

#Download pre-trained model checkpoint from https://github.com/facebookresearch/segment-anything#model-checkpoints

#import necessary packages
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#The function is adopted from original repository
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
#The function is adopted from original repository
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

#Depending on the checkpoint you have downloaded 
mk = "vit_h" #vit_h or vit_b or vit_l
#Depending on the checkpoint you have downloaded 
mdl = "sam_vit_h_4b8939.pth" #replace the name of the model here

sam = sam_model_registry[mk](checkpoint=mdl)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, #you can change the points per side
    pred_iou_thresh=0.9, #vary the threshold
    stability_score_thresh=0.92, #you can very the threshold
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,)  # Requires open-cv to run post-processing)

# load the original input image and display it to our screen
image = cv2.imread("sample.png") #input image path
#large images increase the inference time, therefore you can resize the image to reduce the time
#comment the below lines if you do not want to resize the image
hh = image.shape[0]
ww = image.shape[1]
image = cv2.resize(image,(int(ww/2),int(hh/2)))
# cv2.waitKey(0)

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

#visualize each mask individually
for i, mask_data in enumerate(masks):
    cv2.imshow("Original", image)
    mask = mask_data["segmentation"]
    iou = mask_data["predicted_iou"]
    score = mask_data["stability_score"]
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca()) 
    plt.title(f"Mask {i+1}, Score: {score:.3f} , IoU: {iou:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()      