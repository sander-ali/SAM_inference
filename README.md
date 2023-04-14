# SAM_inference
The code provides python code for performing segmentation using Facebook's recent Segment Anything Model (SAM). You can also used anaconda prompt to perform command-line inference.

The code heavily borrows from original [SAM repository](https://github.com/facebookresearch/segment-anything). Follow the instructions to perform the segmentation

First install segment anything throup pip using following command:  
pip install git+https://github.com/facebookresearch/segment-anything.git  

Make sure to install relevant packages such as  

pip install opencv-python pycocotools matplotlib onnxruntime onnx  

Download pre-trained model checkpoint from https://github.com/facebookresearch/segment-anything#model-checkpoints  

The code is annotated so if you want to play with the parameters, you can modify them, accordingly.

Make sure you keep the model and test image in the same directory and run the following command  

python segment_SAM.py 

Sample Results.

![image](https://user-images.githubusercontent.com/26203136/232099230-d24155d2-05d5-4de2-bb7e-0b748486232a.png)

![image](https://user-images.githubusercontent.com/26203136/232100609-e4f3757e-83f1-45ee-b827-eaae3601f28b.png)


![Figure_1](https://user-images.githubusercontent.com/26203136/232100525-ea4b3afb-d1be-4b52-af85-495dab054080.png)
