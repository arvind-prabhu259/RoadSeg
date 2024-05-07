# RoadSeg
Semantic Segmentation of dashcam footage.
The model is based on U-Net architecture and is trained on the CamVid dataset.
![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/760ecc4d-f0e5-4e13-be82-41ba4e6cc477)

Training loss for the first 20 epochs of training


![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/2423bb05-2fdf-462f-b1e8-8e42693de2da)

(U-Net architecture)
## Training and Validation:

Due to limitations in GPU access, I first trained the model and saved it after every epoch. To perform validation, I loaded the saved model at each epoch at performed validation. The results are as plotted.

### Loss function:

Loss functions are used to provide a measure of how "wrong" a model's prediction is. Here, I used a combination of two different loss functions: Dice Loss and Cross Entropy Loss. Dice Loss is a type of Intersection over Union (IOU) loss function. It encourages the model to create more accurate segmentation masks, i.e., segmentation masks that more closely follow the shapes of objects in the image. Cross Entropy Loss, on the other hand, encourages the model to make more accurate predictions (more correct identification). Using a combinationn of these 2 loss functions, I have trained the model to create both accurate and well-defined seegmentation masks for dashcam footage.

## Results:
After training for 20 epochs:


![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/615af034-6501-4a04-96aa-d2a2b4f8f9ad)

![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/621b9a0d-6af6-42b1-bbce-1ab4d272d7f6)


Training loss for 200 epochs:

![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/d2931852-e67b-4cde-be99-b534026e3f49)

Training loss for 200 epochs using only cross entropy loss:

![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/c3786bcd-35db-41f1-ad86-d73587307f59)


We can see that the model is capable of distinguishing roughly between the sky and the ground. However, it struggles with identifying different types of objects(Trees, cars, etc).
