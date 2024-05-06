# RoadSeg
Semantic Segmentation of dashcam footage.
The model is based on U-Net architecture and is trained on the CamVid dataset.
![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/760ecc4d-f0e5-4e13-be82-41ba4e6cc477)

Training loss for the first 20 epochs of training


![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/2423bb05-2fdf-462f-b1e8-8e42693de2da)

(U-Net architecture)

## Results:
After training for 20 epochs:


![image](https://github.com/arvind-prabhu259/RoadSeg/assets/94371314/615af034-6501-4a04-96aa-d2a2b4f8f9ad)


We can see that the model is capable of distinguishing roughly between the sky and the ground. However, it struggles with identifying different types of objects(Trees, cars, etc).
