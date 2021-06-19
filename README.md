# Co-training-for-NIH-Pancreas-Segmentation
<3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training>

![image](https://user-images.githubusercontent.com/37214771/122636822-d0aa2380-d11d-11eb-995b-6d104d118453.png)


not official code

通过train_BL.py进行不同view的训练，并保存模型；

通过co-training_v4.py进行co-training；

通过inference.py进行滑窗测试（用于单视角）；

通过inference_v2.py进行滑窗测试（用于多视角）；

通过valid.py进行center_crop的validation。
