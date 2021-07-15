# Steering_Angle_Prediction
This project is a scale down version of self driving car. 
Here we are collecting a video from the front dash camera and training a deep learning based model. 
This will predict angle at which the steering will rotate in order to drive a car in regular traffic and weather conditions.


https://user-images.githubusercontent.com/67441175/125787879-09f1fcd6-e81c-4eab-83c4-e73a71e8e393.mp4




## Dataset
The link to the dataset is here. I donot remember the exact link so uploaded on my drive. Feel free to request it. 
https://drive.google.com/drive/u/1/folders/1rvGwBpA5-ILRjHPL_bnhc979i3PB8pyp
Model was trained on 80k images converted from a video and gave great results when tested.

## Model insights
The steering angle was in degrees and it was converted into radians first.
The model was designed with CNN, Flatten, Dense and Dropout Layers.
The activation used in inner layers was relu
The activation used in output layer was linear.
Dropout layers, batch normalization and kernel regularizer were added for regularization and to prevent overfitting.
The model predicted the value of steering angle in radians so later it was converted back to degrees.

## Testing
SDC Simulation 2.py containse code for testing the model and its performance. Video shows the glimpse of the model.

## Weights
Weights were also added. All the weights which were showing good results are uploaded here.

## Video link
https://www.linkedin.com/posts/kartikey-vishnu-12649846_deeplearning-selfdrivingcar-aiml-activity-6702318801695866880-WfD0
Results of how well the car responed to traffic is clearly visible.
