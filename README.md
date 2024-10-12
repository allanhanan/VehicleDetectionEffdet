## Smart AI based Traffic Management System (WIP)
This project improves on the already existing preset traffic light timers by dynamically predicting time based on the no of vehicles available.

Vehicle data is obtained from the already installed CCTV monitoring systems. A frame is captured and is sent to the EfficientDet model which returns the no of vehicle based on each class and the vehicle density.

EfficientDet improves on EfficientNet by providing compound scaing and BiFPN, which makes it significantly more efficient and allows to run on low power hardware or on the cloud.

A Regression Tree is used to predict the time from the data obtained from EfficientDet.

Further, LSTM is used to refine the time predictions by using Historical Data which tailors the model to each Traffic Signal.
