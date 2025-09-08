# Smart AI based Traffic Management System (WIP)
This project improves on the already existing preset traffic light timers by dynamically predicting time based on the no of vehicles available.

https://allanhanan.pages.dev/pages/apps/trafficdetection

Vehicle data is obtained from the already installed CCTV monitoring systems. A frame is captured and is sent to the EfficientDet model which returns the no of vehicle based on each class and the vehicle density.

EfficientDet improves on EfficientNet by providing compound scaing and BiFPN, which makes it significantly more efficient and allows to run on low power hardware or on the cloud.

A Regression Tree is used to predict the time from the data obtained from EfficientDet.

Further, LSTM is used to refine the time predictions by using Historical Data which tailors the model to each Traffic Signal.

## Workflow for Traffic Signal Timing

1. **Green Light Duration Prediction**:  
   Predict the optimal duration for the green light \( T_g \) based on self-assessed traffic conditions.

2. **Yellow Light Duration Prediction**:  
   Upon activation of the yellow light, predict its duration \( T_y \) 

3. **Next Signal Red Light Assignment**:  
   Set the duration for the next signal's red light \( T_{r1} \) as:
   \[
   T_{r1} = T_g + T_y
   \]

4. **Next Signal Green Light Assignment**:  
   For the subsequent signal, determine the green light duration \( T_{g2} \) as:
   \[
   T_{g2} = T_{g1} + T_g
   \]
   where \( T_{g1} \) is the previous signal's green light duration.

5. **Cumulative Timing for Further Signals**:  
   For each additional signal in the loop, assign the green light duration \( T_{gn} \) as the cumulative sum:
   \[
   T_{gn} = T_{g(n-1)} + T_g
   \]
   This ensures that each signal adjusts based on the predicted timings of its predecessors.
