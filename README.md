# Filtered-CoPhy : Unsupervised Learning of Counterfactual Physics in Pixel Space
 
This repository contains the code associated to the paper <a href="https://filteredcophy.github.io/#:~:text=Anonymous%20authors">Filtered-CoPhy : Unsupervised Learning of Counterfactual Physics in Pixel Space</a>

Link: <a href="https://filteredcophy.github.io/"> Project page </a>

# Abstract
Causal discovery is at the core of human cognition. It enables us to reason about the environment and make counterfactual predictions about unseen scenarios, that can vastly differ from our previous experiences. We consider the task of causal discovery from videos in an end-to-end fashion without supervision on the ground-truth graph structure. In particular, our goal is to discover the structural dependencies among environmental and object variables: inferring the type and strength of interactions that have a causal effect on the behavior of the dynamical system. Our model consists of (a) a perception module that extracts a semantically meaningful and temporally consistent keypoint representation from images, (b) an inference module for determining the graph distribution induced by the detected keypoints, and (c) a dynamics module that can predict the future by conditioning on the inferred graph. We assume access to different configurations and environmental conditions, i.e., data from unknown interventions on the underlying system; thus, we can hope to discover the correct underlying causal graph without explicit interventions. We evaluate our method in a planar multi-body interaction environment and scenarios involving fabrics of different shapes like shirts and pants. Experiments demonstrate that our model can correctly identify the interactions from a short sequence of images and make long-term future predictions. The causal structure assumed by the model also allows it to make counterfactual predictions and extrapolate to systems of unseen interaction graphs or graphs of various sizes.



# Dataset
You can download the dataset on this <a href="www.google.com"> link (comming soon)</a>. It contains 112x112 videos as well as ground truth states, confounders and colors for each experiments. Below are some explanations for the files associated to the dataset :

    ```
    CoPhy_112/<TASK NAME>/<ID>
    ├── COLORS.txt # color of each block from bottom to top
    ├── ab # the observed sequence (A,B)
    │   ├── rgb.mp4 # the RGB sequence of 6 seconds long at fps=25 (3 seconds for collisionCF)
    │   └── states.npy # the sequence of state information of each block (3D position, 4D quaternion and their associated velocities)
    ├── cd # sequence including the modified initial state C and the outcome D
    │   ├── rgb.mp4
    │   └── states.npy
    ├── confounders.npy # the confounder information for each block
    └── do_op.txt # a description of the do-operation
    ```
    
    We also released the generation scripts. You can generates each task using the corresponding script in ```data_generation```.
