# Filtered-CoPhy : Unsupervised Learning of Counterfactual Physics in Pixel Space
 
This repository contains the code associated to the paper <a href="https://filteredcophy.github.io/#:~:text=Anonymous%20authors">Filtered-CoPhy : Unsupervised Learning of Counterfactual Physics in Pixel Space</a>

Link: <a href="https://filteredcophy.github.io/"> Project page </a>

# Abstract
Causal discovery is at the core of human cognition. It enables us to reason about the environment and make counterfactual predictions about unseen scenarios, that can vastly differ from our previous experiences. We consider the task of causal discovery from videos in an end-to-end fashion without supervision on the ground-truth graph structure. In particular, our goal is to discover the structural dependencies among environmental and object variables: inferring the type and strength of interactions that have a causal effect on the behavior of the dynamical system. Our model consists of (a) a perception module that extracts a semantically meaningful and temporally consistent keypoint representation from images, (b) an inference module for determining the graph distribution induced by the detected keypoints, and (c) a dynamics module that can predict the future by conditioning on the inferred graph. We assume access to different configurations and environmental conditions, i.e., data from unknown interventions on the underlying system; thus, we can hope to discover the correct underlying causal graph without explicit interventions. We evaluate our method in a planar multi-body interaction environment and scenarios involving fabrics of different shapes like shirts and pants. Experiments demonstrate that our model can correctly identify the interactions from a short sequence of images and make long-term future predictions. The causal structure assumed by the model also allows it to make counterfactual predictions and extrapolate to systems of unseen interaction graphs or graphs of various sizes.



# Dataset
You can download the dataset on this <a href="https://zenodo.org/record/5906002#.YfFG1PXMKCe"> link</a>. It contains 112x112 videos as well as ground truth states, confounders and colors for each experiments. Below are some explanations for the files associated to the dataset :

## Composition
You will find train/validation/test splits in the Datasets directory as text files containing the list of ids for each task. The experiments in FilteredCoPhy are stored in separate files. Each experiment has the same structure :

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
   
   ## Generation
   We also released the generation scripts. You can generates each task using the corresponding script in ```data_generation```. For example, the following command will generate 1000 instances of BlocktowerCF with 4 cubes and save the result in OUTPUT_DIRECTORY :
   
``` python3 generate_blocktower.py --dir_out OUTPUT_DIRECTORY --seed O --n_cubes 4 --n_examples 1000    ```
       
 # Training / Evaluation
 ## De-Rendering module
 The de-rendering module can be trained using the corresponding script ```train_derendering.py```. Below is a description of the relevant parameters :
 
```
     --epoch          : Number of epoch for training. Evaluate the trained model if set to 0
     --lr             : Learning rate
     --n_keypoints    : Number of keypoints
     --n_coefficients : Number of coefficients
     --dataset        : 'blocktower', 'balls' or 'collision'
     --name           : You can specify the name of the file in which the weights will be saved.
     --video_path     : Path to the videos
     --mode           : Default is 'fixed'. If set to 'learned', the de-rendering module will also learn the filter bank. Otherwise, it uses fixed dilatation filters.
```
Note that the script using the dataloaders in ```Dataloaders```. You will have to specify the location of the dataset in the code. We strongly recommend to pre-compute the keypoints for each experiments in the dataset using the following command:
``` python3 generate_keypoints.py --dataset 'blocktower' --n_coefficients 4 --n_keypoints 4 --model_weight_path WEIGHTS_FILE --output_path OUTPUT_PATH  ```
## CoDy
CoDy can be trained using ```train_cody.py``` with the following options :

```
    --epoch : Number of epochs
    --lr             : Learning rate
    --dataset        : 'blocktower', 'balls' or 'collision'
    --keypoints_model : Used to choose over different ensemble of keypoints. See dataloader for explicit usage.
    --n_keypoints    : Number of keypoints
```

The script uses keypoints dataloaders in  ```Dataloaders```. You will have to update the location in the code.

## Evaluation
We also realised our evaluation script. It generates a dictionnary containing metrics for each experiments in the training set.

``` python3 evaluate_pipeline.py --dataset 'blocktower' --derendering_path PATH_TO_DERENDERING_WEIGHTS --cody_path PATH_TO_CODY_WEIGHTS --output metrics.pkl```

# Citation
```@inproceedings{janny2022FilteredCoPhy,
    title = "Filtered-CoPhy: Unsupervised Learning of Counterfactual Physics in Pixel Space",
    author = {Janny, Steeven  and
              Baradel, Fabien  and
              Neverova, Natalia  and
              Nadri, Madiha  and
              Mori, Greg  and
              Wolf, Christian},
    booktitle = "International Conference on Learning Representations (ICLR)",
    year = "2022"}
```
