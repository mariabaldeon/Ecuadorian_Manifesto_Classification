# Ecuadorian_Manifesto_Classification
Content analysis of political manifestos is necessary to understand a party´s strategies, policy positions, and proposed actions. However, manual coding is time-consuming, labor-intensive, and can result in biases. Machine learning models, particularly Transformer networks, have become essential tools for automating this task. In Ecuador, all candidate parties are required to submit their manifestos to exist. Nonetheless, annotated Ecuadorian corpora are scarce, which poses a challenge for developing automatic labelling models. In this work, we develop a Transformer network for Ecuadorian manifesto classification with a cross-domain training strategy. Using the database from the Comparative Manifesto Project, we implement a fractional factorial experimental design to determine which Spanish-written manifestos form the best training set for Ecuadorian manifesto labelling. Furthermore, we statistically analyze which Transformer architecture and preprocessing operations improve the model´s accuracy. The results indicate that creating a training set with manifestos from Spain and Uruguay, along with implementing stemming and lemmatization preprocessing operations, produces the highest classification accuracy. In addition, we find that the DistilBERT and RoBERTa transformer networks perform statistically similarly and consistently well in manifesto classification. Without using any Ecuadorian text during training, DistilBERT and RoBERTa achieve a 60.05% and 57.64% accuracy, respectively, in Ecuadorian manifesto classification. Finally, we investigate the effect of the training set´s composition on performance. The experiments demonstrate that training DistilBERT solely with Ecuadorian manifestos achieves the highest accuracy and F1-score. Moreover, if an Ecuadorian dataset is not available, training the model with datasets from Uruguay and Spain obtains a competitive performance.

# Requirements
* Python 3.7
* Numpy 1.23.5
* argsparse 1.4.0
* pandas 1.4.2
* torch 2.0.0
* smaberta 0.0.2
* glob 0.6 
* nltk 3.7
* simplemma 8.9.1

# Dataset

In this work we use two datasets. The model can be trained with the manifestos from Uruguay and Spain, taken from the Manifesto Comparative Project (CMP). The link to this great database can be found [here]([https://promise12.grand-challenge.org/](https://manifesto-project.wzb.eu/))


# Training 
To carry out the training run: 
```
nohup python3 main.py --task train & 
```
The code assumes the training dataset is located in the directory Datasets/Train. If it is in another directory, specify the path using the --dataTrain argument. The training will be performed in the five folds. For each fold two folders named 2d_training_logs and 3d_training_logs will appear. Inside the folders, the training logs and weights wil be saved for the 2d and 3d CNNs. 

# Evaluation
To carry out the evaluation run: 
```
nohup python3 main.py --task evaluate & 
```
The code assumes the testing dataset is located in the directory Datasets/Test. If it is in another directory, specify the path using the --dataTest argument.The code will evaluate the 2D CNN ensemble, 3D CNN ensemble, and PPZSeg-Net. Evaluation metrics will be saved in a .csv file in a folder named Evaluation_metrics. The evaluation  metrics considered are the Dice similarity coefficient (DS) and Haussdorff distance (HD). These metrices will be calculated for the 2D CNN ensemble, 3D CNN ensemble, and PPZSeg-Net. The segmentation results will be saved in the folders Results_2D.mat, Results_3D.mat, and Results_PPZSegNet.mat for the 2D CNN ensemble, 3D CNN ensemble, and PPZSeg-Net, respectively. The trained weights from this work should be located in the directory Networks/weights (link to weights: [link](https://drive.google.com/drive/folders/1wW_aBqUAe9g6eQCN9de1ILyDLg0dGPb0?usp=share_link) ). These weights will  be used for evaluation. If you wan to use other weights, locate them in this folder with the corresponding name k{fold}_{network}D.hdf5, where fold refers to the fold trained on and network to the type of network 2D or 3D.  

