# Ecuadorian_Manifesto_Classification
<p align="justify"> Content analysis of political manifestos is necessary to understand a party´s strategies, policy positions, and proposed actions. However, manual coding is time-consuming, labor-intensive, and can result in biases. Machine learning models, particularly Transformer networks, have become essential tools for automating this task. In Ecuador, all candidate parties are required to submit their manifestos to exist. Nonetheless, annotated Ecuadorian corpora are scarce, which poses a challenge for developing automatic labelling models. In this work, we develop a Transformer network for Ecuadorian manifesto classification with a cross-domain training strategy. Using the database from the Comparative Manifesto Project, we implement a fractional factorial experimental design to determine which Spanish-written manifestos form the best training set for Ecuadorian manifesto labelling. Furthermore, we statistically analyze which Transformer architecture and preprocessing operations improve the model´s accuracy. The results indicate that creating a training set with manifestos from Spain and Uruguay, along with implementing stemming and lemmatization preprocessing operations, produces the highest classification accuracy. In addition, we find that the DistilBERT and RoBERTa transformer networks perform statistically similarly and consistently well in manifesto classification. Without using any Ecuadorian text during training, DistilBERT and RoBERTa achieve a 60.05% and 57.64% accuracy, respectively, in Ecuadorian manifesto classification. Finally, we investigate the effect of the training set´s composition on performance. The experiments demonstrate that training DistilBERT solely with Ecuadorian manifestos achieves the highest accuracy and F1-score. Moreover, if an Ecuadorian dataset is not available, training the model with datasets from Uruguay and Spain obtains a competitive performance.</p>

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

In this work we used two datasets. First, we make available a new Ecuadorian manifesto dataset, located in *Datasets/Database-Ecuador.csv)*. The Ecuadorian corpus was obtained from the official websites of the two 2021 Ecuadorian presidential candidates, Andrés Arauz and Guillermo Lasso. The documents were pre-processed by removing all images and tables. The annotation process was divided into two tasks, following the Manifesto Coding Instructions 5th edition. First, the text was unitized into sentences or quasi-sentences, where each unit conveyed a similar message. Then, a political expert and a political science student categorized the quasi-sentences into one of the seven domains explained in section 3.1 of our paper. Only quasi-sentences where the two annotators agreed on the same category were kept from training, resulting in a dataset with 1809 sentences and quasi-sentences. 

The second dataset comprised of all Spanish-written manifestos from the Manifesto Comparative Project (CMP). The link to this great database can be found [here](https://manifesto-project.wzb.eu/). We have included in our repository the manifestos from Uruguay and Spain, as it proved to be the best combination for Ecuadorian manifesto classification. However, if you use these two datasets for your own work, dont forget to cite the CMP. 

# Networks 
We test the performance between DistilBERT and RoBERTa. You can utilized either networks through the flag: 
```
--network roberta
--network distilbert
```
This a required argument.
# Training 
This code allows training and evaluating on the Ecuadorian dataset using a fivefold approach by running the following code:  
```
nohup python3 main.py --task train_Ecuador --network roberta & 

nohup python3 main.py --task train_Ecuador --network distilbert & 
```
or training on the dataset composed of manifestos from Uruguay and Spain and evaluated on the Ecuadorian dataset with a fivefold approach through the following code: 
```
nohup python3 main.py --task train_Uruguay_Spain --network roberta & 

nohup python3 main.py --task train_Uruguay_Spain --network distilbert & 
```
