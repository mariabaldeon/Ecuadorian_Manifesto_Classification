import pandas as pd
import numpy as np
import glob
from nltk.stem.snowball import SnowballStemmer
#!pip install simplemma
import simplemma

class DataManager:    
    """
    Reads manifesto from csv file and applies best preprocessing operations. 
    Return the csv file for fold/whole dataset after preprocessing
    path: path csv file
    num_fold: number of folds to create
    fold: number of the fold to return the dataset from 
    """
    def __init__(self, path, num_folds=None, fold=None):
        self.path=path
        if fold and num_folds is not None:
            assert fold<=num_folds, "number of fold should be less than or equal to the total number of folds"
        if num_folds is None or fold is None: 
            assert fold == num_folds, "num_folds and fold must be None"
        self.fold=fold
        self.num_folds=num_folds
    
    def read_csv(self):
        manifesto_dataset=pd.read_csv(self.path)
        return manifesto_dataset
    
    def clean_dataset(self, manifesto_dataset): 
        """
        Cleans the manifesto dataset by removing numbers, special characters, nan rows, and applying lowercase
        """
        manifesto_dataset["text"] = manifesto_dataset["text"].str.replace('\d+', '')
        manifesto_dataset["text"] = manifesto_dataset["text"].str.replace(r"\W"," ")
        manifesto_dataset = manifesto_dataset[manifesto_dataset["text"].notna()]
        manifesto_dataset["text"] = manifesto_dataset["text"].str.lower()
        return manifesto_dataset
    
    def apply_stemmer(self, manifesto_dataset): 
        stemmer = SnowballStemmer("spanish")
        manifesto_dataset['text'] = manifesto_dataset['text'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word
        return manifesto_dataset
    
    def lemmatize_text(self, text):
        return [simplemma.lemmatize(w, lang='es') for w in text]
    
    def apply_lemmatizer(self, manifesto_dataset):
        manifesto_dataset['text'] = manifesto_dataset.text.apply(self.lemmatize_text)
        return manifesto_dataset
    
    def preprocess_dataset(self, manifesto_dataset):    
        """
        Applies stemming and lemmatization to the dataset
        """
        manifesto_dataset['text'] = manifesto_dataset['text'].str.split()
        manifesto_dataset=self.apply_stemmer(manifesto_dataset)
        manifesto_dataset=self.apply_lemmatizer(manifesto_dataset)
        manifesto_dataset['text']= manifesto_dataset['text'].str.join(" ")
        return manifesto_dataset
        
    def generate_crossval(self, manifesto_dataset): 
        """
        Returns the training and validation set for fold
        """
        num_cases= len(manifesto_dataset)
        val_num=int(num_cases*(1/self.num_folds))
        all_cases= list(range(0, num_cases))
        # generates the list with observations for all folds
        val_list=np.random.RandomState(seed=0).choice(all_cases, size=num_cases,replace=False)

        start_fold=self.fold*val_num
        end_fold=start_fold+val_num
        val_fold=val_list[start_fold:end_fold]

        train_fold=[e for e in all_cases if e not in val_fold]
        train_fold=np.sort(train_fold)
        val_fold=np.sort(val_fold)
        val_dataset=manifesto_dataset.iloc[val_fold,]
        train_dataset=manifesto_dataset.iloc[train_fold,]


        return val_dataset, train_dataset
    
    def return_dataset(self): 
        """
        Reads the cvs file, cleans, preprocess and returns dataset
        """
        manifesto_dataset=self.read_csv()
        manifesto_dataset=self.clean_dataset(manifesto_dataset)
        manifesto_dataset=self.preprocess_dataset(manifesto_dataset)
        
        if self.fold is not None and self.num_folds is not None:
            val_dataset, train_dataset=self.generate_crossval(manifesto_dataset)
            return val_dataset, train_dataset
        else: 
            return manifesto_dataset


