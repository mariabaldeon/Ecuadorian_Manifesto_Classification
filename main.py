import argparse
import pandas as pd
from DataManager import DataManager
from RunModel import RunModel

parser = argparse.ArgumentParser(prog="Ecuadorian_Manifesto_Classification")
parser.add_argument('--task', choices=['train_Ecuador', 'train_Uruguay_Spain'], required=True, help='task to do: train model with manifestos from Ecuador or train with datasets from Uruguay and Spain' )
parser.add_argument('--network', choices=['roberta', 'distilbert'], required=True, help='Transformer network to train' )
parser.add_argument('--dataEcuador', type=str, default='./Datasets/Database-Ecuador.csv', help='path to the csv with manifestos from Ecuador')
parser.add_argument('--dataUruguay', type=str, default='./Datasets/Database-Uruguay.csv', help='path to the csv with manifestos from Uruguay')
parser.add_argument('--dataSpain', type=str, default='./Datasets/Database-Spain.csv', help='path to the csv with manifestos from Spain')
parser.add_argument('--folds', type=int, default=5, help='Number of fold to run the experiments')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train Transformer model')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate to train Transformer model')

args = parser.parse_args()

if __name__ =='__main__':
    if args.task =='train_Ecuador': 
        for k in range(args.folds):
            data_processing=DataManager(args.dataEcuador, args.folds, k)
            valid_data, train_data=data_processing.return_dataset()
            model=RunModel(args.network, train_data, args.lr, args.num_epochs)
            model.run_model(valid_data)
    
    elif args.task =='train_Uruguay_Spain':
        uru_processing=DataManager(args.dataUruguay)
        uru_dataset=uru_processing.return_dataset()
        sp_processing=DataManager(args.dataSpain)
        sp_dataset=sp_processing.return_dataset()
        train_data=pd.concat([uru_dataset, sp_dataset], axis=0)
        print("length Uru+Spain", len(train_data))

        model=RunModel(args.network, train_data, args.lr, args.num_epochs)
        model.set_model()
        model.train_model()
        
        for k in range(args.folds):
            valid_processing=DataManager(args.dataEcuador, args.folds, k)
            valid_data, _ =valid_processing.return_dataset()
            model.evaluate_model(valid_data)
        
    


