


import numpy as np
import torch
from smaberta import TransformerModel
from sklearn import metrics
from sklearn.metrics import confusion_matrix
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class RunModel: 
    def __init__(self, network, train_data, lr = 5e-5, epochs=25, output_dir='./saved_model/'):
        self.network= network
        self.train_data=train_data
        self.lr=lr
        self.epochs=epochs
        self.output_dir= output_dir
        
    def set_model(self):
        if self.network=="roberta":
            name='roberta'
            encoding='roberta-base'
        else: 
            name='distilbert'
            encoding='distilbert-base-uncased'
        
        self.model=TransformerModel(name, encoding, num_labels=8, reprocess_input_data=True, 
                                   num_train_epochs=self.epochs, learning_rate=self.lr, 
                  output_dir=self.output_dir, overwrite_output_dir=True, fp16=False, save_steps=50000)
    
    def train_model(self): 
        self.model.train(self.train_data['text'], self.train_data['cmp_code'], show_running_loss=True)
    
    def evaluate_model(self, valid_data):
        result, model_outputs, wrong_predictions = self.model.evaluate(valid_data['text'], valid_data['cmp_code'])
        preds = np.argmax(model_outputs, axis = 1)
        correct=0
        labels = valid_data['cmp_code'].tolist()
        assert len(labels)==len(preds), "the size of the prediction must be equal to the size of the labels"
        for j in range(len(labels)):
            if preds[j] == labels[j]:
                correct+=1
        accuracy = correct/len(labels)
        print("Accuracy_Valid: ", accuracy)
        
        print("Metrics")
        print(metrics.classification_report(labels, preds, digits=3))
        self.print_info(str(metrics.classification_report(labels, preds, digits=3)))
        
        print("Confusion Matrix")
        cm = confusion_matrix(labels,preds)
        print(cm)
        self.print_info(str(cm))
    
    def run_model(self, valid_data): 
        self.set_model()
        self.train_model()
        self.evaluate_model(valid_data)
    
    def print_info(self, data): 
        with open("results.txt", "a") as f: 
            f.write(data)

        




