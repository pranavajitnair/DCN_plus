import torch
import torch.optim as optim

import argparse
import os

from utils import read_from_file
from model import Model
from embeddings import DataLoader

def train(Model,dataloader_train,dataloader_dev,train_iters,dev_iters,epochs,optimizer):
        for epoch in range(epochs):
                Model.train()
                loss=0
                optimizer.zero_grad()
                
                for _ in range(train_iters):
                        context_embeddings,question_embeddings,answer,document=dataloader_train.__get_next__()
                        loss_temp,_,em=Model(context_embeddings,question_embeddings,answer,document)
                        loss+=loss_temp
                        
                training_loss=loss.item()/train_iters
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                        Model.eval()
                        loss_dev=0
                        baseline=0
                        em=0
                        
                        for _ in range(dev_iters):
                                context_embeddings_dev,question_embeddings_dev,answer_dev,document_dev=dataloader_dev.__get_next__()
                                loss_dev_temp,baseline_temp,em_temp=Model(context_embeddings_dev,question_embeddings_dev,answer_dev,document_dev)
                                loss_dev+=loss_dev_temp
                                baseline+=baseline_temp
                                em+=em_temp
                                
                        dev_loss=loss_dev.item()/dev_iters
                        baseline/=dev_iters
                        em/=dev_iters
                
                print('epoch=',epoch,'training loss=',training_loss,'validation loss=',dev_loss,'validation F1 score=',baseline,'validation Exact Match score=',em)
                
def main(args):
        lr=args.learning_rate
        hidden_size=args.hidden_size
#        dropout=args.dropout
        pooling_size=args.pooling_size     
        input_dim=1000
        k=2
        max_decoder_steps=4
        
        epochs=args.epochs
        train_iters=80000
        dev_iters=10000
        
        train_file_path=args.tokenized_train_data_path
        dev_file_path=args.tokenized_dev_data_path
        
        train_data_processed=read_from_file(train_file_path)
        dev_data_processed=read_from_file(dev_file_path)

        model=Model(hidden_size,input_dim,pooling_size,k,max_decoder_steps)
        Dataloader_train=DataLoader(train_data_processed)
        Dataloader_dev=DataLoader(dev_data_processed)

        optimizer=optim.Adam(model.parameters(),lr=lr)
        
        train(model,Dataloader_train,Dataloader_dev,train_iters,dev_iters,epochs,optimizer)
    
def setup():
        parser=argparse.ArgumentParser('Hyperparameters argument parser')
        
        parser.add_argument('--learning_rate',type=float,default=0.1)
        parser.add_argument('--hidden_size',type=int,default=100)
        parser.add_argument('--pooling_size',type=int,default=16)
        parser.add_argument('--epochs',type=int,default=12)
        parser.add_argument('--dropout',type=float,default=0.2)
        parser.add_argument('--tokenized_train_data_path',type=str,default=os.getcwd()+'/Preprocessed/train.pickle')
        parser.add_argument('--tokenized_dev_data_path',type=str,default=os.getcwd()+'/Preprocessed/dev.pickle')
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
#        main(args)