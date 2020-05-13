# DCN+
This is a PyTorch Implenetation of Dynamic Coattention Networks + described in the paper [DCN+: MIXED OBJECTIVE AND DEEP RESIDUAL
COATTENTION FOR QUESTION ANSWERING](https://arxiv.org/pdf/1711.00106.pdf)

A TensorFlow implementation can be found [here](https://github.com/mjacar/tensorflow-dcn-plus) 
## Dataset
The SQuAD dataset can be found [here](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset)
## Preprocessing data and training the Model  
Preprocess the SQuAD data by running 

``` 
python utils.py
```

Optional Arguments

```
   --train_file           path to training data for SQuAD
   --dev_file             path to development data for SQuAD
   --preprocessed_train   path to store the processed training data
   --preprocessed_dev     path to store the processed development data
   ```
   
   To train the Model
   
   ```
   python train.py
   ```
   
   Optional Arguments
   
   ```
        --learning_rate              learning rate
        --hidden_size                hidden dimension for LSTM's and for Experts and Maxouts layers
        --pooling_size               pooling size for Maxout and Experts layers
        --dropout                    dropout for LSTM's
        --tokenized_train_data_path  path to the preprocessed training data
        --tokenized_dev_data_path    path to the preprocessed development data
    ```
