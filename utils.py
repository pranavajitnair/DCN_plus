import json
import re
import string
from collections import Counter
import spacy
import pickle
import argparse
import os

def read_data(filename):
        with open(filename) as f:
                data=json.load(f)['data']
                
        return data
    
def get_data(data):
        output={'contexts':[],'answers':[],'questions':[]}
        
        for articles in data:
                for paragraph in articles['paragraphs']:
                        output['contexts'].append(paragraph['context'])
                        
                        question_store=[]
                        answer_store=[]
                        for qa in paragraph['qas']:
                                question_store.append(qa['question'])
                                answer_store.append(qa['answers'][0])
                                
                        output['answers'].append(answer_store)
                        output['questions'].append(question_store)
                        
        return output
                                
def normalize_answer(s):     
        def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
    
        def white_space_fix(text):
                return ' '.join(text.split())
    
        def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
    
        def lower(text): 
                return text.lower()
    
        return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, decode_bytes=False):
        
        if decode_bytes:
                prediction = prediction.decode('utf-8')
                ground_truth = ground_truth.decode('utf-8')
                
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
                return 0.0
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
          
        return f1

def exact_match_score(prediction, ground_truth):
        
        return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
                score = metric_fn(prediction, ground_truth)
                scores_for_ground_truths.append(score)
                
        return max(scores_for_ground_truths)
    
def tokenize(data):
        tokenizer=spacy.load('en_core_web_lg')
        output={'contexts':[],'questions':[],'answers':[]}
        
        for i,document in enumerate(data['contexts']):
                document_tokens=tokenizer(document)
                output['contexts'].append(document_tokens)
                question_set=[]
                answer_set=[]
                
                for j,question in enumerate(data['questions'][i]):
                        question_tokens=tokenizer(question)
                        answer=find_answer(document_tokens,data['answers'][i][j])
                        if answer!=-1:
                                question_set.append(question_tokens)
                                answer_set.append(answer)
                                
                if len(question_set)==0:
                        output['contexts'].pop()
                else:
                        output['questions'].append(question_set)
                        output['answers'].append(answer_set)
                        
        return output

def find_answer(document,answer):
        start_offset=answer['answer_start']
        end_offset=start_offset+len(answer['text'])-1
        
        answer_insert={'start_offset':-1,'end_offset':-1,'text':answer['text']}
        for i,token in enumerate(document):
                offset=token.idx
                
                if offset==start_offset:
                        answer_insert['start_offset']=i
                if offset+len(token.text)-1==end_offset:
                        answer_insert['end_offset']=i
                        break
                    
        if answer_insert['start_offset']!=-1 and answer_insert['end_offset']!=-1:
                return answer_insert
        else:
                return -1

def read_from_file(file_name):
        file=open(file_name,"rb")
        tokenized_data=pickle.load(file)
        
        return tokenized_data
    
def get_answer(document,ans_start,ans_end,predict_start,predict_end):
        ans=''
        for i in range(ans_start,ans_end+1):
                if i!=ans_end-1:
                        ans+=document[i].text+' '
                else:
                        ans+=document[i].text
        
        predict_ans=''               
        for i in range(predict_start,predict_end+1):
                if i!=predict_end-1:
                        predict_ans+=document[i].text+' '
                else:
                        predict_ans+=document[i].text
                        
        return ans,predict_ans
    
def setup():
        parser=argparse.ArgumentParser('files parser')
        
        parser.add_argument('--train_file',type=str,default=os.getcwd()+'/data/SQuAD 1.1/train-v1.1.json',help=' path to train file')
        parser.add_argument('--dev_file',type=str,default=os.getcwd()+'/data/SQuAD 1.1/dev-v1.1.json',help='path to dev file')
        parser.add_argument('--preprocessed_train',type=str,default=os.getcwd()+'/Preprocessed/train.pickle',help='path to file where preprocessed training data to be stored')
        parser.add_argument('--preprocessed_dev',type=str,default=os.getcwd()+'/Preprocessed/dev.pickle',help='path to file where preprocessed development data to be stored')
        
        args=parser.parse_args()
        
        return args
    
def main(args):
        train_path=args.train_file
        dev_path=args.dev_file
        train_store=args.preprocessed_train
        dev_store=args.preprocessed_dev
        
        data_train=read_data(train_path)
        data_train=get_data(data_train)
        tokenized_data_train=tokenize(data_train)
        
        data_dev=read_data(dev_path)
        data_dev=get_data(data_dev)
        tokenized_data_dev=tokenize(data_dev)
        
        file_train=open(train_store,'wb')
        pickle.dump(tokenized_data_train,file_train)
        
        file_dev=open(dev_store,'wb')
        pickle.dump(tokenized_data_dev,file_dev)
        
if __name__=='__main__':
        args=setup()
        main(args)