import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as MultiNomial

from utils import f1_score,get_answer,exact_match_score

def crossEntropyLoss(start_logits,end_logits,answer):
        lossFunction=nn.CrossEntropyLoss()
        loss=0
        
        for i in range(start_logits.shape[1]):
                loss+=lossFunction(start_logits[:,i,:].view(1,-1),torch.tensor(answer['start_offset']).view(1))
                loss+=lossFunction(end_logits[:,i,:].view(1,-1),torch.tensor(answer['end_offset']).view(1))
                
        return loss
    
def reinforcement_gradient_loss(start_logits,end_logits,start_guess,end_guess):
        lossFunction=nn.CrossEntropyLoss()
        loss=0
                
        for i in range(start_logits.shape[1]):
                loss+=lossFunction(start_logits[:,i,:].view(1,-1),start_guess[i].view(1))
                loss+=lossFunction(end_logits[:,i,:].view(1,-1),end_guess[i].view(1))
                
        return loss
    
def reward(document,sampled_starts,sampled_ends,greedy_start,greedy_end,gold_start,gold_end):
        rewards=[]
        gold_answer,baseline_answer=get_answer(document,gold_start,gold_end,int(greedy_start),int(greedy_end))
        baseline=f1_score(baseline_answer,gold_answer)
        em=exact_match_score(baseline_answer,gold_answer)
        
        for i in range(4):
                gold_answer,sample_answer=get_answer(document,gold_start,gold_end,int(sampled_starts[i]),int(sampled_ends[i]))
                f1=f1_score(sample_answer,gold_answer)
                
                normalized_reward=f1-baseline
                rewards.append(normalized_reward)
                
        return rewards,baseline,em
    
def loss_function_for_reinforcement(start_logits,end_logits,answer,document):
        final_start_logits=start_logits[:,-1,:].view(-1)
        final_end_logits=end_logits[:,-1,:].view(-1)
        
        greedy_start=torch.argmax(final_start_logits)[0]
        greedy_end=torch.argmax(final_end_logits)[0]
        
        sample_start=[]
        sample_end=[]
        for i in range(4):
                start_logit=start_logits[:,i,:].view(-1)
                end_logit=end_logits[:,i,:].view(-1)
                
                start_sample=MultiNomial(logits=start_logit).sample().detach()
                end_sample=MultiNomial(logits=end_logit).sample().detach()
                
                sample_start.append(start_sample)
                sample_end.append(end_sample)
                
        rewards,baseline,em=reward(document,sample_start,sample_end,greedy_start,greedy_end,answer['start_offset'],answer['end_offset'])
        expected_reward=-sum(rewards)/len(rewards)
        
        reinforcement_loss=reinforcement_gradient_loss(start_logits,end_logits,sample_start,sample_end)
        
        return reinforcement_loss+(expected_reward-reinforcement_loss.item()),baseline,em

def get_loss(start_logits,end_logits,answer,document,sigma_ce,sigma_rl):
        ce_loss=crossEntropyLoss(start_logits,end_logits,answer)
        rl_loss,baseline,em=loss_function_for_reinforcement(start_logits,end_logits,document)
        
        ce_loss/=(2*sigma_ce**2)
        rl_loss/=(2*sigma_rl**2)
        
        ce_loss+=torch.log(sigma_ce**2)
        rl_loss+=torch.log(sigma_rl**2)
        
        return ce_loss+rl_loss,baseline,em