import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import get_loss
from maxout_and_experts import HMN,Experts


class Encoder(nn.Module):
        def __init__(self,dim,input_dim):
                super(Encoder,self).__init__()
                
                self.lstm1=nn.LSTM(input_dim,dim,num_layers=1,bidirectional=True,batch_first=True)
                self.lstm2=nn.LSTM(dim*2,dim,num_layers=1,bidirectional=True,batch_first=True)
                self.ulstm=nn.LSTM(12*dim,dim,num_layers=1,bidirectional=True,batch_first=True)
                
                self.sentinal=nn.Parameter(torch.randn(1,1,2*dim))
                self.sentinalq=nn.Parameter(torch.randn(1,1,2*dim))
                self.q_encode=nn.Linear(2*dim,2*dim)
            
        def forward(self,document_embeddings,question_embeddings):
                ED1,(final_hidden_state,final_cell_state)=self.lstm1(document_embeddings,None)
                EQ1,(qfinal_hidden_state,qfinal_cell_state)=self.lstm1(question_embeddings,None)
                EQ1=torch.tanh(self.q_encode(EQ1))
                
                ED1=torch.cat((self.sentinal,ED1),dim=-2)
                EQ1=torch.cat((self.sentinalq,EQ1),dim=-2)
                
                A1=torch.bmm(ED1,EQ1.transpose(1,2))
                
                SD1=torch.bmm(F.softmax(A1,dim=-2),EQ1)
                SQ1=torch.bmm(F.softmax(A1.transpose(1,2),dim=-2),ED1)
                
                SD1=SD1[:,1:,:]
                SQ1=SQ1[:,1:,:]
                A1=A1[:,1:,1:]
                
                CD1=torch.bmm(F.softmax(A1,dim=-2),SQ1)
                
                ED2,(final_hidden_state2,final_cell_state2)=self.lstm2(SD1,None)
                EQ2,(qfinal_hidden_state2,qfinal_cell_state2)=self.lstm2(SQ1,None)
                
                ED2=torch.cat((self.sentinal,ED2),dim=-2)
                EQ2=torch.cat((self.sentinalq,EQ2),dim=-2)
                
                A2=torch.bmm(ED2,EQ2.transpose(1,2))
                
                SD2=torch.bmm(F.softmax(A2,dim=-2),EQ2)
                SQ2=torch.bmm(F.softmax(A2.transpose(1,2),dim=-2),ED2)
                
                SD2=SD2[:,1:,:]
                SQ2=SQ2[:,1:,:]
                A2=A2[:,1:,1:]
                
                CD2=torch.bmm(F.softmax(A2,dim=-2),SQ2)
                
                ED1=ED1[:,1:,:]
                ED2=ED2[:,1:,:]
                input_to_U=torch.cat((ED1,ED2,SD1,SD2,CD1,CD2),dim=-1)
                U,(ufinal_hidden_state,ufinal_cell_state)=self.ulstm(input_to_U,None)
                
                return U
            
            
class Decoder(nn.Module):
        def __init__(self,pooling_size,dim,k,max_decoding_steps):
                super(Decoder,self).__init__()
                self.max_decoding_steps=max_decoding_steps
                
                self.lstm=nn.LSTM(4*dim,dim,batch_first=True)
                
                self.hmn=HMN(pooling_size,dim)
                self.experts=Experts(pooling_size,dim,k)
                
                self.hmn_end=HMN(pooling_size,dim)
                self.experts_end=Experts(pooling_size,dim,k)
                
        def forward(self,U):
                start_index=0
                end_index=U.shape[1]-1
                hidden_state=None
                cell_state=None
                
                start_logits=[]
                end_logits=[]
                
                for i in range(self.max_decoding_steps):
                        U_start=U[:,start_index,:].view(1,1,-1)
                        U_end=U[:,end_index,:].view(1,1,-1)
                        U_input=torch.cat((U_start,U_end),dim=-1)

                        if i==0:
                            hidden,(hidden_state,cell_state)=self.lstm(U_input,None)
                        else:
                            hidden,(hidden_state,cell_state)=self.lstm(U_input,(hidden_state,cell_state))
                        
                        m1_equivalent=self.experts(hidden,U_start,U_end,U)
                        logits_start=self.hmn(m1_equivalent)
                        
                        m1_equivalent_end=self.experts_end(hidden,U_start,U_end,U)
                        logits_end=self.hmn_end(m1_equivalent_end)
                        
                        start_index=int(torch.argmax(logits_start,dim=-1)[0][0])
                        end_index=int(torch.argmax(logits_end,dim=-1)[0][0])
                        
                        start_logits.append(logits_start)
                        end_logits.append(logits_end)
                        
                return torch.cat(start_logits,dim=-2),torch.cat(end_logits,dim=-2)
            
            
class Model(nn.Module):
        def __init__(self,dim,input_dim,pooling_size,k,max_decoding_steps):
                super(Model,self).__init__()
                
                self.encoder=Encoder(dim,input_dim)
                self.decoder=Decoder(pooling_size,dim,k,max_decoding_steps)
                
                self.sigma_ce=nn.Parameter(torch.randn(1))
                self.sigma_rl=nn.Parameter(torch.randn(1))
                
        def forward(self,document_embeddings,question_embeddings,answer,document):
                U=self.encoder(document_embeddings,question_embeddings)
                start_logits,end_logits=self.decoder(U)
                
                loss,baseline,em=get_loss(start_logits,end_logits,answer,document,self.sigma_ce[0],self.sigma_rl[0])
                
                return loss,baseline,em