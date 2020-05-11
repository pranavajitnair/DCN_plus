import torch
import torch.nn as nn
import torch.nn.functional as F


class HMN(nn.Module):
        def __init__(self,pooling_size,dim):
                super(HMN,self)._init__()
                
                self.dim=dim
                self.pooling_size=pooling_size
                
                self.maxout2=nn.Linear(dim,dim*pooling_size)
                self.maxout3=nn.Linear(2*dim,pooling_size)
                
        def forward(self,m1):
                m2=self.maxout2(m1).view(1,-1,self.dim,self.pooling_size).max(3)
                m3=self.maxout3(torch.cat((m1,m2),dim=-1)).view(1,-1,1,self.pooling_size).max(3)
                
                return m3.contiguous().view(1,1,-1)
            
            
class Experts(nn.Module):
        def __init__(self,num_experts,dim,k):
                super(Experts,self).__init__()
                
                self.k=k
                self.num_experts=num_experts
                self.dim=dim
                
                self.non_noise=nn.Linear(3*dim,num_experts*dim)
                self.noise=nn.Linear(3*dim,num_experts*dim)
                self.E=nn.Linear(3*dim,num_experts*dim)
                
                self.r=nn.Linear(5*dim,dim)
                
        def forward(self,h,us,ue,u):
                R=self.r(torch.cat((h,us,ue),dim=-1)).expand(-1,u.shape[1],-1)
                input_to_experts=torch.cat((u,R),dim=-1)
                
                h1=self.non_noise(input_to_experts).view(1,-1,self.dim,self.num_experts)
                h2=self.noise(input_to_experts).view(1,-1,self.dim,self.num_experts)*torch.randn(h1.shape,requires_grad=False)
                h=h1+h2
                
                g=self.sparsify(h)
                e=self.E(input_to_experts).view(1,u.shape[1],self.dim,self.num_experts)
                output=g*e.mean(-1)
                
                return output
                
        def sparsify(self,matrix):
                values,indices=self.get_top_k(matrix)
                
                one_hot=torch.zeros((2,self.num_experts))
                one_hot[indices[0]]=1
                one_hot[indices[1]]=1
                output=matrix*one_hot
                output=F.softmax(output+-1e30*(output==0).type(torch.float),dim=-1)
                
                return output
                
        def get_top_k(self,matrix):
                values=[]
                indices=[]
                
                for i in range(self.k):
                        value,index=torch.max(matrix,dim=-1)
                        values.append(value)
                        indices.append(index)
                        
                        if i+1!=self.k:
                                one_hot=torch.zeros(self.num_experts)
                                one_hot[index]=float('-inf')
                                matrix+=one_hot
                                
                return values,indices