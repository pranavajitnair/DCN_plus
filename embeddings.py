import torch
from torchnlp.word_to_vector import CharNGram

from CoVe import MTLSTM

def get_ngram_char_embeddings(tokenized_text,vector_of):           
        vectors=[]
        for token in tokenized_text:
                vectors.append(vector_of[token.text].view(1,-1))
                
        output_vectors=torch.cat(vectors,dim=0).unsqueeze(0)
        output_vectors.requires_grad=False
        
        return output_vectors
    
def get_GloVe_embeddings(tokenized_text):
        vectors=[]
        for token in tokenized_text:
                vectors.append(torch.from_numpy(token.vector).view(1,-1))
                
        output=torch.cat(vectors,dim=0).unsqueeze(0)
        
        return output
    
def get_CoVe_embeddings(GloVe_embeddings,MTLSTM):
        lengths=torch.tensor(GloVe_embeddings.shape[1]).view(1,)        
        embeddings=MTLSTM(GloVe_embeddings,lengths)
        embeddings_return=embeddings.detach()
        
        return embeddings_return
    
    
class DataLoader(object):
        def __init__(self,tokenized_data):
                self.tokenized_data=tokenized_data
                
                self.vectors=CharNGram()
                self.MTLSTM=MTLSTM()
                
                self.counter=0
                self.counter1=0
                
        def __get_next__(self):
                context=self.tokenized_data['contexts'][self.counter]
                question=self.tokenized_data['questions'][self.counter][self.counter1]
                
                ngram_context=get_ngram_char_embeddings(context,self.vectors)
                GloVe_context=get_GloVe_embeddings(context)
                CoVe_context=get_CoVe_embeddings(GloVe_context,self.MTLSTM)
                context_embeddings=torch.cat((ngram_context,GloVe_context,CoVe_context),dim=-1)
                
                ngram_question=get_ngram_char_embeddings(question,self.vectors)
                GloVe_question=get_GloVe_embeddings(question)
                CoVe_question=get_CoVe_embeddings(GloVe_question,self.MTLSTM)
                question_embeddings=torch.cat((ngram_question,GloVe_question,CoVe_question),dim=-1)
                
                answer=self.tokenized_data['answers'][self.counter][self.counter1]
                
                if len(self.tokenized_data['questions'][self.counter])==self.counter1+1:
                        self.counter+=1
                        self.counter1=0
                else:
                        self.counter1+=1
                        
                if self.counter==len(self.tokenized_data['contexts']):
                        self.counter=0
                        
                return context_embeddings,question_embeddings,answer,context 