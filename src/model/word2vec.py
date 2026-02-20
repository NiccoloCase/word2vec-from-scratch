import random as rng
import numpy as np
from utils import sigmoid

eps = 1e-10

class MyWord2Vec:
  
    def __init__(self,
                 vocab_size: int, # (V) size of the vocabulary 
                 embed_dim: int  = 100, # (D) dimensionality of embeddings 
                 k: int = 5, # (K) number of negative samples for contrastive learning
                 lr: float       = 0.025, # learning rate
        ): 

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.k = k
        self.lr = lr

        # W_in -> word embeddings  (V, D)
        self.W_in  = (rng.random((vocab_size, embed_dim)) - 0.5) / embed_dim # uniform init 

        # W_out -> context embeddings (V, D)
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64) # zero init 


    def step(self,
            centre_idx: int, # index of the centre word
            pos_idx: int, # index of the positive context word 
            neg_idxs: np.ndarray, # indices of the negative samples 
            lr: float) -> float:
        
        """
        Perform a single training step for one (centre, context) pair and K negative samples.
        Runs the forward and backward passes, and updates the embeddings in-place with SGD.
        Returns the loss for this training example.
        """
            

        # notation:
        # c: centre word
        # o: positive context word
        # k: negative samples


        # FORWARD
        
        # first we get the word embedding for the target word 
        v_c = self.W_in[centre_idx] # (D)

        # for the positive and negative samples we get their context embeddings
        v_o = self.W_out[pos_idx]   # (D) 
        V_n = self.W_out[neg_idxs]  # (K, D)


        # now we can compute the scores for the positive and negative samples as thier dot product with the centre embedding

        # positive score between centre embedding and positive context embedding
        s_o  = v_c @ v_o  # (D) @ (D) -> scalar
        sig_o = sigmoid(s_o) # σ(s_o) = P(pos | centre)

        # negative scores between centre embedding and negative context embeddings
        s_k = V_n @ v_c # (K, D) @ (D) -> (K)
        sig_k = sigmoid(s_k) # σ(s_k) = P(neg | centre)


        # now we compute the loss for this training example
        # negative log-likelihood:
        # L = − y log(σ(s)) − (1−y) log(1−σ(s)) 
        # for the positive:
        # L_pos = -log σ(s_o)  (since y=1)
        # for the negatives:
        # L_neg = - sum_k { log(1 - σ(s_k)) }  (since y=0)
        # putting it together:
        # L = L_pos + L_neg = -log σ(s_o) - sum_k { log(1 - σ(s_k)) }       
        loss = -np.log(sig_o + eps) - np.sum(np.log(1.0 - sig_k + eps))  # we use log(1 - σ(s_k)) = log σ(-s_k) for numerical stability

        # BACKWARD
        # we have to perform backpropagation to compute the gradients for the input and output embeddings


        # 1) derivative of the loss w.r.t. the positive scores (s_o)

        # we consider only the positive part of the loss: -log σ(s_o)
        # and we know that the derivative of the sigmoid function is σ'(x) = σ(x) * (1 - σ(x))
        # for the chain rule we have that the derivative of -log σ(s_o) w.r.t. s_o is:
        # -1/σ(s_o) * σ'(s_o) = -1/σ(s_o) * σ(s_o)(1 - σ(s_o)) = -(1 - σ(s_o)) = σ(s_o) - 1
        grad_s_o = sig_o - 1.0                


        # 2) derivative of the loss w.r.t. the negative scores (s_k)

        # we consider only the negative part of the loss: - sum_k { log(1 - σ(s_k)) }
        # we fix a specifc k and we compute the derivative w.r.t. s_k:
        # -1/(1 - σ(s_k)) * (-σ'(s_k)) = σ'(s_k) / (1 - σ(s_k)) = σ(s_k) * (1 - σ(s_k)) / (1 - σ(s_k)) = σ(s_k) 
        grad_s_k = sig_k     


        # 3)  derivaive of the loss w.r.t. the centre embedding v_c 

        # dL = dL_pos + dL_neg
        # dL_pos / dv_c = dL/ds_o * ds_o/dv_c 
        # we also know that s_o = v_c @ v_o so for the multiplication rule ds_o/dv_c = v_o
        grad_vc_pos = grad_s_o * v_o  # (D)

        # dL_neg / d v_c = sum_k { dL/ds_k * ds_k/dv_c }
        # s_k = V_n @ v_c => ds_k/dv_c = V_n
        grad_vc_neg = grad_s_k @ V_n  # (K) @ (K, D) -> (D)

        grad_vc = grad_vc_pos + grad_vc_neg  # (D)


        # 4) derivative of the loss w.r.t. the positive context embedding v_o:

        # we consider only the positive part of the loss
        # dL/dv_o = dL/ds_o * ds_o/dv_o
        # ds_o/dv_o = v_c
        grad_vo = grad_s_o * v_c  # (D)
        
        
        # 5) derivative of the loss w.r.t. the negative context embeddings V_n:

        # we consider only the negative part of the loss
        # for a specific k we have:
        # dL/dV_n[k] = dL/ds_k * ds_k/dV_n[k]
        # s_k = V_n @ v_c => ds_k/dV_n[k] = v_c
        grad_Vn = np.outer(grad_s_k, v_c) # (K) @ (D) -> (K, D)


        # OPTIMIZER:
        # Stochastic gradient descent update
        self.W_in[centre_idx]  -= lr * grad_vc
        self.W_out[pos_idx]    -= lr * grad_vo
        self.W_out[neg_idxs]   -= lr * grad_Vn

        return float(loss)
