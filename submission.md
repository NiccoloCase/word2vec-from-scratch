I have implemented the skip-gram variant of Word2Vec with negative sampling, following Mikolov
et al. (2013), entirely in NumPy. I used the text8 corpus for training, which is a cleaned version of Wikipedia text. The implementation includes data preprocessing (tokenization, subsampling), model definition, training loop with negative sampling, and utilities for evaluating the learned nearest neighbors and analogies).

Code of the submission: https://github.com/NiccoloCase/word2vec-from-scratch

A more detailed README is included in the repo, outlining the project structure, training instructions, and design choices.

Concerning the mathematical derivation of the loss function and gradients for backpropagation, I have included in the repo a pdf report (derivation.pdf) that walks through the derivation of the negative sampling loss and its gradients with respect to the input and output embeddings.

I traind on 3M tokens from the text8 corpus for 1 epoch, with an embedding dimension of 100, a maximum window size of 3, and 3 negative samples per positive pair. The training took approximately 1 hour on a snellius HCP rome partition with 3d gigs of ram.

The final embeddings are saved in output/embeddings_test0.npz, and the loss curve is saved in output/loss_test0.png.

I have also included a Jupyter notebook (eval_embeddings.ipynb) that loads the saved embeddings and allows for exploration of nearest neighbors and analogies.

Finaly i want to remark that the task aliged very closely with my academic background, and I would be excited to contribute to it. I particularly enjoyed working on this task because it connects with my background. In my Deep Learning I course, our first assignment required implementing the same MLP both in PyTorch and in pure NumPy, while my Machine Learning I course emphasised deriving backpropagation by hand. Therefore, the task felt conceptually familiar, although I had never implemented word2vec from scratch before. I studied word2vec and skip-gram during my Natural Language Processing course at a theoretical level, but implementing it allowed me to better understand the mechanics of negative sampling and contrastive learning in practice, which I found very interesting.
