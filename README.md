
# Machine-Learning-Ass

## LDA
There are two different definitions about with-class and between-class scatter matrix

The first one is as follows, which is used by this repo:

\mathbf{S}_w = \sum_{i=1}^n (\mathbf{x}_i - \boldsymbol{\mu}_{y_i}) (\mathbf{x}_i - \boldsymbol{\mu}_{y_i})^T

\mathbf{S}_b = \sum_{k=1}^m n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) (\boldsymbol{\mu}_k - \boldsymbol{\mu})^T

The second one is as follows, which is used by numpy and sklearn:
\mathbf{S}_w^* &= n \sum_{k=1}^m \frac{1}{n_k} \sum_{i \mid y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_{k}) (\mathbf{x}_i - \boldsymbol{\mu}_{k})^T

\mathbf{S}_b^* &= n \sum_{k=1}^m (\boldsymbol{\mu}_k - \boldsymbol{\mu}^*) (\boldsymbol{\mu}_k - \boldsymbol{\mu}^*)^T

Note that, generally we can use LDA both for dimensionality reduction and classificiation, when for classification we use all the eigenvectors and while choosing the leading c-1(c is the total class number) eigenvectors for dimensionality reduction.
