
# Machine-Learning-Ass

## Usage
To run the code, requirements should be meet as indicated in requirements.txt.

`pip install -r requirements.txt`

For training,

`python classifier.py --mode train --classifier <classifier name>`

where "classifier name" refers to different classifiers, including "LDA", "Bayes", "DT".

For testing,

`python classifier.py --mode test --classifier <classifier name>`

For visualization,

`python classifier.py --mode vis`

## Points about LDA
There are two different definitions about with-class and between-class scatter matrix

The first one is as follows, which is used by this repo:
![1648607797(1)](https://user-images.githubusercontent.com/55084546/160739741-c123a144-8bca-4532-847b-459526a73ebc.png)

The second one is as follows, which is used by this repo and sklearn. The difference lays not only on multiplying Sw and Sb with priors probabilities, but also on changing the mean vector for deduction in Sb. 
![1648608329(1)](https://user-images.githubusercontent.com/55084546/160740728-60df572e-7188-4b15-9a04-4a0f26542261.png)

Note that, generally we can use LDA both for dimensionality reduction and classificiation, when for classification we use **all** the eigenvectors and while choosing the **leading c-1**(c is the total class number) eigenvectors for dimensionality reduction.

## Points about matrix 𝐀′𝐀

Geometrically, matrix 𝐀′𝐀 is called matrix of scalar products (= dot products, = inner products). Algebraically, it is called sum-of-squares-and-cross-products matrix (SSCP).

Its 𝑖-th diagonal element is equal to ∑𝑎2(𝑖), where 𝑎(𝑖) denotes values in the 𝑖-th column of 𝐀 and ∑ is the sum across rows. The 𝑖𝑗-th off-diagonal element therein is ∑𝑎(𝑖)𝑎(𝑗).

There is a number of important association coefficients and their square matrices are called angular similarities or SSCP-type similarities:

Dividing SSCP matrix by 𝑛, the sample size or number of rows of 𝐀, you get MSCP (mean-square-and-cross-product) matrix. The pairwise formula of this association measure is hence ∑𝑥𝑦/𝑛 (with vectors 𝑥 and 𝑦 being a pair of columns from 𝐀).

If you center columns (variables) of 𝐀, then 𝐀′𝐀 is the scatter (or co-scatter, if to be rigorous) matrix and 𝐀′𝐀/(𝑛−1) is the covariance matrix. Pairwise formula of covariance is ∑𝑐𝑥𝑐𝑦/𝑛−1 with 𝑐𝑥 and 𝑐𝑦 denoting centerted columns.

If you z-standardize columns of 𝐀 (subtract the column mean and divide by the standard deviation), then 𝐀′𝐀/(𝑛−1) is the Pearson correlation matrix: correlation is covariance for standardized variables. Pairwise formula of correlation is ∑𝑧𝑥𝑧𝑦/𝑛−1 with 𝑧𝑥 and 𝑧𝑦 denoting standardized columns. The correlation is also called coefficient of linearity.

If you unit-scale columns of 𝐀 (bring their SS, sum-of-squares, to 1), then 𝐀′𝐀 is the cosine similarity matrix. The equivalent pairwise formula thus appears to be ∑𝑢𝑥𝑢𝑦=∑𝑥𝑦/√∑𝑥2√∑𝑦2 with 𝑢𝑥 and 𝑢𝑦 denoting L2-normalized columns. Cosine similarity is also called coefficient of proportionality.

If you center and then unit-scale columns of 𝐀, then 𝐀′𝐀 is again the Pearson correlation matrix, because correlation is cosine for centered variables1,2: ∑𝑐𝑢𝑥𝑐𝑢𝑦=∑𝑐𝑥𝑐𝑦/√∑𝑐2𝑥√∑𝑐2𝑦
