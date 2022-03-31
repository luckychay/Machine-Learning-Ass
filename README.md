
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
