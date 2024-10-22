# Decision Trees
This is an project coursework for COMP60012 Introduction to Machine Learning.

## Installation
To download all dependencies please run the following code in the terminal:
```
pip install requirements
```

# Members
| Name                              | CID        | Course |
|-----------------------------------|------------|--------|
| Tianqi Hu                         | 02190317   | EIE    |
| Samuel Khoo                       |            | EIE    |
| Anson Chin                        | 02194736   | EIE    |
| Constance Geneau De Lamarliere    | 02209964   | EIE    |

# How 


1) sort according to wifi number in decreasing order. Only consider the wifi column and the class column for less complexity
2) Evaluate CUTS where the room number changes for a specific valu of wifi number. Evaluate and compare to previous entropy and if better, store in the form: best_split = (attribute, value, entropy)
3) Iterate though all the cuts of this wifi column and only store if better
4) Once a full wifi has been evaluated and the best split (cut) has been found, move on to the next wifi