### Version 0.1

#### Key Hyperparameters
- decistion threshold = 0.3
- learning rate = 0.001
- number of epochs = 10



#### Model build

Classifier
- (linear1): Linear(in_features=7, out_features=128, bias=True)
- (linear2): Linear(in_features=128, out_features=64, bias=True)
- (linear3): Linear(in_features=64, out_features=1, bias=True)
- (act1): ReLU()
- (act2): ReLU()
- (act3): Sigmoid()

Optimizer:
- Adam

Loss:
- Binary Cross-Entropy Loss


#### Model stats
- accuracy : 0.803
- f1_score : 0.1825726141078838
- precision : 0.5365853658536586
- recall : 0.11
- roc_auc : 0.41771875

#### Analysis of results:
1. Worse than random Performance: A score below 0.5 (0.42) means that the model is performing worse than in random quessing which means that the model is criticlly bad.
2. Failure to identify Positives: Extremly low recall shows that model fails to identify 89% of all true postives. 
3. Misleading accuracy: High metric of accuracy with really lor Roc Auc probably means that model quesses one classes that occures more often and thats way achieves high score with ignoring second class.


