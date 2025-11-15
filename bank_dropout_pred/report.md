
### Version 0.2

#### Key Hyperparameters
- decistion threshold = 0.5
- learning rate = 0.001
- number of epochs = 50



#### Model build

Classifier
- (norm0): BatchNorm1d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
- (linear1): Linear(in_features=7, out_features=128, bias=True)
- (norm1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
- (linear2): Linear(in_features=128, out_features=64, bias=True)
- (norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
- (linear3): Linear(in_features=64, out_features=1, bias=True)
- (act1): ReLU()
- (act2): ReLU()


Optimizer:
- Adam

Loss:
- Binary Cross-Entropy Loss with logit loss


#### Model stats
- accuracy : 0.832
- f1_score : 0.6233183856502242
- precision : 0.5650406504065041
- recall : 0.695
- roc_auc : 0.867103125

#### Analysis of results:
1. Better than random Performance: A score higher than 0.5 (0.87) and close to 1 means that our model starts to understand difference between classes and learns to distinct them.
2. Some ability to identify Positives: Almost 70% of all positive classes were correctly classified that is big step forward and show us that model stopped to blindly choose one class.
3. Mid precision: Precision score around 50% show that model still have some problem in differenting one class from another.




### Version 0.1

#### Key Hyperparameters
- decistion threshold = 0.3
    - standard 0.5 wasn't chosen due to the class inbalance that wasn't adressed in vesrion 0.1
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


