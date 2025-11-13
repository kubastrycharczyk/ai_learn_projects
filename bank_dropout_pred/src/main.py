import model
import config
import trainer
import torch.nn as nn
import torch
import torch.optim as optim
import data_loader

model =  model.Classifier(config.len_dataset)
trainer = trainer.Trainer(model,
                          optim.Adam(model.parameters(), lr=0.001),
                          data_loader.loader,
                          nn.BCELoss(),
                          10
                          )
trainer.train(patience=10)
trainer.tester(['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc'], 0.3)