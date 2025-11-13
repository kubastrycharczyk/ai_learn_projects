import torch
import config
import torch.nn as nn
import torch.optim as optim
import data_loader
import importlib
importlib.reload(config)
importlib.reload(data_loader)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



class ClassNet(nn.Module):
    def __init__(self, in_size):
        super(ClassNet, self).__init__()
        self.linear_1 = nn.Linear(in_size, 128)
        self.act1 = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.linear_3 = nn.Linear(64, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act1(x)
        x = self.linear_2(x)
        x = self.act2(x)
        x = self.linear_3(x)
        x = self.act3(x)
        return x
    

model = ClassNet(7)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

epoch_num = 500
best_val_loss = float("inf")
patience_count = 0
patience = 10



def evaluate_validation(model, val_loader, loss_fn):
    model.eval() 
    total_val_loss = 0.0
    

    with torch.no_grad(): 
        for X, y in val_loader:

            predictions = model(X)
            

            loss = loss_fn(predictions, y)
            total_val_loss += loss.item() * X.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    
    model.train() 
    
    return avg_val_loss
        
def train_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    total_train_loss = 0.0
    for batch_idx, (X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * X.size()[0]
    avg_train_loss = total_train_loss /len(train_loader.dataset)
    return avg_train_loss






for epoch in range(epoch_num):
    train_epoch(model, data_loader.train_loader, loss_func, optimizer)

    val_loss = evaluate_validation(model, data_loader.val_loader, loss_func)

    if val_loss<best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"Błąd walidacyjny nie spadł przez {patience} epok. Zatrzymuję uczenie.")
            break

model.load_state_dict(torch.load('best_model.pth', weights_only=True))




def evaluate_test(model, test_loader):
    model.eval() 
    
    all_y_true = []
    all_y_pred_probs = [] 
    
    with torch.no_grad(): 
        for X, y in test_loader:
            predictions = model(X)
            
            all_y_true.append(y.cpu())
            all_y_pred_probs.append(predictions.cpu())

    y_true = torch.cat(all_y_true).numpy()
    y_pred_probs = torch.cat(all_y_pred_probs).numpy()
    

    y_pred = (y_pred_probs >= 0.25).astype(int) 

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc_roc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc_roc = 'N/A (Potrzebne obie klasy w zbiorze testowym)'


    model.train() 
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc
    }



if hasattr(data_loader, 'test_loader'):
    test_metrics = evaluate_test(model, data_loader.test_loader)

    print("\n--- Wyniki Oceny na Zbiorze Testowym ---")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"**{metric}:** {value:.4f}")
        else:
            print(f"**{metric}:** {value}")
else:
    print("Brak 'test_loader' w module 'data_loader'. Upewnij się, że masz zbiór testowy do oceny!")