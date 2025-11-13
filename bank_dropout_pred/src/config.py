x_labels = ["country", "gender", "age", "balance", "active_member", "products_number"]
x_labels_cat = ["country", "gender"]
y_labels = ["churn"]
path = "data/Bank Customer Churn Prediction.csv"
batch_size = 32
lenght = len(x_labels)
split_size = [0.8, 0.1, 0.1]
len_dataset = len(x_labels) + len(y_labels)