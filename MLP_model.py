import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

# Function to train the model
def create_train_MLP_model(X_input, y_input, perturb, X_test, hidden_size = 128,    learning_rate = 0.001,
                           num_epochs=100,    batch_size = 256, ):
    def binarize(y, thrs):
        ind= y<thrs
        y[ind]=0
        y[~ind]=1
        return y
    input_size = X_input.shape[1]
    output_size = 1
    model = MLPClassifier(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_train = torch.from_numpy(X_input.astype(np.float32))
    y_train = torch.from_numpy(y_input.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for e in range(num_epochs):
        loss_i = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_i.append(loss.item())
        #print(np.mean(loss_i))
    model.eval()
    weightp = model.fc2.weight.data + perturb.astype(np.float32)
    weightn = model.fc2.weight.data - perturb.astype(np.float32)
    y_pred = model(X_test).detach().cpu().numpy()  # orignal model
    model.fc2.weight.data = weightp
    y_predp = model(X_test).detach().cpu().numpy()  # perturbated positive
    model.fc2.weight.data = weightn
    y_predn = model(X_test).detach().cpu().numpy()  # perturbated negative
    y_pred = binarize(y_pred, thrs=0.5)
    y_predp = binarize(y_predp, thrs=0.5)
    y_predn = binarize(y_predn, thrs=0.5)


    return y_pred.squeeze(), y_predp.squeeze(), y_predn.squeeze()


