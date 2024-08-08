import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os

class MLPClassifier(nn.Module):
    """
    MLP mocdel
    """
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
def train_perturb_evaluate_MLP_model(X_input, y_input, perturb, X_test, model, optimizer,  criterion,
                           num_epochs=100,    batch_size = 256, device=None, out_f_name=None, perturb_mode='gaussian',
                                     forced=False):
    """
    Train the model and perturb the weights to create two different models
    :return:
    """


    perturb1, perturb2=perturb
    def binarize(y, thrs):
        ind= y<thrs
        y[ind]=0
        y[~ind]=1
        return y

    need_repeat = False
    if os.path.isfile(out_f_name) and not forced:
        try:
            state_dict0 = torch.load(out_f_name,map_location=device)
            model.load_state_dict(state_dict0)  # reset model weights
            model.eval()
        except:
            need_repeat = True
    else:
        need_repeat = True
    if need_repeat:

        X_train, X_val, y_train, y_val = train_test_split(X_input, y_input, test_size=0.2)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model.train()
        best_acc_v = np.inf
        patience = 5 #parameter of early stopping
        #start_time = time.time()
        early_stop_counter = 0
        for e in range(num_epochs):
            loss_i = []
            breaked = False
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()

                model.eval()
                out_v = model(X_val)
                acc_v = criterion(out_v, y_val.float().unsqueeze(1)).item()
                if acc_v< best_acc_v:
                    best_acc_v = acc_v
                    early_stop_counter = 0
                else:
                    early_stop_counter+= 1
                model.train()
                if early_stop_counter>patience:
                    breaked = True
                    break

            if breaked:
                break



        model.eval()
        torch.save(model.state_dict(), out_f_name)
    if perturb_mode is not None:
        with torch.no_grad():
            if perturb_mode == 'gaussian':
                """
                old

                """
                if perturb2 is not None:
                    weightp = model.fc2.weight.data + perturb1
                    weightn = model.fc2.weight.data + perturb2
                else:
                    weightp = model.fc2.weight.data + perturb1
                    weightn = model.fc2.weight.data - perturb1
            elif perturb_mode == 'uniform':
                weightp = model.fc2.weight.data + perturb1
                weightn = model.fc2.weight.data - perturb1
            y_pred = model(X_test)  # orignal model
            y_pred_train = model(X_input) # original model X_train
            model.fc2.weight.data = weightp
            y_predp = model(X_test)  # perturbated positive
            model.fc2.weight.data = weightn
            y_predn = model(X_test)  # perturbated negative
        y_pred = binarize(y_pred, thrs=0.5)
        y_predp = binarize(y_predp, thrs=0.5)
        y_pred_train = binarize(y_pred_train, thrs=0.5)
        y_predn = binarize(y_predn, thrs=0.5)


        return y_pred.squeeze(), y_predp.squeeze(), y_predn.squeeze(), y_pred_train.squeeze()
    else:
        with torch.no_grad():

            y_pred = model(X_test)  # orignal model
            y_pred_train = model(X_input)  # original model X_train

        y_pred = binarize(y_pred, thrs=0.5)
        y_pred_train = binarize(y_pred_train, thrs=0.5)

        return y_pred.squeeze(), y_pred_train.squeeze()

def calc_accuracy(y_pred, y_true):
    correct = torch.sum(y_true == y_pred).item()
    total = len(y_true)
    return correct / total
