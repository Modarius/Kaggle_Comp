# This code is heavilly based on code written for CS6440 Image Processing homework
# which was based on the code provided in pytorch-tutorial-main
# which was provided as an example for the homework

from collections import OrderedDict
from time import time
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

## https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class banknote(Dataset):
    def __init__(self, filename):
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/a/5137509
        self.filename = filename
        self.data = importData(dir_path + filename, ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        temp = self.data.iloc[index].to_numpy().squeeze()
        x = temp[0:-1]
        y = temp[-1]
        return x, y

class modelNN(torch.nn.Module):
    def __init__(self, hl_width, depth, activation="relu"):
        super(modelNN, self).__init__()
        
        #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential
        model = OrderedDict()
        model["linear_input"] = torch.nn.Linear(4, hl_width) # input is 4 data points 'variance', 'skewness', 'curtosis', 'entropy'
        if (activation=="relu"):
            model["relu_input"] = torch.nn.ReLU()
        elif (activation=="tanh"):
            model["tanh_input"] = torch.nn.Tanh()

        for i in range(1, depth):
            model["linear" + str(i)] = torch.nn.Linear(hl_width, hl_width)
            if (activation=="relu"):
                model["relu" + str(i)] = torch.nn.ReLU()
            elif (activation=="tanh"):
                model["tanh" + str(i)] = torch.nn.Tanh()

        model["linear_output"] = torch.nn.Linear(hl_width, 1) # note that output is a linear activation as opposed to the rest of the model
        self.network = torch.nn.Sequential(model)
        if (activation=="relu"):
            self.network.apply(init_weights_he)
        elif (activation=="tanh"):
            self.network.apply(init_weights_xav)

    def forward(self, x):
        return self.network(x.type(dtype=torch.float32))

# pretty much directly from https://stackoverflow.com/a/49433937
# this is a function that will be applied to each module in the model (eg. linear, then relu, then linear, then...)
def init_weights_he(m):
    if isinstance(m, torch.nn.Linear): # check to see if the current module is an instance of Linear type
        torch.nn.init.kaiming_uniform_(m.weight)

# pretty much directly from https://stackoverflow.com/a/49433937
# this is a function that will be applied to each module in the model (eg. linear, then tanh, then linear, then...)
def init_weights_xav(m):
    if isinstance(m, torch.nn.Linear): # check to see if the current module is an instance of Linear type
        torch.nn.init.xavier_uniform_(m.weight)

def train(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    model.train()

    train_loss = 0
    for batch, (data, label) in enumerate(dataloader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        prediction = model(data.flatten())
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader)
    return train_loss

def test(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            prediction = model(data.flatten())
            loss = loss_fn(prediction, label)
            test_loss += loss.item()
    test_loss /= len(dataloader)
    return test_loss
            
def importData(filename, column_labels=None, index_col=None, header=None):
    # index_col is a number indicating which column, if any, is the index into the data
    # header is the line of the data if any that where column labels are indicated
    terms = pd.read_csv(filename, sep=',', names=column_labels, index_col=index_col, header=header, dtype=np.float32) # read in the csv file into a DataFrame object , index_col=index_col
    # if needed any processing can be done here
    terms['label'].where(terms['label'] != 0, -1, inplace=True) # change labels to be {-1, 1}
    return terms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    nn_width = [5,10,25,50,100]
    nn_depth = [3,5,9]
    activation = 'tanh'
    EPOCHS = 100

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    train_data = banknote('/bank-note/train.csv')
    train_dataloader = DataLoader(train_data, shuffle=True) # data gets shuffled after all points have been iterated through (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    test_data = banknote('/bank-note/test.csv')
    test_dataloader = DataLoader(test_data, shuffle=True)

    final_train_errors = np.zeros((len(nn_width), len(nn_depth)))
    final_test_errors = np.zeros((len(nn_width), len(nn_depth)))
    for w in range(len(nn_width)):
        for d in range(len(nn_depth)):
            width = nn_width[w]
            depth = nn_depth[d]
            train_errors = np.array([])
            test_errors = np.array([])
            print("Width: " + str(width) + ', Depth: ' + str(depth), end='', flush=True)
            tic = time()
            # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
            model = modelNN(hl_width=width, depth=depth, activation=activation).to(device=device)
            # print(model)
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.MSELoss()
            prev_train_errors = 0
            for i in range(0,EPOCHS):
                train(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
                train_error = test(train_dataloader, model=model, loss_fn=loss_fn)
                test_error = test(test_dataloader, model=model, loss_fn=loss_fn)
                train_errors = np.append(train_errors, train_error)
                test_errors = np.append(test_errors, test_error)
                if i%10 == 0: print('.',end="",flush=True)
                if (abs(prev_train_errors - train_errors[i]) < .001 and i >= 20): break
                prev_train_errors = train_errors[i]
            print(' done, time: ' + str(time() - tic), flush=True)
            f = plt.figure(0)
            plt.plot(range(1,len(train_errors)+1),train_errors, label="Training")
            plt.plot(range(1,len(test_errors)+1),test_errors, label="Testing")
            plt.title("Error for width: " + str(width) + ' and depth: ' + str(depth))
            plt.xlabel("Epochs")
            plt.ylabel("Error (MSE Loss)")
            plt.legend()
            file_name = activation + 'nn_bonus_w' + str(width) + '_d' + str(depth)
            plt.savefig(file_name)
            plt.close(0)
            np.savetxt(file_name+'_train.csv', train_errors, fmt='%.6f', delimiter=',')
            np.savetxt(file_name+'_test.csv', test_errors, fmt='%.6f', delimiter=',')
            final_train_errors[w,d] = train_errors[-1]
            final_test_errors[w,d] = test_errors[-1]
            np.savetxt(file_name+'_final_train_errors.csv', final_train_errors, fmt='%.6f', delimiter=',')
            np.savetxt(file_name+'_final_test_errors.csv', final_test_errors, fmt='%.6f', delimiter=',')
            print(' done, time: ' + str(time() - tic), flush=True)
    return

if __name__ == "__main__":
    main()