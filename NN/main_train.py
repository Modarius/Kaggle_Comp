# Written by Alan Felt for CS6350 Machine Learning

from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
import matplotlib.pyplot as plt

## https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class income_info(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        curr_example = self.data.iloc[index]
        y = curr_example.pop('income_50K').astype(np.float32)
        x = curr_example.to_numpy().squeeze().astype(np.float32)
        return x, y

class modelNN(torch.nn.Module):
    def __init__(self, input_width, hl_width, depth, activation="relu"):
        super(modelNN, self).__init__()
        
        #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential
        model = OrderedDict()
        model["linear_input"] = torch.nn.Linear(input_width, hl_width)
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
    wrong = 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            prediction = model(data.flatten())
            loss = loss_fn(prediction, label)
            test_loss += loss.item()
            wrong += abs(label.item() - round(prediction.item()))
    test_loss /= len(dataloader)
    test_error = wrong/len(dataloader)
    return test_loss, test_error

# input should be a single column of S
def bestValue(S, empty_indicator=None):
    l, c = np.unique(S.to_numpy(), return_counts=True) # find the most comon value in attribute A
    if (empty_indicator != None):
        # this is a hacky way of getting the index of 'unknown' in l and c (unique labels, and their counts)
        idx = np.squeeze(np.where(l == empty_indicator))[()] # https://thispointer.com/find-the-index-of-a-value-in-numpy-array/, https://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar
        l = np.delete(l, idx) # remove unknown from the running for most common value
        c = np.delete(c, idx) # remove unknown from the running for most common value
    best_value = l[c.argmax()] # find the most common value (index into L with the index of the largest # in c)
    return best_value

def getPWeights(S):
    labels = S['label'].unique()
    summed_weight = np.zeros(len(labels))
    for i in np.arange(len(labels)):
        summed_weight[i] = S[S['label'] == labels[i]]['weight'].sum()
    return dict(zip(labels,summed_weight)) # https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/

def bestLabel(S):
    pWeight = getPWeights(S)
    return max(zip(pWeight.values(), pWeight.keys()))[1] # https://www.geeksfor .org/python-get-key-with-maximum-value-in-dictionary/

def validData(terms, discrete_attribs, attrib_values):
    for attrib in discrete_attribs: # for the attributes with discrete values
        unique_attrib_values = set(terms.get(attrib).unique()) # get the set of values for attribute A
        if (not attrib_values[attrib].issubset(unique_attrib_values)): # check if there is a value from the data not included in the possible values for A
            # print the offending invalid attribute value
            print("Attribute " + attrib + " cannot take value " +
                  str(unique_attrib_values.difference(attrib_values[attrib])))
            return False
    # if (not set(terms.index.unique().to_numpy()).issubset(data_labels)): # also check that all the labels are valid
    #     print("Data Label cannot take value " +
    #           str(set(terms.index.unique()).difference(data_labels))) # return values that are not valid
    #     return False
    return True

def importData(filename, attrib_values, attribs, discrete_attribs, index_col=None, empty_indicator=None, change_label=None):
    terms = pd.read_csv(filename, sep=',', names=attribs, index_col=index_col, header=0) # read in the csv file into a DataFrame object , index_col=index_col
    
    if (empty_indicator != None):
        for label in terms.columns.to_numpy():
            if(terms[label].unique().__contains__(empty_indicator)): # if the column contains unknown values
                column = terms[label] # get that column
                best_value = bestValue(terms[label], empty_indicator)
                terms[label].where(column != empty_indicator, best_value, inplace=True) # when column2 doesnt equal indicator, keep it as is, else replace indicator with most common value
    
    if (change_label != None):
        for raw_label in change_label.keys():
            terms['label'].where(terms['label'] != raw_label, change_label[raw_label], inplace=True)
    
    if (not validData(terms, discrete_attribs, attrib_values)): # check for incorrect attribute values
        return
    
    return terms

def convertDiscrete(S, discrete_attribs, attrib_values):
    for attrib in discrete_attribs: # for each attribute
        for value in attrib_values[attrib]: # for each value in the attribute
            S[attrib+"_"+value] = np.where(S[attrib] == value, 1, 0)
        S.drop([attrib], axis='columns', inplace=True) # remove the offending attribute
    return S

def normalizeContinous(S, continous_attribs):
    for attrib in continous_attribs:
        max_val = S[attrib].max()
        min_val = S[attrib].min()
        diff_val = max_val - min_val
        query = attrib + " = (" + attrib + " - " + str(min_val) + ") / " + str(diff_val) # https://www.statology.org/normalize-data-between-0-and-1/
        S.eval(query, inplace=True)
    return S

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    discrete_attribs = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    continous_attribs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    attrib_values = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'workclass':{'Private', 'Self_emp_not_inc', 'Self_emp_inc', 'Federal_gov', 'Local_gov',
                    'State_gov', 'Without_pay', 'Never_worked'
                    },
        'fnlwgt':{'low', 'high'},
        'education':{'Bachelors', 'Some_college', '11th', 'HS_grad', 'Prof_school', 'Assoc_acdm', 'Assoc_voc', '9th',
                    '7th_8th', '12th', 'Masters', '1st_4th', '10th', 'Doctorate', '5th_6th', 'Preschool'},
        'education_num':{'low', 'high'},
        'marital_status': {'Married_civ_spouse', 'Divorced', 'Never_married', 'Separated', 'Widowed', 'Married_spouse_absent', 'Married_AF_spouse'},
        'occupation': {'Tech_support', 'Craft_repair', 'Other_service', 'Sales', 'Exec_managerial', 'Prof_specialty', 'Handlers_cleaners',
                      'Machine_op_inspct', 'Adm_clerical', 'Farming_fishing', 'Transport_moving', 'Priv_house_serv', 'Protective_serv', 'Armed_Forces'},
        'relationship': {'Wife', 'Own_child', 'Husband', 'Not_in_family', 'Other_relative', 'Unmarried'},
        'race': {'White', 'Asian_Pac_Islander', 'Amer_Indian_Eskimo', 'Other', 'Black'},
        'sex': {'Female', 'Male'},
        'capital_gain': {'low','high'},
        'capital_loss': {'low','high'},
        'hours_per_week': {'low','high'},
        'native_country': { 'United_States', 'Cambodia', 'England', 'Puerto_Rico', 'Canada', 'Germany', 'Outlying_US(Guam_USVI_etc)', 'India', 'Japan',
                            'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
                            'Portugal', 'Ireland', 'France', 'Dominican_Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                            'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El_Salvador', 'Trinadad_Tobago', 'Peru', 'Hong'},
        'label': {0,1}
        }
    kaggle_train_filepath = 'train_final.csv'
    
    pd_training_data = importData(kaggle_train_filepath, attrib_values=attrib_values, attribs=None, discrete_attribs=discrete_attribs, index_col=False, empty_indicator='?', change_label=None)
    del attrib_values['label'] # remove the label as a valid attribute
    pd_training_data = convertDiscrete(pd_training_data, discrete_attribs=discrete_attribs, attrib_values=attrib_values)
    pd_training_data = normalizeContinous(pd_training_data, continous_attribs=continous_attribs)

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    nn_training_data = income_info(pd_training_data)
    nn_train_dataloader = DataLoader(nn_training_data, shuffle=True) # data gets shuffled after all points have been iterated through (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

    nn_input_width = len(pd_training_data.columns -1) # -1 column for the label
    nn_width = [10]
    nn_depth = [5]
    EPOCHS = 100
    activation = 'tanh'

    final_train_errors = np.zeros((len(nn_width), len(nn_depth)))
    for w in range(len(nn_width)):
        for d in range(len(nn_depth)):
            width = nn_width[w]
            depth = nn_depth[d]
            train_errors = np.array([])
            print("Width: " + str(width) + ', Depth: ' + str(depth), end='', flush=True)
            tic = time()
            # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
            model = modelNN(input_width=nn_input_width, hl_width=width, depth=depth, activation=activation).to(device=device)
            # print(model)
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.MSELoss()
            prev_train_errors = 0
            for i in range(0,EPOCHS):
                train(nn_train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
                train_loss, train_error = test(nn_train_dataloader, model=model, loss_fn=loss_fn)
                train_errors = np.append(train_errors, train_error)
                if i%10 == 0: print('.',end="",flush=True)
                #if (abs(prev_train_errors - train_errors[i]) < .001 and i >= 20): break
                prev_train_errors = train_errors[i]
            print(' done, time: ' + str(time() - tic), flush=True)
            f = plt.figure(0)
            plt.plot(range(1,len(train_errors)+1),train_errors, label="Training")
            plt.title("Error for width: " + str(width) + ' and depth: ' + str(depth))
            plt.xlabel("Epochs")
            plt.ylabel("Error (MSE Loss)")
            plt.legend()
            file_name = activation + '_nn_w' + str(width) + '_d' + str(depth)
            plt.savefig(file_name)
            plt.close(0)
            np.savetxt(file_name + '_train.csv', train_errors, fmt='%.6f', delimiter=',')
            final_train_errors[w, d] = train_errors[-1]
            print(' done, time: ' + str(time() - tic), flush=True)
            torch.save(model, "w" + str(width) + "_d" + str(depth) + ".pth")
    np.savetxt('final_train_errors.csv', final_train_errors, fmt='%.6f', delimiter=',')
    return


if __name__ == "__main__":
    main()
