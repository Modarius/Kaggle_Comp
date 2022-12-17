# Written by Alan Felt for CS6350 Machine Learning

from copy import deepcopy
import numpy as np
import pandas as pd
import MLib as ml

def main():
    attrib_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 
   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'workclass':{'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                    'State-gov', 'Without-pay', 'Never-worked'
                    },
        'fnlwgt':{'low', 'high'},
        'education':{'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                    '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'},
        'education_num':{'low', 'high'},
        'marital_status': {'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'},
        'occupation': {'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'},
        'relationship': {'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'},
        'race': {'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'},
        'sex': {'Female', 'Male'},
        'capital_gain': {'low','high'},
        'capital_loss': {'low','high'},
        'hours_per_week': {'low','high'},
        'native_country': { 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                            'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
                            'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                            'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong'},
        'label': {-1,1}
        }
    kaggle_train_filepath = 'train_final.csv'
    kaggle_test_filepath = 'test_final.csv'
    numeric_data = {
        'age':['young', 'old'],
        'fnlwgt':['low', 'high'],
        'education_num':['low', 'high'],
        'capital_gain': ['low','high'],
        'capital_loss': ['low','high'],
        'hours_per_week': ['low','high']
        }
    new_labels = {0: -1, 1: 1}

    training_data = ml.importData(kaggle_train_filepath, attribs, attrib_labels = None, index_col=False, numeric_data=numeric_data, empty_indicator='?', change_label=new_labels)
    del attribs['label'] # remove the label as a valid attribute
    test_data = ml.importData(kaggle_test_filepath, attribs, attrib_labels, index_col=0, numeric_data=numeric_data, empty_indicator='?', change_label=None)
    train_error = np.zeros([50,1])
    bT = None
    eh = deepcopy(training_data)
    for iter in np.arange(start=0, stop=50, step=1):
        print("iteration: " + str(iter)) # print out current max_depth value
        bT = ml.baggedDecisionTree(training_data, attribs=attribs, T=1, m=50, prev_ensemble=bT)
        pred = pd.DataFrame(bT.HFinal(test_data),index=test_data.index, columns=['Prediction'])
        train_error[iter] = sum(training_data['label'] != bT.HFinal(training_data)) / len(training_data) # test the error for the given dataset
        pred.to_csv("labeled_data_bT_tree_" + str(iter+1) + ".csv",index=True, index_label='ID')
    print(train_error)
    return

if __name__ == "__main__":
    main()
