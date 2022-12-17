# Written by Alan Felt for CS6350 Machine Learning

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
        'label': {0,1}
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

    training_data = ml.importData(kaggle_train_filepath, attribs, None, index_col=False, numeric_data=numeric_data, empty_indicator='?', change_label=None)
    del attribs['label'] # remove the label as a valid attribute
    test_data = ml.importData(kaggle_test_filepath, attribs, attrib_labels, index_col=0, numeric_data=numeric_data, empty_indicator='?', change_label=None)
    train_error = np.zeros([14,1])
    for max_depth in np.arange(start=1, stop=15, step=1):
        print("Depth: " + str(max_depth)) # print out current max_depth value
        tree = ml.ID3(training_data, attribs, None, 'entropy', max_depth) # build the tree
        pred = pd.DataFrame(ml.processData(tree, test_data),index=test_data.index, columns=['Prediction'])
        train_error[max_depth - 1 ] = ml.treeError(tree, training_data) # test the error for the given dataset
        pred.to_csv("labeled_data_dt_" + str(max_depth),index=True, index_label='ID')
    print(train_error)
    return

if __name__ == "__main__":
    main()
