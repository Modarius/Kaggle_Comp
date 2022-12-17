# Written by Alan Felt for CS6350 Machine Learning

from copy import deepcopy
import numpy as np
import pandas as pd


# for leaf nodes, name is the same as the branch it is connected to
# for regular nodes, name is the name of the attribute it represents. Its children will be named the values of the attributes values
class Node:
    def __init__(self, name, type, parent, children, label, depth, weight = 1):
        self.name = name  # name of the node
        self.type = type  # 'root', 'node', 'leaf', 'unknown'
        self.parent = parent  # will include a node (if not the root)
        self.children = children  # will include node(s) instance
        self.label = label
        self.depth = depth
        self.weight = weight
        return

    def setChild(self, child_name_in, child_node_in):
        if self.children is None: # if there are no children, add one in dictionary form
            self.children = {child_name_in: child_node_in}
        elif child_name_in not in self.children: # otherwise check if the child is already there
            self.children[child_name_in] = child_node_in
        else: # this statement should never run but is there just in case something goes wrong
            print("could not add" + child_name_in)
        return

    def getDepth(self):
        return self.depth

    def getName(self):
        return self.name
    
    def getLabel(self):
        return self.label

    def getType(self):
        return self.type

    def getChildren(self):
        return self.children

    def getChild(self, child_name_in):
        if child_name_in not in self.children:
            return None # this should never be called
        else:
            return self.children[child_name_in] # return the child node with the attribute provided in child_name_in
    
    def getWeight(self):
        return self.weight


def getPWeights(S):
    labels = S['label'].unique()
    summed_weight = np.zeros(len(labels))
    for i in np.arange(len(labels)):
        summed_weight[i] = S[S['label'] == labels[i]]['weight'].sum()
    return dict(zip(labels,summed_weight)) # https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/


def entropy(S):
    pWeights = getPWeights(S)
    pTotal = sum(pWeights.values())
    p = list(pWeights.values()) / pTotal
    H_S = -np.sum(p * np.log2(p)) # sum of the probabilites multiplied by log2 of probabilities
    return H_S

# untested with weights but should be working
def majorityError(S): 
    pWeights = getPWeights(S)
    pTotal = sum(pWeights.values())

    best_choice = max(zip(pWeights.values(), pWeights.keys()))[1]
    del pWeights[best_choice]
    # delete the count of the label with the greatest representation
    # sum up the number of remaining labels
    neg_choice = np.sum(pWeights.values())

    # calculate ratio of # of not best labels over the total number of labels
    m_error = neg_choice / pTotal

    # return this number
    return m_error


def giniIndex(S):
    pWeights = getPWeights(S)
    pTotal = sum(pWeights.values())
    p_l = list(pWeights.values) / pTotal # calculate the probability of each label
    gi = 1 - np.sum(np.square(p_l)) # square and sum the probabilities
    return gi


def bestAttribute(S, attribs, method='entropy', rand_tree=False):
    if(rand_tree & S.columns.size > 2): # has more than the label and weight columns, eg, has an attribute to split on
        T = S[attribs.keys()]
        T = T.sample(frac=1/10, replace=False, axis='columns')
        attribs = {key:attribs[key] for key in T.columns.to_numpy()}
        T['label'] = S['label']
        T['weight'] = S['weight']
        S = T

    if (method == 'entropy'):
        Purity_S = entropy(S)
    elif (method == 'gini'):
        Purity_S = giniIndex(S)
    elif (method == 'majority_error'): # choose method to use
        Purity_S = majorityError(S)
    else:
        print("Not a valid method")
        return
    
    num_S = np.size(S, 0) # number of entries in S
    ig = dict() # initialize a dictionary
    best_ig = 0 # track the best information gain
    best_attribute = "" # track the Attribute of the best information gain

    for A in attribs:  # for each attribute in S except for the label https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-in-pandas
        total = 0
        # get the unique values that attribute A has in S
        values_A = S.get(A).unique()
        for v in values_A:  # for each of those values
            # select a subset of S where S[A] equals that value of A
            Sv = S[S[A] == v]
            # get the size of the subset (number of entries)
            num_Sv = np.size(Sv, 0)
            if (method == 'entropy'):
                Purity_Sv = entropy(Sv)
            elif (method == 'gini'):
                Purity_Sv = giniIndex(Sv)
            elif (method == 'majority_error'): # choose method to use
                Purity_Sv = majorityError(Sv)
            else:
                print("Not a valid method")
                return
            # sum the weighted values of each purity for v in A
            total = total + num_Sv/num_S * Purity_Sv
        # subtract the sum from the purity of S to get the information gain
        ig[A] = Purity_S - total
        if (ig[A] >= best_ig):  # if that information gain is better than the others, select that attribute as best
            best_attribute = A
            best_ig = ig[A]
    if (best_attribute == ""): # handle edge case where no value is best, in that case just return what is in A (should be one value)
        best_attribute = A
    # once we have checked all attributes A in S, return the best attribute to split on
    return best_attribute


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


def bestLabel(S):
    pWeight = getPWeights(S)
    return max(zip(pWeight.values(), pWeight.keys()))[1] # https://www.geeksfor .org/python-get-key-with-maximum-value-in-dictionary/


def validData(terms, attrib):
    for label in attrib.keys(): # for all the attributes possible
        label_values = set(terms.get(label).unique()) # get the set of values for attribute A
        if (not attrib[label].issubset(label_values)): # check if there is a value from the data not included in the possible values for A
            # print the offending invalid attribute value
            print("Attribute " + label + " cannot take value " +
                  str(label_values.difference(attrib[label])))
            return False
    # if (not set(terms.index.unique().to_numpy()).issubset(data_labels)): # also check that all the labels are valid
    #     print("Data Label cannot take value " +
    #           str(set(terms.index.unique()).difference(data_labels))) # return values that are not valid
    #     return False
    return True


def importData(filename, attrib, attrib_labels, index_col=None, numeric_data=None, empty_indicator=None, change_label=None):
    terms = pd.read_csv(filename, sep=',', names=attrib_labels, index_col=index_col, header=0) # read in the csv file into a DataFrame object , index_col=index_col
    if (numeric_data != None): # if there is information on which columns are numeric
        for label in numeric_data.keys(): #the for all the labels in numeric data
            column = terms.get(label) # get the column pertaining to that label
            new_column = column.copy(deep=True) # make a second copy, but not a linked copy
            split_value = np.median(column.to_numpy()) # find the median numeric value to split on
            # find all the values in the column that are less than and equal to 
            # and greater than and replace them with a label from numeric_data
            new_column.where(column <= split_value, numeric_data[label][0], inplace=True) 
            new_column.where(column > split_value, numeric_data[label][1], inplace=True)
            terms[label] = new_column # replace the column with the updated one
    if (empty_indicator != None):
        for label in terms.columns.to_numpy():
            if(terms[label].unique().__contains__(empty_indicator)): # if the column contains unknown values
                column = terms[label] # get that column
                best_value = bestValue(terms[label], empty_indicator)
                terms[label].where(column != empty_indicator, best_value, inplace=True) # when column2 doesnt equal indicator, keep it as is, else replace indicator with most common value
    if (change_label != None):
        for raw_label in change_label.keys():
            terms['label'].where(terms['label'] != raw_label, change_label[raw_label], inplace=True)
    if (not validData(terms, attrib)): # check for incorrect attribute values
        return
    D = np.ones(len(terms))
    weight = pd.DataFrame(D, columns=['weight'])
    weightS = terms.join(weight)
    return weightS

# assumes that there is at least one attribute to split on
def stump(S, attribs, method="entropy"):
    A = bestAttribute(S, attribs=attribs, method=method)
    new_root = Node(A, "root", None, None, None, 0)
    for v in attribs[A]:
        Sv = S[S[A] == v].drop(A, axis=1) # find the subset where S[A] == v and drop the column A
        if (Sv.index.size == 0):  # if the subset is empty, make a child with the best label in S
            label = bestLabel(S)
        else:
            label = bestLabel(Sv)
        v_child = Node(name=v, type="leaf", parent=new_root, children=None, label=label, depth=1)
        new_root.setChild(v, v_child) # set a new child of new_root
    return new_root


def ID3(S, attribs, root=None, method="entropy", max_depth=np.inf, rand_tree=False):
    # Check if all examples have one label
    # Check whether there are no more attributes to split on
    # if so make a leaf node
    if (S['label'].unique().size == 1 or len(attribs) == 0): 
        label = bestLabel(S)
        return Node(label, "leaf", None, None, label, root.getDepth() + 1)

    A = bestAttribute(S, attribs=attribs, method=method, rand_tree=rand_tree) # get the best attribute to split on
    attribsv = deepcopy(attribs)
    del attribsv[A] # delete the key that we are now splitting on

    if (root == None): # if there is no root, make one
        new_root = Node(A, "root", None, None, None, 0)
    else:
        new_root = Node(A, "node", None, None, None, root.getDepth() + 1)

    # v is the branch, not a node, unless v splits the dataset into one with no subsets
    for v in attribs[A]:
        Sv = S[S[A] == v].drop(A, axis=1) # find the subset where S[A] == v and drop the column A
        if (Sv.index.size == 0):  # if the subset is empty, make a child with the best label in S
            v_child = Node(v, "leaf", None, None, bestLabel(S), new_root.getDepth() + 1)
        elif (new_root.getDepth() == (max_depth - 1)): # if we are almost at depth, truncate and make a child with the best label in the subset
            #print("At depth, truncating")
            v_child = Node(v, "leaf", None, None, bestLabel(Sv), new_root.getDepth() + 1)
        else:  # if the subset is not empty make a child with the branch v but not the node name v, node name will be best attribute found for splitting Sv
            v_child = ID3(Sv, attribsv, new_root, method, max_depth, rand_tree=rand_tree) # recursive call down the tree
        new_root.setChild(v, v_child) # set a new child of new_root
    return new_root


def follower(data, tree):
    if (tree.getType() != 'leaf'): # if we're not at a leaf
        v = data.pop(tree.getName()) # pop off the value from data which corresponds to the name of the current node
        return follower(data, tree.getChild(v)) # recursive call down the tree
    else:
        return tree.getLabel() # if we're at a leaf, return the label of the leaf


# this is a kludgy way of printing the tree
# referenced https://simonhessner.de/python-3-recursively-print-structured-tree-including-hierarchy-markers-using-depth-first-search/
def printTree(tree):
    ttype = tree.getType() # get all the data for the current node
    tname = tree.getName()
    tlabel = tree.getLabel()
    tchildren = tree.getChildren()
    print('\t' * tree.getDepth(), end='') # pad the name by depth # of tab characters

    if(ttype == "leaf"): # if we're a leaf, print the label with || around it
        print('|' + str(tlabel) + '|')
        return
    elif(ttype == "root" or ttype == "node"): # if we're a root or node, just print the name
        print(tname)
    for c in tchildren: # for each of the children of the current node
        print('\t' * tree.getDepth() + '-> ' + c) # pad the child name by depth tabs and a ->
        printTree(tchildren[c]) # recursively call this function
    return


def treeError(tree, S):
    c_right = 0
    c_wrong = 0
    for data in S.itertuples(): # for each datapoint in S
        prediction = follower(data._asdict(), tree)
        if (data.label != prediction): # check if the label returned by the tree is the same as the provided label
            c_wrong += 1
            #print("not a match")
        else:
            c_right += 1
            #print("matched!")
        
    error = c_wrong / (c_right + c_wrong) # find the ratio of wrong answers to all answers (error)
    return error


def processData(tree, S):
    ht_xi = np.zeros(S.index.size)
    for data in S.itertuples(index=True): # for each datapoint in S
        ht_xi[data.Index-1] = follower(data._asdict(), tree)
    return ht_xi
        
class Ensemble:
    def __init__(self, weights, trees):
        self.weights = weights # numpy array of weights for each tree
        self.trees = trees # numpy array of root nodes
        return
    def HFinal(self, S):
        num_S = len(S)
        out = np.empty(num_S, dtype=int)
        data = S.to_dict(orient='records')
        for s_idx in np.arange(num_S): # for each data point in S
            ht_x = np.zeros(len(self.trees))
            for t_idx in np.arange(len(self.trees)): # for each tree, get its hypothesis on the current data
                in_data = deepcopy(data[s_idx])
                ht_x[t_idx] = follower(in_data, self.trees[t_idx]) # save the hypothesis
            out[s_idx] = np.sign(np.sum(self.weights * ht_x))
        return out

def adaBoost(S, attribs, T):
    S = deepcopy(S)
    m = len(S)
    D = np.ones(m) * 1/m
    S['weight'] = D
    h_t = np.empty(T, dtype=Node)
    alpha_t = np.zeros(T)
    for t in np.arange(T):
        h_t[t] = stump(S=S, attribs=attribs, method='entropy')
        h_t_xi = processData(tree=h_t[t], S=S)
        e_t = sum(D[h_t_xi != S['label']])
        alpha_t[t] = 1/2 * np.log((1-e_t)/e_t)
        inner = (-alpha_t[t] * S['label'].to_numpy() * h_t_xi)
        D_1 = D * np.exp(inner.astype(float)) # https://stackoverflow.com/questions/47966728/how-to-fix-float-object-has-no-attribute-exp
        D = D_1 / sum(D_1)
        S['weight'] = D
    return Ensemble(alpha_t, h_t)

def baggedDecisionTree(S, attribs, T, m):
    trees = np.empty(T, dtype=Node)
    for t in np.arange(T):
        bag = S.sample(m, replace=True)
        trees[t] = ID3(bag, attribs)
    return Ensemble(np.ones(T), trees)

def randomTree(S, attribs, T, m):
    trees = np.empty(T, dtype=Node)
    for t in np.arange(T):
        bag = S.sample(m, replace=True)
        trees[t] = ID3(bag, attribs, rand_tree=True)
    return Ensemble(np.ones(T), trees)