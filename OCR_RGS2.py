#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from string import ascii_lowercase 
from itertools import product
import random
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt


# In[2]:


def readFile(file_name):
    input_values = []
    temp_array = []
    f = open(file_name)
    for line in f:
        temp_array.append(line)
        if line == '\t\t\t\n':
            input_values.append(temp_array)
            temp_array = []
    return input_values


# In[3]:


def splitInputOutputIndividualLine(file):
    x_values = []
    y_values = []
    split_values_array = []
    for i in range(len(file)-1):
        array_input = np.asarray(file[i])
        string_input = np.array2string(array_input)
        split_values_array = string_input.split('\\t')
        x = split_values_array[1].replace('im','')
        y = split_values_array[2]
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values


# In[4]:


def inputOutputFromFile(file_name):
    x_array = []
    y_array = []
    input_values = readFile(file_name)
    for i in range(len(input_values)):
        individual_line_single_input = [line.split(",") for line in input_values[i]]
        x, y = splitInputOutputIndividualLine(individual_line_single_input)
        y = [element.lower() for element in y]
        x_array.append(x)
        y_array.append(y)
    return x_array, y_array


# In[5]:


def addTwoNumbers(input1,input2):
    c = [a + b for a, b in zip(input1, input2)]
    return np.asarray(c)


# In[6]:


def subtractTwoNumbers(input1,input2):
    c = [a - b for a, b in zip(input1, input2)]
    return np.asarray(c)


# In[7]:


def string2intArray(x):
    convert_to_list = list(x)
    string_to_int = [int(i) for i in convert_to_list]
    return np.asarray(string_to_int)


# In[8]:


#Returns all valid combinations of given word for specificed pairinf
def possibleCombinationsOfGivenWordForNPairing(word_array, n):
    word = ''.join(word_array)
    #Take the relevant substrings
    word_substrings = []
    for i in range(n):
        string = word[i:len(word)]
        word_substrings.append(string)
   
    #Split a substring into individual characters
    word_substrings_character = []
    for i in range(len(word_substrings)):
        split_words = list(word_substrings[i]) 
        word_substrings_character.append(split_words)
        
    #combine all the elements in ith index
    combination_array = list(zip(*word_substrings_character))
    possible_combinations = [''.join(combination_array[i]) for i in range(len(combination_array))]   
    return possible_combinations


# In[9]:


def nPairWiseFeatureVector(word_array, n, features_size):
    features = [0] * features_size
    possible_labels = []

    for combination in product(ascii_lowercase, repeat = n): #n=2 aa ab... n=3 aaa aab...
        possible_labels.append(''.join(combination))   

    possible_word_combinations = possibleCombinationsOfGivenWordForNPairing(word_array, n) #n=2 om mm... n=3 omm mma...

    for word_combo in possible_word_combinations: #Replace 1 in the array whereever we see the word combos from above
        index = possible_labels.index(word_combo)
        features[index] = 1
        
    return features


# In[10]:


def unaryFeatures(x, y, possible_labels, number_of_possible_labels):
    unary_features_size = len(x[0]) * number_of_possible_labels
    unary_features = [0] * unary_features_size
    unary_features_split = np.array_split(unary_features, number_of_possible_labels)
    for j in range(len(y)):
        input_value = x[j]
        label = y[j] 
        index_of_letter = possible_labels.find(label)
        temporay = unary_features_split[index_of_letter]
        if np.all(temporay == 0): #Check if array is having only zeros or not
            unary_features_split[index_of_letter] = string2intArray(input_value) #Add the new array to it 
        else: #Add the existing binary valyes with the new binary values
            unary_features_split[index_of_letter] = addTwoNumbers(temporay, string2intArray(input_value))
    unary_features = np.concatenate(unary_features_split)
    unary_features = list(unary_features)
    return unary_features


# In[11]:


def pairWiseFeatures(number_of_possible_labels, y):
    pairwise_features_size = number_of_possible_labels ** 2 # Eg: 26^2 if n=2
    pairwise_features = nPairWiseFeatureVector(y, 2, pairwise_features_size)
    return pairwise_features


# In[12]:


def tripleFeatures(number_of_possible_labels, y):
    triple_features_size = number_of_possible_labels ** 3 # Eg: 26^3 if n=3
    triple_features = nPairWiseFeatureVector(y, 3, triple_features_size)
    return triple_features


# In[13]:


def quadrupleFeatures(number_of_possible_labels, y):
    quadruple_features_size = number_of_possible_labels ** 4 # Eg: 26^3 if n=3
    quadruple_features = nPairWiseFeatureVector(y, 4, quadruple_features_size)
    return quadruple_features


# In[14]:


def returnSizeBasedOnFeatures(x, number_of_possible_labels, n):
    if n==1:
        return len(x[0][0]) * number_of_possible_labels
    elif n==2:
        return len(x[0][0]) * number_of_possible_labels + number_of_possible_labels**2
    elif n==3:
        return len(x[0][0]) * number_of_possible_labels + number_of_possible_labels**2 + number_of_possible_labels**3
    else:
        return len(x[0][0]) * number_of_possible_labels + number_of_possible_labels**2 + number_of_possible_labels**3 + number_of_possible_labels**4


# In[15]:


def featureFunction(n, x, y, possible_labels, number_of_possible_labels):
    if n==1:
        return unaryFeatures(x, y, possible_labels, number_of_possible_labels)
    elif n==2:
        return unaryFeatures(x, y, possible_labels, number_of_possible_labels) + pairWiseFeatures(number_of_possible_labels, y)
    elif n==3:
        return unaryFeatures(x, y, possible_labels, number_of_possible_labels) + pairWiseFeatures(number_of_possible_labels, y) + tripleFeatures(number_of_possible_labels, y)
    else:
        return unaryFeatures(x, y, possible_labels, number_of_possible_labels) + pairWiseFeatures(number_of_possible_labels, y) + tripleFeatures(number_of_possible_labels, y) + quadrupleFeatures(number_of_possible_labels, y)


# In[41]:


def plot_learning_curve(loss_1, loss_2, loss_3, loss_4, iterations, type_of_plot, save_file_name):
    if loss_1: plt.plot(iterations, loss_1, color = 'red', marker='o', linestyle='solid', label='Unary features')
    if loss_2: plt.plot(iterations, loss_2, color = 'green', marker='o', linestyle='solid', label='First Order')
    if loss_3: plt.plot(iterations, loss_3, color = 'blue', marker='o', linestyle='solid', label='Second Order')
    if loss_4: plt.plot(iterations, loss_4, color = 'orange', marker='o', linestyle='solid', label='Third Order')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title("Learning curve for %s" %(type_of_plot))
    plt.savefig(save_file_name)
    plt.legend()
    plt.show()


# In[42]:


def plot_accuracy_curve(acc_1, acc_2, acc_3, acc_4, iterations, type_of_plot, save_file_name):
    if acc_1: plt.plot(iterations, acc_1, color = 'red', marker='o', linestyle='solid', label='Unary features')
    if acc_2: plt.plot(iterations, acc_2, color = 'green', marker='o', linestyle='solid', label='First Order')
    if acc_3: plt.plot(iterations, acc_3, color = 'blue', marker='o', linestyle='solid', label='Second Order')
    if acc_4: plt.plot(iterations, acc_4, color = 'orange', marker='o', linestyle='solid', label='Third Order')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title("Accuracy curve for %s" %(type_of_plot))
    plt.savefig(save_file_name)
    plt.legend()
    plt.show()


# In[18]:


def getHammingScores(y_pred, y_true):
    losses = []
    accuracies = []
    for index in range(len(y_pred)):
        loss = hamming_loss(y_true[index], y_pred[index])
        accuracy = 1 - loss
        losses.append(loss)
        accuracies.append(accuracy)
    return sum(losses)/float(len(y_pred)), sum(accuracies)/float(len(y_pred))


# In[19]:


def rgsInference(n, x, y, w, repeats, possible_labels, number_of_possible_labels):
    y_hat_random = []
    s_max = -1
    for i in range(len(y)):
        y_letter_random = secure_random.choice(possible_labels)
        y_hat_random.append(y_letter_random)
    phi_y_random = featureFunction(n, x, y_hat_random, possible_labels, number_of_possible_labels)
    s_best = np.dot(w, phi_y_random)
    for i in range(repeats):
        random.shuffle(y_hat_random)
        s_array = []
        y_array = []
        y_start = y_hat_random.copy()
        index_random = secure_random.choice(list(range(0,len(y_start))))
        for letter in possible_labels:
            y_temp = y_start.copy()
            y_temp[index_random] = letter
            phi_y_temp = featureFunction(n, x, y_temp, possible_labels, number_of_possible_labels)
            s_value = np.dot(w, phi_y_temp)
            s_array.append(s_value)
            y_array.append(y_temp)
        best_index = np.argmax(s_array)    
        s_max = s_array[best_index]
        y_start = y_array[best_index].copy()
        s_array.clear()
        y_array.clear()
        s_best = s_max
    return y_start


# In[20]:


def rgsTrainingAlgorithm(n, x, y, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels):
    losses = []
    accuracies = []
    y_pred = []
    y_true = []
    mistake = 0
    mistakes = []
    feature_size = returnSizeBasedOnFeatures(x, number_of_possible_labels, n)
    w = np.zeros(feature_size)
    for i in range(max_iterations):
        for example_index in range(len(x)):
            y_hat = rgsInference(n, x[example_index], y[example_index], w, repeats, possible_labels, number_of_possible_labels)
            if y_hat != y[example_index]:
                phi_y_hat = featureFunction(n, x[example_index], y_hat, possible_labels, number_of_possible_labels)
                phi_y = featureFunction(n, x[example_index], y[example_index], possible_labels, number_of_possible_labels)
                w = addTwoNumbers(w , learning_rate * (subtractTwoNumbers(phi_y, phi_y_hat)))
                mistake = mistake + 1
            y_pred.append(y_hat)
            y_true.append(y[example_index])
        loss, accuracy = getHammingScores(y_pred, y_true)
        losses.append(loss)
        mistakes.append(mistake)
        accuracies.append(accuracy)
        y_pred.clear()
        y_true.clear()
        mistake = 0
    return losses, accuracies, mistakes, w


# In[21]:


def rgsTestingAlgorithm(n, x, y, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels):
    losses = []
    accuracies = []
    y_pred = []
    y_true = []
    mistakes = []
    mistake = 0
    feature_size = returnSizeBasedOnFeatures(x, number_of_possible_labels, n)
    w = np.zeros(feature_size)
    for i in range(max_iterations):
        for example_index in range(len(x)):
            y_hat, y_rand = rgsInference(n, x[example_index], y[example_index], w, repeats, possible_labels, number_of_possible_labels)
            if y_hat != y[example_index]:
                mistake = mistake + 1
            y_pred.append(y_hat)
            y_true.append(y[example_index])
        loss, accuracy = getHammingScores(y_pred, y_true)
        losses.append(loss)
        mistakes.append(mistake)
        accuracies.append(accuracy)
        y_pred.clear()
        y_true.clear()
        mistake = 0
    return losses, accuracies, mistakes


# In[34]:


file_name_train = "ocr_fold0_sm_train.txt"
x_train, y_train = inputOutputFromFile(file_name_train)
file_name_test = "ocr_fold0_sm_test.txt"
x_test, y_test = inputOutputFromFile(file_name_test)


# In[24]:


max_iterations = 20
repeats = 10
learning_rate = 0.01
secure_random = random.SystemRandom()


# In[25]:


possible_labels = ascii_lowercase
number_of_possible_labels = 26


# In[26]:


losses_1,accuracies_1, mistakes_1, w_1 = rgsTrainingAlgorithm(1, x_train, y_train, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)


# In[27]:


losses_2,accuracies_2, mistakes_2, w_2 = rgsTrainingAlgorithm(2, x_train, y_train, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)


# In[28]:


losses_3,accuracies_3, mistakes_3, w_3 = rgsTrainingAlgorithm(3, x_train, y_train, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)


# In[36]:


losses_4,accuracies_4, mistakes_4, w_4 = rgsTrainingAlgorithm(4, x_train, y_train, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)


# In[49]:


iterations = [x+1 for x in list(range(max_iterations))]
plot_learning_curve(losses_1, losses_2, losses_3, losses_4, iterations, "OCR RGS Inference Training", "lcRgsTraining2")
plot_accuracy_curve(accuracies_1, accuracies_2, accuracies_3, accuracies_4, iterations, "OCR RGS Inference Training", "acRgsTraining2")


# In[ ]:


losses_1,accuracies_1, mistakes_1, w_1 = rgsTestingAlgorithm(1, x_test, y_test, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)
losses_2,accuracies_2, mistakes_2, w_2 = rgsTestingAlgorithm(2, x_test, y_test, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)
losses_3,accuracies_3, mistakes_3, w_3 = rgsTestingAlgorithm(3, x_test, y_test, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)
losses_4,accuracies_4, mistakes_4, w_4 = rgsTestingAlgorithm(4, x_test, y_test, max_iterations, repeats, learning_rate, number_of_possible_labels, possible_labels)


# In[68]:


iterations = [x+1 for x in list(range(max_iterations))]
plot_learning_curve(losses_1, losses_2, losses_3, losses_4, iterations, "OCR RGS Inference Testing", "lcRgsTesting2")
plot_accuracy_curve(accuracies_1, accuracies_2, accuracies_3, accuracies_4, iterations, "OCR RGS Inference Testing", "acRgsTesting2")

