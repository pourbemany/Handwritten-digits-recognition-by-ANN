import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import copy
from sklearn.metrics import mean_squared_error

def convert_to_csv(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
        
    for image in images:
        list = [str(pix) for pix in image]
        o.write(",".join(list)+"\n")
    f.close()
    o.close()
    l.close()

def svm_classifing(X_train,y_train,X_test,y_test,kernel_type):
    start_time = time.time()
    if kernel_type == 'poly':
        svclassifier = SVC(kernel=kernel_type, degree=8)
    else:
        svclassifier = SVC(kernel=kernel_type)                
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    print("*********************************************************************************")
    print("Traning time for SVM %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_train)), (time.time() - start_time)))
    print("*********************************************************************************")
    print("score = %3.2f" %svclassifier.score(X_test,y_test))
    
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    return y_pred

def show_samples(X_test, y_pred, rand_list, name):
    for i in rand_list:
        two_d = (np.reshape(X_test.values[i], (28, 28))).astype(np.uint8)
        plt.title('predicted label: {0}'. format(y_pred[i]))
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        plt.savefig('fig/'+name+'_'+str(i)+'.jpg')
        plt.show()

def show_mse(mse, name):
    plt.plot(mse)
    plt.title('Mean Square Error: '+name)
    plt.ylabel('Mean Square Error')
    plt.xlabel('iteration')
    plt.savefig('fig/'+'Mean Square Error_'+name+'.jpg')
    plt.show()
    
def show_digits_error(y_test, y_pred, name):
    y_test=list(y_test)
    correct_guess = [0] * 10
    incorrect_correct_guess = [0] * 10
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            incorrect_correct_guess[y_test[i]] += 1
        else:
            correct_guess[y_test[i]] += 1
            
    N = 10
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    
    p1 = ax.bar(ind, correct_guess, width)
    p2 = ax.bar(ind + width, incorrect_correct_guess, width)
    
    ax.set_title('Correct VS incorrect prediction: '+name)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    ax.legend((p1[0], p2[0]), ('Correct', 'Incorrect'))
    ax.autoscale_view()
    plt.savefig('fig/'+'Correct VS incorrect_'+name+'.jpg')
    plt.show()   

    percentage_err = [round(j*100 / (i+j), 2) for i, j in zip(correct_guess, incorrect_correct_guess)]    
    plt.xticks( range(len(percentage_err)) ) # location, labels
    plt.bar(np.arange(len(percentage_err)), percentage_err)
    plt.title('Percentage Error of Digits: '+name)
    plt.ylabel('Percentage Error')
    plt.xlabel('digit')
    plt.savefig('fig/'+'Percentage Error of Digits_'+name+'.jpg')
    plt.show()          
    
# implementing a sigmoid activation function
def activation_function(v,kernel_type):
    s = copy.deepcopy(v)
    vv = copy.deepcopy(v)
    if kernel_type == 'sigmoid':
        s = (1/ (1 + np.exp(-vv)))   
        s[s >= 0.5] = 1
        s[s < 0.5] = 0

    if kernel_type == 'threshold':
        s[s >= 0] = 1
        s[s < 0] = 0
    return s

def ann_modeling(X_train, y_train, kernel_type, beta, epsilon, iteration):
    start_time = time.time()

    x = X_train.to_numpy()
    d = one_hot(y_train, 10)
    
    w = np.ones((10,784)) * 0.01 # w initialization
    w0 = 0 #np.zeros((10, 1)) # zero initialization

    mse = 1
    mse_arr = []
    while mse > epsilon:
        v = np.dot(w,np.transpose(x)) + w0
        y = activation_function(v,kernel_type)
        e = np.transpose(d)-y    
        mse = mean_squared_error(np.transpose(d),y)
        dw = np.dot(beta*e,x)
        w = w + dw
        mse_arr.append(mse)
        if len(mse_arr) > iteration:
           break
    
    print('Iteration #:  %s'% len(mse_arr))
    print("*********************************************************************************")
    print("Traning time for ANN %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_train)), (time.time() - start_time)))
    print("*********************************************************************************")
        
    return w, mse_arr

def ann_classifing(X_test,y_test,w,kernel_type):
    start_time = time.time()

    x = X_test.to_numpy()
    d = one_hot(y_test, 10)

    v = np.dot(w,np.transpose(x)) 
    y = activation_function(v,kernel_type)
    e = np.transpose(d)-y    
    mse = mean_squared_error(np.transpose(d),y)
    
    y_pred = [(np.argmax(r) if np.sum(r)==1 else -1) for r in np.transpose(y)]
    print('Mean Square Error:  %s'% mse)
    print('Percentage Error:  %s'% (abs(sum(i for i in y_pred if i == -1))/len(y_pred)))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("*********************************************************************************")
    print("Test time for ANN %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_test)), (time.time() - start_time)))
    print("*********************************************************************************")
    
    return y_pred
   
def one_hot(labels, length):
    one_hot_labels = np.zeros((labels.size,length))
    one_hot_labels[np.arange(labels.size),labels] = 1
    return one_hot_labels
   
def normalize(df):
    return df.div(255)

def data_selection(df, train_size, train_start, test_size, test_start, binary_threshold = 150):
    df1 = copy.deepcopy(df)
    X_train = df1.iloc[train_start:train_start+train_size,1:]
    if binary_threshold >=0:
        X_train[X_train <= binary_threshold] = 0
        X_train[X_train > binary_threshold] = 1
    else:
        X_train = normalize(X_train)
    
    X_test = df1.iloc[test_start:test_start+test_size,1:]
    if binary_threshold >=0:
        X_test[X_test <= binary_threshold] = 0
        X_test[X_test > binary_threshold] = 1
    else:
        X_test = normalize(X_test)

    y_train = df1.iloc[train_start:train_start+train_size, 0]
    y_test = df1.iloc[test_start:test_start+test_size, 0]
    
    return X_train, y_train, X_test, y_test
    
def ann_svm(df, train_size, train_start, test_size, test_start, binary_threshold, training_rate, epsilon, iteration, kernel_type):
    X_train, y_train, X_test,y_test = data_selection(df, train_size, train_start, test_size, test_start, binary_threshold)
    sns.countplot(y_train)
    plt.savefig('fig/'+'digit_bias_'+str(train_size)+'.jpg')
    plt.show()
    y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')
    for beta in training_rate:
        name = str(kernel_type)+' - train size ' +str(train_size)+' - beta '+ str(beta)
        print('*********************************************************************************')
        print(name)
        ann_w, mse = ann_modeling(X_train, y_train, kernel_type, beta, epsilon, iteration)
        y_pred = ann_classifing(X_test,y_test, ann_w, kernel_type)
        show_mse(mse, name)
        show_digits_error(y_test, y_pred, name)
    return X_test, y_pred 

#*******************************main******************************
start_time = time.time()     
# convert_to_csv("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
#         "mnist_handwritten.csv", 22000)

rand_list = (np.random.randint(0,100,6))

df = pd.read_csv('mnist_handwritten.csv', sep=',', header=None)    
print("converting time: --- %s seconds ---" % (time.time() - start_time))
print(df.head(3))
print(df.shape)

sns.countplot(df[0])
plt.show()

training_rate = [0.9, 0.7, 0.5, 0.2, 0.01]
epsilon = 0.001
iteration = 3000
binary_threshold = 150 # -1 for ignoring binary thresholding


X_test, y_pred = ann_svm(df, 500, 0, 100, 20000, 125, training_rate, epsilon, iteration, 'threshold')
show_samples(X_test, y_pred, rand_list, 'threshold - train size 500'+' - beta 0.01')
X_test, y_pred = ann_svm(df, 10000, 0, 1000, 20000, 125, training_rate, epsilon, iteration, 'threshold')
show_samples(X_test, y_pred, rand_list, 'threshold - train size 10000'+' - beta 0.01')
X_test, y_pred = ann_svm(df, 10000, 0, 1000, 20000, -1, training_rate, epsilon, iteration, 'sigmoid')
show_samples(255*X_test, y_pred, rand_list, 'sigmoid - train size 10000'+' - beta 0.01')

    
print("total time: --- %s seconds ---" % (time.time() - start_time))    