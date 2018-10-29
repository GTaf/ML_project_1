


%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *
%load_ext autoreload
%autoreload 2

# The fuction below contains a function to dislay the histogram of a choosen feature


def histo(index_feature, y, x,bins):
    
    x_array = x[:,index_feature]
    x_mask = np.where(yb>0, True, False) # Transforming in a boolean mask
    
    print np.mean(x_array)
    print np.std(x_array)
    
    # Displaying the histogram for the whole data
    plt.hist(x_array,bins)
    plt.ylabel("Feature "+str(i))
    
    # Displaying the histogram for the data where y = 1
    plt.figure()
    plt.hist(x_array[x_mask],bins)
    plt.ylabel("Feature "+str(i))
    
    # Displaying the histogram for the data where y = -1
    plt.figure()
    plt.hist(x_array[np.invert(x_mask)],bins)
    plt.ylabel("Feature "+str(i))
    plt.show()


# The 3 functions below split the data according to :
#     - The value of PRI_jet_num (0,1,2 or 3)
#     - The value of DER_Mass_MMC (-999 or well-defined value)




def splitting1(y,x,ids):
    for i in range(4):
        mask = (x[:,22] == i)
        yield y[mask], x[mask], ids[mask]
        
def splitting2(y,x,ids):
    feature0 = x[:,0]
    masks = [0,0]
    mask1 = np.where(feature0==-999,False,True)
    masks[0] = mask1
    masks[1] = np.invert(mask1)
    for mask in masks:
        yield y[mask], x[mask], ids[mask]
        
def full_splitting(y,x,ids):
    """
    Will yields in this order :
    split_id = 0) Data with a defined feature0 ans with PRI_jet_num = 0
    split_id = 1) Data with a defined feature0 ans with PRI_jet_num = 1
    split_id = 2) Data with a defined feature0 ans with PRI_jet_num = 2
    split_id = 3) Data with a defined feature0 ans with PRI_jet_num = 3
    split_id = 4) Data with an undefined feature0 ans with PRI_jet_num = 0
    split_id = 5) Data with an undefined feature0 ans with PRI_jet_num = 1
    split_id = 6) Data with an undefined feature0 ans with PRI_jet_num = 2
    split_id = 7) Data with an undefined feature0 ans with PRI_jet_num = 3
    """
    split_id = -1
    for y2,x2,ids2 in splitting2(y,x,ids):
        for y1,x1,ids1 in splitting1(y2,x2,ids2):
            split_id+=1
            yield y1,x1,ids1, split_id           


# The 4 functions below prepare the data :
#     - Remove some unrelevant features if needed
#     - Transform the angle features in two features by applying sin() and cos()
#     - Applying the absolute value to symetric features
#     



def remove_features_jetnum(x,split_id):
    
    """
    Remove the useless features if the jet num feature is equal to 0 (--> split_id equal to 0 or 4)
    Remove the useless features if the jet num feature is equal to 1 (--> split_id equal to 1 or 5)
    """
    useless_features_index = []
    if split_id in [0,4]:
        useless_features_index = [4, 5, 6, 12,22, 23, 24, 25 , 26, 27, 28,29,33,34] 
    elif split_id in [1,5]:
        useless_features_index = [4, 5, 6, 12,22, 26, 27, 28,34]
    else:
        return x
    mask = np.ones(int(x.shape[1]), dtype=bool)
    mask[(useless_features_index)] = False
    return x[:,mask]

def angle_processing(x):
    
    angle_features = [15, 18, 20, 25, 28]
    new_x = np.zeros((x.shape[0], x.shape[1] + len(angle_features) ))
    
    for k in range(x.shape[1]):
        if k not in angle_features:
            new_x[:, k] = x[:, k]
        
    for idx, column in enumerate(angle_features): 
        
        new_x[:, column] = np.cos(x[:, column])
        new_x[:, x.shape[1] + idx] = np.sin(x[:, column])
    
    return new_x

def absolute_value(x,split_id):
    
    list0 = [10,11,13,14,16,18,19,20]
    list1 = [10,11,13,14,16,19,20,22,23,24,25]
    list2 = [14,15,17,18,20,24,25,27,28,30,31,32,33,34]
    list3 = [14,15,17,18,20,24,25,27,28,30,31,32,33,34]

    full_list = [list0, list1, list2, list3]
    
    list_ = full_list[split_id%4]

    for index in np.array(list_):
        x[:,index] = np.absolute(x[:,index])
    return x    

def full_data_processing(y,x,ids):
    data_list = []
    for y0,x0,ids0, split_id in full_splitting(y,x,ids):
        
        x0 = angle_processing(x0)
        x0 = remove_features_jetnum(x0,split_id)
        x0 = standardize(x0)
        x0 = absolute_value(x0,split_id)
    
        data_list.append([y0,x0,ids0])
        
    return data_list  


# The two functions below are implementing the regression


def regression(y_test, x_test, y_train, x_train,lambda_):
    
    # Here we can choose which method to use :
    
    w, train_loss = ridge_regression(y_train, x_train, lambda_)   
    #w, train_loss = least_squares_GD_adapt_step(y_train, x_train, 0*np.random.rand(int(x_train.shape[1])), 100, 0.00001,computeLoss = True);
    #w, train_loss = lasso_GD_adapt_step(y_train, x_train, 0*np.random.rand(int(x_train.shape[1])), 200, 0.000001, lambda_, computeLoss = True)
    #w, train_loss = logistic_regression(y_train, x_train, 0*np.random.rand(int(x_train.shape[1])), 100, 0.00001)
    #w, train_loss = reg_logistic_regression(y, tx, lambda_,initial_w, max_iter, gamma)
    
    test_loss = compute_loss(y_test, x_test, w)
    y_pred = predict_labels(w, x_test)
    
    return w, train_loss, test_loss, y_pred 

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


# The 3 functions below train the model using K-Fold Cross-Validation



def build_k_indices(y, k_fold, seed):
    
    """
    Build k indices for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def K_split(y,x,k_indices):
    
    for indices_list in k_indices:
        mask = np.ones(y.shape[0],dtype=bool)
        mask[indices_list] = False
        
        yield y[mask], standardize(x[mask]), y[np.invert(mask)], standardize(x[np.invert(mask)])


def Kfold_regression(y, x, k,degree,lambda_):
    
    seed = 1
    k_indices = build_k_indices(y, k, seed)
    
    train_loss_list = []
    test_loss_list = []
    score_list = []
    
    for y_train, x_train, y_test, x_test, in K_split(y,x,k_indices):
        x_train_poly = build_poly(x_train,degree)
        x_test_poly = build_poly(x_test,degree)
        w, train_loss, test_loss, y_pred = regression(y_test, x_test_poly, y_train, x_train_poly,lambda_)
        train_loss_list+=[train_loss]
        test_loss_list+=[test_loss]
        score_list+=[np.mean(y_pred == y_test)]
        
    return train_loss_list, test_loss_list,score_list  


# The function below tests the 2 parameters : the degree and lambda_


def param_test_simultane(y, x, k):
    
    lambda_list = np.logspace(-8, -1, 15)
    degree_list = np.arange(3,15)
    
    final_train_loss_list = np.zeros((len(degree_list),len(lambda_list)))
    final_test_loss_list = np.zeros((len(degree_list),len(lambda_list)))
    final_score_list = np.zeros((len(degree_list),len(lambda_list)))

    
    
    for i, lambda_ in enumerate(lambda_list):
        for j,degree in enumerate(degree_list):
            
            train_loss_list, test_loss_list,score_list = Kfold_regression(y, x,k,degree,lambda_)
            train_loss, test_loss, score = np.mean(train_loss_list), np.mean(test_loss_list), np.mean(score_list)
            final_train_loss_list[j,i]=train_loss
            final_test_loss_list[j,i]=test_loss
            final_score_list[j,i]=score
            
            print ("Degree = " + str(degree) + " and Lambda = " + str(lambda_) + ", Score = " + str(score))
    
        """
        plt.figure()
        plt.plot(degree_list,final_train_loss_list[:,i])
        plt.ylabel("Train Loss")
        plt.figure()
        plt.plot(degree_list,final_test_loss_list[:,i])
        plt.ylabel("Test Loss")
        plt.figure()
        plt.plot(degree_list,final_score_list[:,i], label = str(lambda_))
        plt.ylabel("Score for lambda = " + str(lambda_))
        plt.ylim(0.6, 1)
        plt.legend()
        plt.show()
        """

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    for i, lambda_ in enumerate(lambda_list):
        plt.plot(degree_list, final_score_list[:,i], label = str(lambda_))
        plt.ylabel("Score")

    plt.ylim(np.min(final_score_list), np.max(final_score_list))
    plt.legend()
    plt.savefig("test" + str(i) + ".png")
    plt.show()
    
    return final_train_loss_list, final_test_loss_list, final_score_list


# The function below make a csv file to upload it on Kaggle


y, x, ids =  load_csv_data("train.csv")
y_test, x_test, ids_test = load_csv_data("test.csv")
data_list = full_data_processing(y,x,ids)


def submission(deg_list, lambda_list,y, x, ids, y_test, x_test, ids_test):
    
    """
    Produces the prediction csv file
    """
    data_list_train = full_data_processing(y,x,ids)
    data_list_test = full_data_processing(y_test,x_test,ids_test)
    
    ids_final = np.array([])
    y_pred_final = np.array([])

    for i in range(len(data_list_train)):
        degree = deg_list[i]
        lambda_ = lambda_list[i]
        x_poly = build_poly(data_list_train[i][1], degree)
        x_poly_test = build_poly(data_list_test[i][1], degree)
        
        w, train_loss, test_loss, y_pred = regression(data_list_test[i][0], x_poly_test, data_list_train[i][0], x_poly, lambda_)
    
        ids_final = np.append(ids_final, data_list_test[i][2])
        y_pred_final = np.append(y_pred_final, y_pred)
        
    create_csv_submission(ids_final, y_pred_final, "prediction.csv")


# Below is the degrees and lambdas used for our 8 different models


deg_list = [12,12,12,12,8,4,5,4]
lambda_list = [1e-05,1e-05,0.001,0.0001,1e-05,1e-05,0.0001,0.001]
submission(deg_list,lambda_list,y, x, ids, y_test, x_test, ids_test)

