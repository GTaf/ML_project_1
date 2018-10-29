import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    std_x = np.where(std_x==0,1,std_x)
    x = x / std_x
    return x

def calculate_mse(e):
    """Calculate the mse for vector e"""
    return (1./2)*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = (y - tx.dot(w))
    return calculate_mse(e)

def least_squares(y, tx,computeLoss = True): 
    """calculate the least squares solution using normal equations"""  
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    if computeLoss:
        loss = compute_loss(y, tx, w)
        return w, loss
    else:
        return w

def compute_gradient(y, tx, w):
    """Compute the gradient"""
    N = int(y.shape[0])
    e = y - tx.dot(w)

    return -(np.transpose(tx).dot(e))/(N)

def least_squares_GD(y, tx, initial_w, max_iters, gamma,computeLoss = True):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        print(grad)
        w = w - gamma*grad
        gamma = 0.5*gamma
        print(w)
    if computeLoss:
        loss = compute_loss(y, tx, w)
        return w,loss
    else:
        return w
    
def least_squares_GD_adapt_step(y, tx, initial_w, max_iters, gamma,computeLoss = True):
    """Gradient descent algorithm."""
    w = initial_w
    grad = compute_gradient(y,tx, w)
    for n_iter in range(max_iters):
        
        w1 = w - gamma*grad
        grad1 = compute_gradient(y,tx,w1)
        gamma = ((grad-grad1).T.dot(w-w1))/((grad-grad1).T.dot(grad-grad1))
        w = w1
        grad = grad1
        print (gamma)
        
    if computeLoss:
        loss = compute_loss(y, tx, w)
        return w,loss
    else:
        return w

def stochastic_gradient_descent1(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent algorithm"""
    N = int(y.shape[0])
    w0 = initial_w
    for n_iters in range(max_iters-1):
        i = np.random.randint(0,N)
        w0 = least_squares_GD(y[i:i+1], tx[i:i+1], w0, 1, gamma,computeLoss = False)
    i = np.random.randint(0,N)
    w,loss = least_squares_GD(y[i:i+1], tx[i:i+1], w0, 1, gamma)
       
    return w,loss

def ridge_regression(y, tx, lambda_): 
    """calculate the ridge regression solution using normal equations"""  
    N = int(tx.shape[0])
    a = tx.T.dot(tx) + 2*N*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w,loss


def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = np.sum(np.log(np.ones(len(y)) + np.exp(tx.dot(w))) ) - y.T.dot(tx.dot(w))
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    gradient = tx.T.dot(sigmoid(tx.dot(w) - y))   
    return gradient

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    a = sigmoid(tx.dot(w))
    len_a = len(a)
    S = np.diag(np.diag( (a * (np.ones(len_a) - a))))
    b = tx.T.dot(S)
    hessian = b.dot(tx)
    return hessian

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)

def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w  = w - np.linalg.inv(hessian).dot(gradient)
    return loss, w



def logistic_regression_imp(y, tx, initial_w, max_iter, gamma_):

    # init parameters
    threshold = 1e-8
    losses = []

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses[-1], w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w, 2)**2
    gradient = calculate_gradient(y, tx, w) + 2*lambda_*w
    hessian = calculate_hessian(y, tx, w) + 2*lambda_
    return loss, gradient, hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w  = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w

def logistic_regression_penalized_imp(y, tx, initial_w, max_iter, gamma, lambda_):

    # init parameters
    threshold = 1e-8
    losses = []

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses[-1], w
