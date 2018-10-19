import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e"""
    return (1./2)*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx.dot(w)
    return calculate_mse(e)

def least_squares(y, tx): 
    """calculate the least squares solution using normal equations"""  
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def compute_gradient(y, tx, w):
    """Compute the gradient"""
    N = int(y.shape[0])
    e = y - tx.dot(w)
    return np.transpose(tx).dot(e)/(-N)

def least_squares_GD(y, tx, initial_w, max_iters, gamma,computeLoss = True):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad
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



def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def compute_gradient_lasso(y, tx, w,lambda_):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad + lambda_ * np.sign(w)

def lasso_GD(y, tx, initial_w, max_iters, gamma,computeLoss = True):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_lasso(y, tx, w)
        w = w - gamma*grad
    if computeLoss:
        loss = compute_loss(y, tx, w)
        return w,loss
    else:
        return w


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def stochastic_gradient_descent_lasso(
        y, tx, initial_w, batch_size, max_iters, gamma, lambda_):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient_lasso(y_batch, tx_batch, w, lambda_)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            # store w and loss
            ws.append(w)

    return ws[-1]

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te