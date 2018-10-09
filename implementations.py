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

def ridge_regression(y, tx, lambda_): 
    """calculate the ridge regression solution using normal equations"""  
    N = int(y.shape[0])
    a = tx.T.dot(tx) + 2*N*lambda_*np.eye(N)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w,loss