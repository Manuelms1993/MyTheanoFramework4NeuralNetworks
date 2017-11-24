from theano import tensor as T

def rectify(X):
    return T.maximum(X, 0.)

def sigmoid(X):
    return T.nnet.sigmoid(X)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def tanh(X):
    return T.tanh(X)

def arcTan(X):
    return T.arctan(X)

def softPlus(X):
    return T.nnet.softplus(X)