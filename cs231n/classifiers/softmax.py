import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, _ = X.shape
  _, num_classes = W.shape
  for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        expsco = np.exp(scores)
        expscosum = np.sum(expsco)
        loss -= np.log(expsco[y[i]] / expscosum)
        dW[:,y[i]] -= X[i]
        for j in xrange(num_classes):
            dW[:,j] += X[i]*expsco[j]/expscosum
                    
  loss /= X.shape[0]
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, _ = X.shape
  _, num_classes = W.shape

  scores = X.dot(W)
  scores = scores - np.max(scores, axis=1).reshape(-1,1)
  expscores = np.exp(scores)
  expscoresum = np.sum(expscores, axis=1).reshape(-1,1)
  prob = expscores/ expscoresum
  loss = -np.sum(np.log(prob[range(num_train),y]))
  loss = loss/num_train + 0.5*reg*np.sum(W * W)

  indices = np.zeros(prob.shape)
  indices[range(num_train), y] = 1
  dW = np.dot(X.T, prob - indices)
  dW = dW/num_train + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

