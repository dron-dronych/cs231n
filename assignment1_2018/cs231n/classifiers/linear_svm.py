import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W) # D-dimensional vector with class-score for i-th observation
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i, :].T # update j-th class
        dW[:,y[i]] -= X[i, :].T # update j-th class that is correct class


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W 


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################



  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # the scores matrix
  correct_class_scores = scores[np.arange(X.shape[0]),y] 
  correct_class_idx = scores != correct_class_scores[:, np.newaxis]
  # print(correct_class_idx.shape)

  margins = scores[correct_class_idx].reshape((scores.shape[0], -1)) - correct_class_scores[:, np.newaxis] + 1 # should be a one less dim matrix!
  # print(margins.shape) # margins matrix is 500 x 9
  loss = np.sum(margins[margins > 0])

  # X shape is 500 x 3073

  # what to do:
  # 1. find observations (indices) that have margins > 0 per each class (margins matrix has 500 x 9 dimensions) --- where???
  # 2. sum those observations values from X (500 x 3073 dimensions) matrix for each class found in (2)
  # 3. update dW based on (2) - only those from (2), keep 0 where (2) not applicable
  # print(X[np.where(margins > 0)].shape)
  # print((margins > 0).shape)

  correct_class_idx_ = scores == correct_class_scores[:, np.newaxis]
  margins_ = scores - correct_class_scores[:, np.newaxis] + 1
  margins_[correct_class_idx_] = 0
  binary = np.zeros(margins_.shape)
  incorrect_class_idx = scores != correct_class_scores[:, np.newaxis]
  binary[margins_ > 0] = 1
  # binary[incorrect_class_idx] = 1
  # binary[correct_class_idx_] = -1

  row_sum = np.sum(binary, axis=1)
  binary[np.arange(X.shape[0]), y] = -row_sum.T
  dW = X.T.dot(binary)
  # print(binary[:10])


  # TODO these should be the sums of Xi, i.e. need to work with matrices of ones!


  

  # margins_like = np.zeros(margins_.shape) 
  # margins_like[correct_class_idx_] = -1
  # print(margins_like[:5])
  # print('*****************')
  # # print(y[:5])
  # # print(np.max(y))
  # dW_correct = X.T.dot(margins_like)
  # # print(dW_correct[:5])
  # print(X.T[:5])

  # dW += dW_correct


  
  # dW += np.sum(-X, axis=0)[:, np.newaxis]
  # print(np.sum(-X, axis=0).shape)
  loss /= X.shape[0]

  dW /= X.shape[0]
  dW += reg * W

  loss += reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
