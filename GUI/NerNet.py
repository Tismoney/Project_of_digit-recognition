#!/usr/bin/python3


import lasagne
from theano import tensor as T
from lasagne.nonlinearities import *
import numpy as np

import time
import matplotlib.pyplot as plt

from mnist import load_dataset

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
	    indices = np.arange(len(inputs))
	    np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
	    if shuffle:
	        excerpt = indices[start_idx:start_idx + batchsize]
	    else:
	        excerpt = slice(start_idx, start_idx + batchsize)
	    yield inputs[excerpt], targets[excerpt]

def NerNet(batchsize = 50, epochs = 10):

	def Ner(X_inputs, y_targets, batch_size = 50, num_epochs = 10):
		input_X = T.tensor4('Input')
		target_y = T.vector('Target', dtype='int32')

		input_layer  = lasagne.layers.InputLayer(shape=(None,1,28,28), input_var=input_X, name = "Input")
		dense_layer  = lasagne.layers.DenseLayer(input_layer,num_units=100, nonlinearity=sigmoid, name = "Dense")
		output_layer = lasagne.layers.DenseLayer(dense_layer,num_units = 10, nonlinearity=softmax, name = "Output")

		y_predicted = lasagne.layers.get_output(output_layer)
		all_weights = lasagne.layers.get_all_params(output_layer)

		loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
		accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
		updates_sgd = lasagne.updates.rmsprop(loss, all_weights,learning_rate=0.01)

		train_fun = theano.function([input_X,target_y],[loss,accuracy],updates= updates_sgd)
		accuracy_fun = theano.function([input_X,target_y],accuracy)
		pred_fun = theano.function([input_X], y_predicted)

		for epoch in range(num_epochs):
		    # In each epoch, we do a full pass over the training data:
		    train_err = 0
		    train_acc = 0
		    train_batches = 0
		    start_time = time.time()
		    for batch in iterate_minibatches(X_inputs, y_targets, batch_size):
		        inputs, targets = batch
		        train_err_batch, train_acc_batch= train_fun(inputs, targets)
		        train_err += train_err_batch
		        train_acc += train_acc_batch
		        train_batches += 1

		    # And a full pass over the validation data:
		    val_acc = 0
		    val_batches = 0
		    for batch in iterate_minibatches(X_val, y_val, batch_size):
		        inputs, targets = batch
		        val_acc += accuracy_fun(inputs, targets)
		        val_batches += 1

		    
		    # Then we print the results for this epoch:
		    print("Epoch {} of {} took {:.3f}s".format(
		        epoch + 1, num_epochs, time.time() - start_time))

		    print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
		    print("  train accuracy:\t\t{:.2f} %".format(
		        train_acc / train_batches * 100))
		    print("  validation accuracy:\t\t{:.2f} %".format(
		        val_acc / val_batches * 100))

		return pred_fun, accuracy_fun



	X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
	return Ner(X_train, y_train, batch_size = batchsize, num_epochs = epochs)




if __name__ == '__main__':

	pred_fun, accuracy_fun = NerNet(50, 11)
	X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()

	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 500):
	    inputs, targets = batch
	    acc = accuracy_fun(inputs, targets)
	    test_acc += acc
	    test_batches += 1
	print("Final results:")
	print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

