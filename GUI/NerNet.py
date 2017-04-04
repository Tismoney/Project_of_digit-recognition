#!/usr/bin/python


import lasagne
from theano import tensor as T
from lasagne.nonlinearities import *
import numpy as np
import time

from mnist import load_dataset

from PyQt4.QtCore import QObject, pyqtSignal

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

class NerNet():

	new_epoch = pyqtSignal(int, name='new_epoch')

	def __init__(self, parent = None):
		self.train_X = 0
		self.train_y = 0
		self.test_X = 0
		self.test_y = 0
		self.val_X = 0
		self.val_y = 0
		self.batch_size = 50
		self.num_epochs = 10

	def accuarcy_fun():
		return

	def pred_fun():
		return

	def init_data(self):
		X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
		self.train_X = X_train
		self.train_y = y_train
		self.test_X = X_test
		self.test_y = y_test
		self.val_X = X_val
		self.val_y = y_val

	def make_and_fit(self, timer):
		
		input_X = T.tensor4('Input')
		target_y = T.vector('Target', dtype='int32')

		input_layer  = lasagne.layers.InputLayer(shape=(None,1,28,28), input_var=input_X, name = "Input")
		dense_layer  = lasagne.layers.DenseLayer(input_layer, num_units=100, nonlinearity=sigmoid, name = "Dense")
		output_layer = lasagne.layers.DenseLayer(dense_layer,num_units = 10, nonlinearity=softmax, name = "Output")

		y_predicted = lasagne.layers.get_output(output_layer)
		all_weights = lasagne.layers.get_all_params(output_layer)

		loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
		accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
		updates_sgd = lasagne.updates.rmsprop(loss, all_weights,learning_rate=0.01)

		train_fun = theano.function([input_X,target_y],[loss,accuracy],updates= updates_sgd)
		self.accuracy_fun = theano.function([input_X,target_y],accuracy)
		self.pred_fun = theano.function([input_X], y_predicted)

		#foo = Foo()
		for epoch in range(self.num_epochs):
		    # In each epoch, we do a full pass over the training data:
		    train_err = 0
		    train_acc = 0
		    train_batches = 0
		    start_time = time.time()
		    for batch in iterate_minibatches(self.train_X, self.train_y, self.batch_size):
		        inputs, targets = batch
		        train_err_batch, train_acc_batch= train_fun(inputs, targets)
		        train_err += train_err_batch
		        train_acc += train_acc_batch
		        train_batches += 1

		    # And a full pass over the validation data:
		    val_acc = 0
		    val_batches = 0
		    for batch in iterate_minibatches(self.val_X, self.val_y, self.batch_size):
		        inputs, targets = batch
		        val_acc += self.accuracy_fun(inputs, targets)
		        val_batches += 1

		    
		    # Then we print the results for this epoch:
		    new_epoch.emit(epoch + 1)
		    print("Epoch {} of {} took {:.3f}s".format(
		        epoch + 1, self.num_epochs, time.time() - start_time))

		    print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
		    print("  train accuracy:\t\t{:.2f} %".format(
		        train_acc / train_batches * 100))
		    print("  validation accuracy:\t\t{:.2f} %".format(
		        val_acc / val_batches * 100))

	def get_accuracy(self):
		test_acc = 0
		test_batches = 0
		for batch in iterate_minibatches(self.test_X, self.test_y, 500):
		    inputs, targets = batch
		    acc = self.accuracy_fun(inputs, targets)
		    test_acc += acc
		    test_batches += 1
		print("Final results:")
		print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
		return test_acc / test_batches * 100


	def get_result(self, X):
		y_pred = self.pred_fun(X)
		y_pred = np.array(y_pred)
		pred_num = y_pred[y_pred.argmax()]
		print("Predict is {} with accuracy {}".format(y_pred, ))

