#!/home/paul/anaconda2/bin/python2.7


import lasagne
import theano
from theano import tensor as T
from lasagne.nonlinearities import *
import numpy as np
import time
import os
import pandas as pd

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

def architecture_one(input_X, weight = False, weig = None):
	if (weight == False):
		input_layer   = lasagne.layers.InputLayer   (shape=(None,1,28,28), input_var=input_X, name = "Input")
		drop_layer    = lasagne.layers.DropoutLayer (input_layer, p=0.2)
		dense_1_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=200, nonlinearity=rectify, name = "Dense_1")
		drop_layer    = lasagne.layers.DropoutLayer (dense_1_layer, p=0.5)
		dense_2_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=100, nonlinearity=sigmoid, name = "Dense_2")
		drop_layer    = lasagne.layers.DropoutLayer (dense_2_layer, p=0.5)
		dense_3_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=50, nonlinearity=rectify, name = "Dense_3")
		drop_layer    = lasagne.layers.DropoutLayer (dense_3_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer   (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output")
	else:
		input_layer   = lasagne.layers.InputLayer   (shape=(None,1,28,28), input_var=input_X, name = "Input")
		drop_layer    = lasagne.layers.DropoutLayer (input_layer, p=0.2)
		dense_1_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=200, nonlinearity=rectify, name = "Dense_1", 
		                                                W = weig[0], b = weig[1].reshape(weig[1].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer (dense_1_layer, p=0.5)
		dense_2_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=100, nonlinearity=sigmoid, name = "Dense_2",
		                                                W = weig[2], b = weig[3].reshape(weig[3].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer (dense_2_layer, p=0.5)
		dense_3_layer = lasagne.layers.DenseLayer   (drop_layer, num_units=50, nonlinearity=rectify, name = "Dense_3", 
		                                                W = weig[4], b = weig[5].reshape(weig[5].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer (dense_3_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer   (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output", 
		                                                W = weig[6], b = weig[7].reshape(weig[7].shape[0], ))
	return output_layer

def architecture_two(input_X, weight = False, weig = None):
	if (weight == False):
		input_layer   = lasagne.layers.InputLayer   (shape=(None,1,28,28), input_var=input_X, name = "Input")
		drop_layer    = lasagne.layers.DropoutLayer (input_layer, p=0.2)
		dense_layer   = lasagne.layers.DenseLayer   (drop_layer, num_units=800, nonlinearity=rectify, name = "Dense")
		drop_layer    = lasagne.layers.DropoutLayer (dense_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer   (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output")
	else:
		input_layer   = lasagne.layers.InputLayer   (shape=(None,1,28,28), input_var=input_X, name = "Input")
		drop_layer    = lasagne.layers.DropoutLayer (input_layer, p=0.5)
		dense_layer   = lasagne.layers.DenseLayer   (drop_layer, num_units=800, nonlinearity=rectify, name = "Dense",
					                                    W = weig[0], b = weig[1].reshape(weig[1].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer (dense_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer   (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output",
					                                    W = weig[2], b = weig[3].reshape(weig[3].shape[0], ))
	return output_layer

def architecture_three(input_X, weight = False, weig = None):
	if (weight == False):
		input_layer   = lasagne.layers.InputLayer    (shape=(None,1,28,28), input_var=input_X, name = "Input")
		conv_layer    = lasagne.layers.Conv2DLayer   (input_layer, num_filters = 32, filter_size = (5, 5), nonlinearity=rectify, name = "Conv2D")
		maxpool_layer = lasagne.layers.MaxPool2DLayer(conv_layer, pool_size = (2,2), name = "MaxPool")
		drop_layer    = lasagne.layers.DropoutLayer  (maxpool_layer, p=0.5) 
		dense_layer   = lasagne.layers.DenseLayer    (drop_layer, num_units=256, nonlinearity=rectify, name = "Dense")
		drop_layer    = lasagne.layers.DropoutLayer  (dense_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer    (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output")
	else:
		input_layer   = lasagne.layers.InputLayer    (shape=(None,1,28,28), input_var=input_X, name = "Input")
		conv_layer    = lasagne.layers.Conv2DLayer   (input_layer, num_filters = 32, filter_size = (5, 5), nonlinearity=rectify, name = "Conv2D",
					                                    W = weig[0], b = weig[1].reshape(weig[1].shape[0], ))
		maxpool_layer = lasagne.layers.MaxPool2DLayer(conv_layer, pool_size = (2,2), name = "MaxPool",
					                                    W = weig[2], b = weig[3].reshape(weig[3].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer  (maxpool_layer, p=0.5) 
		dense_layer   = lasagne.layers.DenseLayer    (drop_layer, num_units=256, nonlinearity=rectify, name = "Dense",
					                                    W = weig[4], b = weig[5].reshape(weig[5].shape[0], ))
		drop_layer    = lasagne.layers.DropoutLayer  (dense_layer, p=0.5)
		output_layer  = lasagne.layers.DenseLayer    (drop_layer,num_units = 10, nonlinearity=softmax, name = "Output",
					                                    W = weig[6], b = weig[7].reshape(weig[7].shape[0], ))
	return output_layer	


class NerNet(QObject):
	new_epoch = pyqtSignal(int, name='new_epoch')
	def __init__(self, parent = None):
		super(NerNet, self).__init__(parent)
		self.train_X = 0
		self.train_y = 0
		self.test_X = 0
		self.test_y = 0
		self.val_X = 0
		self.val_y = 0
		self.batch_size = 50
		self.num_epochs = 1 # it must be 10
		self.acc = 0

	def signalConnect(self, obj):
		self.new_epoch.connect(obj)

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

	def put_weights(self, weights, path = "Weight", conv = False):
		for i, weight in enumerate(weights):
			p = weight.get_value()
			if ( (conv == True) & (i == 0) ):
				p = p.reshape(32, 25)
			matrix = pd.DataFrame(p)
			matrix.to_csv(path + "/" + str(i) + ".csv")

	def get_weight(self, path = "Weight", conv = False):
		weights = []
		num_files = len(os.listdir(path))
		for i in range(num_files):
		    matrix = pd.read_csv(path + "/" + str(i) + ".csv")
		    weight = np.delete(matrix.as_matrix(), [0], axis=1)
		    if ( (conv == True) & (i == 0) ):
				weights = weights.reshape(32, 1, 5, 5)
		    weights.append(weight)
		return weights




	def make_and_fit(self):
		
		input_X = T.tensor4('Input')
		target_y = T.vector('Target', dtype='int32')

		output_layer = architecture_three(input_X, weight = False)
		y_predicted = lasagne.layers.get_output(output_layer)
		all_weights = lasagne.layers.get_all_params(output_layer)

		loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
		accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
		updates_sgd = lasagne.updates.rmsprop(loss, all_weights,learning_rate=0.01)

		train_fun = theano.function([input_X,target_y],[loss,accuracy],updates= updates_sgd)
		self.accuracy_fun = theano.function([input_X,target_y],accuracy)
		self.pred_fun = theano.function([input_X], y_predicted)

		for epoch in range(self.num_epochs):
		    # In each epoch, we do a full pass over the training data:
		    self.new_epoch.emit(epoch + 1)
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
		    
		    print("Epoch {} of {} took {:.3f}s".format(
		        epoch + 1, self.num_epochs, time.time() - start_time))

		    print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
		    print("  train accuracy:\t\t{:.2f} %".format(
		        train_acc / train_batches * 100))
		    print("  validation accuracy:\t\t{:.2f} %".format(
		        val_acc / val_batches * 100))

            	self.put_weights(all_weights, conv = True)

	def make_and_get(self):
		input_X = T.tensor4('Input')
		target_y = T.vector('Target', dtype='int32')
		
		output_layer = architecture_three(input_X, weight = True, weig = self.get_weight(conv = True))
		y_predicted = lasagne.layers.get_output(output_layer)
		all_weights = lasagne.layers.get_all_params(output_layer)

		loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
		accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
		updates_sgd = lasagne.updates.rmsprop(loss, all_weights,learning_rate=0.01)

		train_fun = theano.function([input_X,target_y],[loss,accuracy],updates= updates_sgd)
		self.accuracy_fun = theano.function([input_X,target_y],accuracy)
		self.pred_fun = theano.function([input_X], y_predicted)
		self.new_epoch.emit(self.num_epochs)


	def make_and_check(self, path = "Weight"):
		if (len(os.listdir(path)) == 0): self.make_and_fit()
		else: self.make_and_get()
	#	self.make_and_fit()

	def get_accuracy(self):
		if (self.acc == 0): 
			test_acc = 0
			test_batches = 0
			for batch in iterate_minibatches(self.test_X, self.test_y, 500):
			    inputs, targets = batch
			    acc = self.accuracy_fun(inputs, targets)
			    test_acc += acc
			    test_batches += 1
			print("Final results:")
			print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
			self.acc = test_acc / test_batches * 100
			return test_acc / test_batches * 100
		else:
			return self.acc


	def get_result(self, X):
		self.get_accuracy()
		y_pred = self.pred_fun(X)
		
		y_pred = np.array(y_pred)
		#print y_pred
		pred_num = y_pred.argmax()
		pred_prob = y_pred[0, pred_num] * self.acc
		print("Predict is {} with probabylity {:.2f} %".format(pred_num, pred_prob))
		return pred_num, pred_prob
