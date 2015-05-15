# Solves Problem 4(a), 4(b)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
import numpy as np
from PIL import Image
from sklearn.metrics import adjusted_rand_score
import random
import load

NUM_EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 0.01
MOMENTUM = 0.9

def load_data():
    X_train_raw = []
    y_train_raw = []
    X_test_raw = []
    # load data from the .tiff files
    for i in range(1,31):
        im_train = Image.open('data/segmentation_data/trainvolume'+str(i)+'.tiff')
        X_train_raw.append(np.array(im_train, dtype = theano.config.floatX))
        im_label = Image.open('data/segmentation_data/trainlabels'+str(i)+'.tiff')
        y_train_raw.append(np.array(im_label, dtype = theano.config.floatX))
        im_test = Image.open('data/segmentation_data/testvolume'+str(i)+'.tiff')
        X_test_raw.append(np.array(im_test, dtype = theano.config.floatX))
    X_train_raw = X_train_raw.reshape((X_train_raw.shape[0],1,512,512)) # (30,1,512,512) 30 training images of dimension 512 * 512
    X_test_raw = X_test_raw.reshape((X_test_raw.shape[0],1,512,512)) # (30,1,512,512) 30 test images of dimension 512 * 512
    X_train_windows = np.zeros((7864320,1,95,95)) # 30 * 512 * 512 = 7864320 windows of size 95 * 95 as required by network 4
    X_test_windows = np.zeros((7864320,1,95,95)) # 30 * 512 * 512 = 7864320 windows of size 95 * 95 as required by network 4
    for k in range(30):
        X_train_raw[k,1,:,:] = np.pad(X_train_raw,(47,47),mode = 'reflect') # mirror image at the boundaries
        X_test_raw[k,1,:,:] = np.pad(X_test_raw,(47,47),mode = 'reflect') # mirror image at the boundaries
        for i in range(48,560):
            for j in range(48,560):
                X_train_windows[i*j] = X_train_raw[k,1,i-47:i+47,j-47:j+47]
                X_test_windows[i*j] = X_test_raw[k,1,i-47:i+47,j-47:j+47]
    X_valid = X_train_windows[0:2621440] # 10 * 512 * 512 = 2621440 10 out of 30 training images for validation
    X_train_windows = X_train_windows[2621440:] # 5242880 20 out of 30 training images for training
    y_train_raw = y_train_raw.reshape((y_train_raw.shape[0],1,512,512)) # (30,1,512,512) 30 training label images of dimension 512 * 512
    y_train_all = np.zeros(7864320) # each pixel/window has a label, flatten label images into an array
    for k in range(30):
        for i in range(512):
            for j in range(512):
                y_train_all[i*j*k] = y_train_raw[i,1,j,k]
    y_valid = y_train_all[0:2621440] # split into validation and training set as above
    y_train_all = y_train_all[2621440:]
    X_train = X_train_windows[np.where(y_train_all == 1)] # take all membrane pixels for training, around 50000 pixels per image
    np.append(X_train,X_train_windows[random.sample(np.where(y_train_all == 0),X_train.shape[0])]) # take equal number of nonmembrane pixels
    y_train = y_train_all[np.where(y_train_all == 1)]
    np.append(y_train,np.zeros(X_train.shape[0]-y_train.shape[0]))

    return dict(
        # we need to create a data set that is efficient for GPU computing. 
        # shared variables are stored in the GPU. Always use floatX for GPU stored 
        # variables
        X_train = theano.shared(lasagne.utils.floatX(X_train)),
        # data is always stored as float onthe GPU, since we will need to use y a label
        # we cast it before using it
        y_train = T.cast(theano.shared(y_train),'int32'), 
        X_valid = theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid = T.cast(theano.shared(y_valid),'int32'),
        X_test = theano.shared(lasagne.utils.floatX(X_test)),
        y_test = T.cast(theano.shared(y_test), 'int32'),
        num_examples_train = X_train.shape[0], # number of samples training
        num_examples_valid = X_valid.shape[0], # number of samples validation
        num_examples_test = X_test.shape[0], # number of samples testing
        # image size
        input_height = X_train.shape[2], 
        input_width = X_train.shape[3],
        # number of labels        
        output_dim = 2, 
        )

def build_model(input_width, input_height, output_dim,
                batch_size = BATCH_SIZE):
    
    # INPUT LAYER
    # for info type in ipython lasagne.layers.InputLayer?                    
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_width, input_height),
        )

    # FIRST CONVOLUTIONAL LAYER
    # or info type in ipython lasagne.layers.dnn.Conv2DDNNLayer?     
    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in, # incoming layer
        num_filters = 48, # i.e. depth
        filter_size = (4,4),  # size of the filter
        strides = (1,1), # overlap among filters
        nonlinearity = lasagne.nonlinearities.rectify, # ReLu        
        W = lasagne.init.Uniform(), # how to initialize weight. Uniformely distributed and normalized to fan in fan out
        b = lasagne.init.Constant(0.0),
        # border_mode = None, # can be 'valid','full' or 'same' as in Matlab 
        # pad = 2 # addign zeros on the sides,
        )
    
    # FIRST POOLING LAYER, using MAX    
    l_pool1 = lasagne.layers.MaxPool2DLayer(
        l_conv1, # input layer
        ds = (2,2) # downsampling across the two image dimensions
        )
    
    # SECOND CONVOLUTIONAL LAYER
    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1, # input layer
        num_filters = 48, # i.e. depth
        filter_size = (5,5), # filter size
        nonlinearity = lasagne.nonlinearities.rectify,
        strides = (1,1), # overlap among filters
        W = lasagne.init.Uniform(), # initialization of weights
        b = lasagne.init.Constant(0.0),
        # pad=2 # addign zeros on the sides,
        )

    # SECOND POOLING LAYER, using MAX            
    l_pool2 = lasagne.layers.MaxPool2DLayer(
        l_conv2, # input layer
        ds = (2,2)
        )
   
    # THIRD CONVOLUTIONAL LAYER
    l_conv3 = lasagne.layers.Conv2DLayer(
        l_pool2,# input layer
        num_filters = 48, # i.e. depth
        filter_size = (4,4), # filter size
        nonlinearity = lasagne.nonlinearities.rectify,
        strides = (1,1), # overlap among filters
        W = lasagne.init.Uniform(), # initialization of weights
        b = lasagne.init.Constant(0.0),
        # pad=2 # addign zeros on the sides,
        )

    # THIRD POOLING LAYER, using MAX            
    l_pool3 = lasagne.layers.MaxPool2DLayer(
        l_conv3, # input layer
        ds = (2,2)
        )

    # FOURTH CONVOLUTIONAL LAYER
    l_conv4 = lasagne.layers.Conv2DLayer(
        l_pool3, # input layer
        num_filters = 48, # i.e. depth
        filter_size = (4,4), # filter size
        nonlinearity = lasagne.nonlinearities.rectify,
        strides = (1,1), # overlap among filters
        W = lasagne.init.Uniform(), # initialization of weights
        b = lasagne.init.Constant(0.0),
        # pad=2 # addign zeros on the sides,
        )

    # FOURTH POOLING LAYER, using MAX                       
    l_pool4 = lasagne.layers.MaxPool2DLayer(
        l_conv4, # input layer
        ds = (2,2)
        )

    # DENSE (i.e. fully connected) hidden layer
    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool4, # input later
        num_units = 200, # number of units
        nonlinearity = lasagne.nonlinearities.rectify, # nonlinearity
        W = lasagne.init.Uniform(), # initialization of weights
        )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units = output_dim, # number of labels
        nonlinearity = lasagne.nonlinearities.softmax,
        W = lasagne.init.Uniform(),
        )

    return l_out
    
# CREATE FUNCTIONS CALLED AT EACH ITERATION OF THE ALGORITHM
def create_iter_functions(dataset, 
                          output_layer, 
                          X_tensor_type = T.matrix,
                          batch_size = BATCH_SIZE,
                          learning_rate = LEARNING_RATE, 
                          momentum = MOMENTUM
                          ):            
    
    """
    Create symbolic functions that are used for iterating in different epochs

    :type dataset: dict
    :param dataset: must contain fields for train, validation and test
                    as created by the output of load_data() function

    :type output_layer: theano.tensor.dtensor4
    :param lasagne.layers: symbolic tensor. recursively contain the function representing the holw network

    :type X_tensor_type: function
    :param X_tensor_type: specify the type of data structure used

    :type batch_size: int
    :param batch_size: number of training example to beconsider in one iteration

    :type momentum: float
    :param momentum:parameter used for improving gradient descent, see 
                    http://cs231n.github.io/neural-networks-3/ Parameters Update section
    """
    
    batch_index = T.iscalar('batch_index') # symblic variable used to slice the input
    X_batch = X_tensor_type('x') # symbolic variable used for the feature data
    y_batch = T.ivector('y') # symbolic integer variable used for the label data
    
    # symbolic function that create a sliceing vector 
    batch_slice = slice(
        batch_index*batch_size,(batch_index+1)*batch_size)

    # we define the objective functions
    objective = lasagne.objectives.Objective(output_layer, 
                                            # symbolic function representing the neural network
                                            # symbolic representation cost function see T.nnet.categorical_crossentropy?                      
                                            # and defintion of lasagne.objectives.multinomial_nll                        
                                             loss_function = lasagne.objectives.multinomial_nll 
                                             )
    # symbolic expression for the loss function for training                                             
    loss_train = objective.get_loss(X_batch, target = y_batch)
    # symbolic expression for the loss function for evaluation and test
    # deterministic=True if a dropout layer is present it will behave deterministically
    loss_eval = objective.get_loss(X_batch, target = y_batch, deterministic = True)

    # symbolic expression for the prediction function
    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic = True), axis = 1)
    
    # symbolic expression for the accuracy (average of correctly classified samples)    
    accuracy = T.mean(T.eq(pred, y_batch))

    # symmbolic expression for the network parameters (all weights and biases) 
    # try typing all_params + enter in ipython 
    all_params = lasagne.layers.get_all_params(output_layer)
    
    # expression determining how the all_params are changed at each batch iteration. In this case
    # the function will express symbolically the derivatives of loss_train wrt to each of the 
    # parameters in all_params and apply Stochastic Gradient Descent with learning_rate
    # big advantage here because of vactorization on GPU!
    updates = lasagne.updates.sgd(loss_train,all_params, learning_rate)
    #updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)
    
    # we normally use SGD. Here we pass the loss function and the params, along the learning rate
    # see function definition and explain example! 
    #updates = lasagne.updates.nesterov_momentum(loss_train,all_params, learning_rate, momentum)
    
    # symbolic function that is called every time we train a batch
    iter_train = theano.function(
        [batch_index], # this tells us which batch to use for training, input to the syboilc function
        outputs=loss_train, # symbolic function defined above
        updates=updates, 
        # we specified how to update the parameters at each iteration
        # this part fundamentally tells the function loss_train to use the data in dataset['X_train'][batch_slice]
        # and in dataset['y_train'][batch_slice] instead of the generic variables X_batch and y_batch dfined above        
        givens={ 
            X_batch: dataset['X_train'][batch_slice], 
            y_batch: dataset['y_train'][batch_slice],
            }
        )
        
    # SIMILARLY FOR VALIDATION AND TEST   
    iter_valid = theano.function(
        [batch_index], 
        outputs=[loss_eval, accuracy], # in this case two outputs
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            },
        )

    iter_test = theano.function(
        [batch_index], outputs=pred,
        givens={
            X_batch: dataset['X_test'][batch_slice],
            #y_batch: dataset['y_test'][batch_slice],
            },
        )

    return dict(
        train=iter_train, # these are symbolic functions that receive as input the batch identifier and 
        valid=iter_valid, # output the loss function, while simultaneously updating all_params,
        test=iter_test,    # notice that all_params are composed of shared variables and so accessible
        )                   # between calls
  
def main(num_epochs=NUM_EPOCHS,batch_size=BATCH_SIZE):
    # LOAD DATASET
    dataset = load_data()
    # number of batches for training
    num_batches_train = dataset['num_examples_train'] // batch_size
    # number of batches for evaluation
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    # number of batches for testing
    num_batches_test = dataset['num_examples_test'] // batch_size

    # symbolic expression for the neural network
    output_layer = build_model(
        input_height=dataset['input_height'],
        input_width=dataset['input_width'],
        output_dim=dataset['output_dim'],
        )
    # create symbolic expressions for training a batch
    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor4,
        )
    
    # run the set of all batches every epoch
    total_train_loss=[];
    total_valid_loss=[];
    total_valid_accuracy=[];
    for epoch in range(num_epochs):
        batch_train_losses = []
        for b in range(num_batches_train):
            """ recall this function takes as input the batch index and 
                returns the loss_function. BUT every time it is called it will also update the 
                weights according to the update rule (in our case SGD)
            """
            batch_train_loss = iter_funcs['train'](b) 
            batch_train_losses.append(batch_train_loss)            
        
        
        batch_valid_losses = []
        batch_valid_accuracies = []

        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)
        avg_train_loss = np.mean(batch_train_losses)
        
        print("Epoch %d of %d" % (epoch + 1,num_epochs) )
        print("  training loss:\t\t%.6f" % avg_train_loss)
        print("  validation loss:\t\t%.6f" % avg_valid_loss)
        print("  validation accuracy:\t\t%.2f %%" % (avg_valid_accuracy * 100))
        
        total_train_loss=np.append(total_train_loss,avg_train_loss);
        total_valid_loss=np.append(total_valid_loss,avg_valid_loss);
        total_valid_accuracy=np.append(total_valid_accuracy,avg_valid_accuracy);
#       drawnow(draw_figure)
    
    total_test_pred = [];
    for b in range(num_batches_test):
        batch_test_pred = iter_funcs['test'](b) 
        total_test_pred = np.append(total_test_pred,batch_test_pred)    
    
    # if y_test not available to students, use validation set instead
    total_valid_pred = [];
    for b in range(num_batches_valid):
        batch_valid_pred = iter_funcs['valid'](b) 
        total_valid_pred = np.append(total_valid_pred,batch_valid_pred) 

        
    return dict(
        output_layer=output_layer, 
        total_train_loss=total_train_loss,
        total_valid_loss=total_valid_loss,
        total_valid_accuracy=total_valid_accuracy,
        total_valid_pred=total_valid_pred,
        total_test_pred=total_test_pred)
        

#%% 
#if __name__ == '__main__':
#    main(num_epochs=1)
#%%
    
result=main(num_epochs=NUM_EPOCHS);
test_pred = result['total_test_pred'] # binary predictions for all training windows

# Assuming test labels are known and stored in the array "y_test", following shows how to calculate pixel error and Rand error
# If test labels are not available to students, then use validation pixels X_valid and labels y_valid instead
# pixel error = total proportion of pixels that are wrongly classified
pixel_err = (np.where((test_pred - y_test) != 0).shape[0]) / (30 * 512 ** 2)
# Rand error = 1 - rand index/score
# rank index = (a + b) / (n choose 2)
# where a = the number of pixels that are in the same segment in test_pred and in the same segment in y_test
# and b = the number of pixels that are in different segments in test_pred and in different segments in y_test
rand_err = 1 - adjusted_rand_score(y_test, test_pred)