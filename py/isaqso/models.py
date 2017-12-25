from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, Concatenate
from keras.models import Model, load_model, save_model
from keras.preprocessing import image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras import regularizers

def SimpleNet(input_shape =  None, classes = 6):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=5
    for stage in range(nlayers):
        X = Conv1D(2*stage+8, 2*(nlayers-stage), strides = 1,name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=10, strides = 2)(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='SimpleNet')

    return model

def SimpleNetZ(input_shape =  None, classes = 6, reg = 0.):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=5
    nfilters_max = 256
    for stage in range(nlayers):
        nfilters = min(2**(stage+6), nfilters_max)
        filter_size = int(X.shape[1]//10)
        strides = 2
        print X.shape
        X = Conv1D(nfilters, filter_size, strides = strides, name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(reg))(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)

    # output layer
    X = Flatten()(X)
    X_softmax = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    X_reg = Dense(1, activation='relu', name='fc_reg', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = [X_softmax, X_reg], name='SimpleNet')

    return model


def IsaQSO(input_shape =  None, classes = 6, reg = 0.):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=5
    nfilters_max = 256
    for stage in range(nlayers):
        nfilters = min(2**(stage+6), nfilters_max)
        filter_size = int(X.shape[1]//10)
        strides = 2
        print X.shape
        X = Conv1D(nfilters, filter_size, strides = strides, name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(reg))(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)

    # output layer
    X = Flatten()(X)
    X_softmax = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    X_reg = Dense(1, activation='relu', name='fc_reg', kernel_initializer = glorot_uniform(seed=0))(X)
    X_bal = Dense(1, activation='sigmoid', name='fc_bal', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = [X_softmax, X_reg, X_bal], name='Isabal')

    return model


def newIsaQSO(input_shape =  None, classes = 6, reg = 0.):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=5
    filter_size = 10
    strides = 2
    max_nfilters = 256
    for stage in range(nlayers):
        print X.shape, filter_size, strides
        nfilters = min(2**(6+stage), max_nfilters)
        X = Conv1D(nfilters, filter_size, strides = strides, name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(reg))(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)

    # output layer
    X = Flatten()(X)
    X_softmax = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    X_reg = Dense(1, activation='relu', name='fc_reg', kernel_initializer = glorot_uniform(seed=0))(X)
    X_bal = Dense(1, activation='sigmoid', name='fc_bal', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = [X_softmax, X_reg, X_bal], name='Isabal')

    return model


