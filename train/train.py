
import numpy as np
import h5py
import os
import time
import pickle
import argparse


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Permute, Concatenate, Flatten, Reshape, BatchNormalization, Activation, Dense, Conv1D
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras import utils
from qkeras import *
import itertools


# for pT, eta_rel, phi_rel
#   myJetConstituentList = np.array(f.get("jetConstituentList")[:,:,[5,8,11]])
# for px, py, pz
#   myJetConstituentList = np.array(f.get("jetConstituentList")[:,:,[0,1,2]])
#   myJetConstituentList = np.array(f.get("jetConstituentList"))
#
# Jet Constituents Features =  [0='j1_px', 1='j1_py', 2='j1_pz', 3='j1_e', 4='j1_erel', 5='j1_pt', 6='j1_ptrel',
#                         7='j1_eta', 8='j1_etarel', 9='j1_etarot', 10='j1_phi', 11='j1_phirel', 12='j1_phirot',
#                         13='j1_deltaR', 14='j1_costheta', 15='j1_costhetarel', 16='j1_pdgid']

#Data PATH
#TRAIN_PATH = '/homes/zque/ws/CERN/jedi_lhc_dataset/150p_dataset/train/'

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-nmax", type=int, default=8, help="number of particle")
parser.add_argument("-De", type=int, default=8, help="De")
parser.add_argument("-NL", type=int, default=0, help="number of layer for the MLP_e")
parser.add_argument("-SE", type=int, default=0, help="scale_e")
parser.add_argument("-SN", type=int, default=2, help="scale_n")
parser.add_argument("-batch", type=int, default=512, help="batch")
parser.add_argument("-epochs", type=int, default=200, help="epochs")
parser.add_argument("-patience", type=int, default=20, help="patience")
parser.add_argument("-acc", type=int, default=0, help="accuracy or loss")
parser.add_argument("-seed", type=int, default=7, help="random seed")
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)



jetConstituent = np.load('jetConstituent_150_3f.npy')
target = np.load('jetConstituent_target_150_3f.npy')


# Restric the number of constituents to a maximum of NMAX
nmax = args.nmax
jetConstituent = jetConstituent[:,0:nmax,:]

# The dataset is N_jets x N_constituents x N_features
njet     = jetConstituent.shape[0]
nconstit = jetConstituent.shape[1]
nfeat    = jetConstituent.shape[2]


print('Number of jets =',njet)
print('Number of constituents =',nconstit)
print('Number of features =',nfeat)


# Shuffles jet constituents
print("Before --->> jetConstituent[0,0:4,0] = ",jetConstituent[0,0:4,0])
for i in range(jetConstituent.shape[0]):
  jetConstituent[i] = jetConstituent[i, np.random.permutation(nconstit), :]
print("After --->> jetConstituent[0,0:4,0] = ",jetConstituent[0,0:4,0])


from sklearn.model_selection import train_test_split

X = jetConstituent
Y = target
del jetConstituent , target

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

print(X_train_val.shape, X_test.shape, Y_train_val.shape, Y_test.shape)

print('number of G jets for training/validation: %i'%np.sum( np.argmax(Y_train_val, axis=1)==0 ))
print('number of Q jets for training/validation: %i'%np.sum( np.argmax(Y_train_val, axis=1)==1 ))
print('number of W jets for training/validation: %i'%np.sum( np.argmax(Y_train_val, axis=1)==2 ))
print('number of Z jets for training/validation: %i'%np.sum( np.argmax(Y_train_val, axis=1)==3 ))
print('number of T jets for training/validation: %i'%np.sum( np.argmax(Y_train_val, axis=1)==4 ))


print('number of G jets for testing: %i'%np.sum( np.argmax(Y_test, axis=1)==0 ))
print('number of Q jets for testing: %i'%np.sum( np.argmax(Y_test, axis=1)==1 ))
print('number of W jets for testing: %i'%np.sum( np.argmax(Y_test, axis=1)==2 ))
print('number of Z jets for testing: %i'%np.sum( np.argmax(Y_test, axis=1)==3 ))
print('number of T jets for testing: %i'%np.sum( np.argmax(Y_test, axis=1)==4 ))

# baseline keras model

njet     = X_train_val.shape[0]
nconstit = X_train_val.shape[1]
ntargets =  Y_train_val.shape[1]
nfeat =  X_train_val.shape[2]


print("#jets = ",njet)
print("#constituents = ",nconstit)
print("#targets = ",ntargets)
print("#features = ",nfeat)

###############################################################

'''
# Params for 32 constituents
De=6                  # size of latent edges features representations
Do=6                  # size of latent nodes features representations
scale_e = 3           # multiplicative factor for # hidden neurons in Edges MLP 
scale_n = 3           # multiplicative factor for # hidden neurons in Nodes MLP 
scale_g = 0.12          # multiplicative factor for # hidden neurons in Graph MLP 
'''

'''
# Params for 16 constituents
De=6                  # size of latent edges features representations
Do=6                  # size of latent nodes features representations
scale_e = 4           # multiplicative factor for # hidden neurons in Edges MLP 
scale_n = 4           # multiplicative factor for # hidden neurons in Nodes MLP 
scale_g = .35         # multiplicative factor for # hidden neurons in Graph MLP 
'''

'''
# Params for 8 constituents
De=6                  # size of latent edges features representations
Do=6                  # size of latent nodes features representations
#scale_e = 2           # multiplicative factor for # hidden neurons in Edges MLP 
#scale_n = 2           # multiplicative factor for # hidden neurons in Nodes MLP 
scale_e = 5           # multiplicative factor for # hidden neurons in Edges MLP 
scale_n = 5           # multiplicative factor for # hidden neurons in Nodes MLP 
scale_g = 1           # multiplicative factor for # hidden neurons in Graph MLP 
'''

# Params for 8 constituents - try
De=args.De                  # size of latent edges features representations
Do=6                  # size of latent nodes features representations
#scale_e = 2           # multiplicative factor for # hidden neurons in Edges MLP 
#scale_n = 2           # multiplicative factor for # hidden neurons in Nodes MLP 
scale_e = args.SE           # multiplicative factor for # hidden neurons in Edges MLP 
scale_n = args.SN           # multiplicative factor for # hidden neurons in Nodes MLP 

if(nmax==32):
    scale_g = 0.12
elif(nmax==16):
    scale_g = 0.35        # multiplicative factor for # hidden neurons in Graph MLP 
elif(nmax==8):
    scale_g = 1           # multiplicative factor for # hidden neurons in Graph MLP 
    

NL=args.NL


# Interaction Network model parameters
N = nconstit
P = nfeat
Nr = N*(N-1) # number of relations ( edges )
Dr = 0                     
Dx = 0

#############################################################################

# Quantized bits
nbits=8
integ=0

#qbits = quantized_bits(nbits,integ,alpha=1.0)
#qact = 'quantized_relu('+str(nbits)+',0)'

# Set QKeras quantizer and activation 
if nbits == 1:
    qbits = 'binary(alpha=1)'
elif nbits == 2:
    qbits = 'ternary(alpha=1)'
else:
    qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

qact = 'quantized_relu({},0)'.format(nbits)



# Print
print("Training with max # of contituents = ", nconstit)
print("Number of node features = ", nfeat)
print("Quantization with nbits=",nbits)
print("Quantization of integer part=",integ)


#############################################################################

# Create the sender and receiver relations ( incidence matrices )   
def assign_matrices(N,Nr):
    Rr = np.zeros([N, Nr], dtype=np.float32)
    Rs = np.zeros([N, Nr], dtype=np.float32)
    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0]!=i[1]]
    for i, (r, s) in enumerate(receiver_sender_list):
        Rr[r, i]  = 1
        Rs[s, i] = 1
    return Rs, Rr

#############################################################################



# Matrix of relations features (Dr X Nr) initializes as ones        
#Ra = tf.ones([Dr, Nr])  

### Format input into fully contected graph  ###


# Input shape
inp = Input(shape=(nconstit, nfeat), name="in_layer")
       
# Batch normalize the inputs
x = BatchNormalization(name='batchnorm')(inp)

# Swap axes of input data (batch,nodes,features)->(batch,features,nodes)
#x = tf.transpose(x, perm=[0, 2, 1])  
x = Permute((2, 1), input_shape=x.shape[1:])(x)

# Create a fully conected adjacency matrix ( sender and receiver incidence matrices format )
Rs, Rr = assign_matrices(N,Nr)

# Marshaling function ( use Conv1D to multiply matrices )
ORr = Conv1D(Rr.shape[1], kernel_size=1, use_bias=False, trainable=False, name='tmul_1')(x)
ORs = Conv1D(Rs.shape[1], kernel_size=1, use_bias=False, trainable=False, name='tmul_2')(x)
B = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
del ORr,ORs

###################################################################################
### Edges MLP ( takes as inputs nodes features of a fully conected graph edges ) ###

# Transpose input matrix permutating columns 1&2
inp_e = Permute((2, 1), input_shape=B.shape[1:])(B)

# Define the Edges MLP layers
nhidden_e = int( (2 * P + Dr)*scale_e )
#h = QConv1D(nhidden_e, kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits , name='conv1D_e1')(inp_e)
#h = QActivation(qact, name='qrelu_e1')(h)
#h = QConv1D(int(nhidden_e/2), kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e2' )(h)
#h = QActivation(qact, name='qrelu_e2')(h)
#h = QConv1D(De, kernel_size=1,kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e3' )(h)
if (NL==2): 
    h = QConv1D(nhidden_e, kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits , name='conv1D_e1')(inp_e)
    h = QActivation(qact, name='qrelu_e1')(h)
    h = QConv1D(int(nhidden_e/2), kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e2' )(h)
    h = QActivation(qact, name='qrelu_e2')(h)
    h = QConv1D(De, kernel_size=1,kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e3' )(h)
if (NL==1): 
    h = QConv1D(nhidden_e, kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits , name='conv1D_e1')(inp_e)
    h = QActivation(qact, name='qrelu_e1')(h)
    h = QConv1D(De, kernel_size=1,kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e3' )(h)
elif(NL==0):
    h = QConv1D(De, kernel_size=1,kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_e3' )(inp_e)

out_e = QActivation(qact, name='qrelu_e3')(h)

# Transpose output and permutes columns 1&2  
out_e = Permute((2, 1))(out_e)

# Multiply edges MLP output by receiver nodes matrix Rr ( use Conv1D to multiply matrices )
out_e = Conv1D(np.transpose(Rr).shape[1], kernel_size=1, use_bias=False, trainable=False, name='tmul_3')(out_e)

##############################################################################
### Nodes MLP ( takes as inputs node features and embeding from edges MLP )###

# Concatenate input Node features and Edges MLP output for the Nodes MLP input
inp_n = Concatenate(axis=1)([x, out_e])   #  Original IN was C = tf.concat([N,x,E], axis=1) 

# Transpose input and permutes columns 1&2
inp_n = Permute((2, 1), input_shape=inp_n.shape[1:])(inp_n)

# Define the Nodes MLP layers
nhidden_n = int( (P + Dx + De)*scale_n ) # number of neurons in Nodes MLP hidden layer 
h = QConv1D(nhidden_n, kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_n1')(inp_n)
h = QActivation(qact, name='qrelu_n1')(h)
h = QConv1D(int(nhidden_n/2), kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_n2')(h)
h = QActivation(qact, name='qrelu_n2')(h)
h = QConv1D(Do, kernel_size=1, kernel_quantizer=qbits, bias_quantizer=qbits, name='conv1D_n3' )(h)
out_n = QActivation(qact, name='qrelu_n3')(h)


#################################
###  Graph classification MLP ###

# Flatten input for the Graph classifier MLP
inp_g = Flatten()(out_n)

# Define Graph classifier MLP  layers
nhidden_g = int( (Do * N)*scale_g )  # Number of nodes in graph MLP hidden layer
h = QDense(nhidden_g, kernel_quantizer=qbits, bias_quantizer=qbits,name='dense_g1' )(inp_g)
h = QActivation(qact,name='qrelu_g1')(h)
h = QDense(ntargets,kernel_quantizer=qbits, bias_quantizer=qbits ,name='dense_g2')(h)
out = Activation('softmax',name='softmax_g2')(h)


#hg2 = QDense(n_targets, kernel_quantizer=qbits, bias_quantizer=qbits )(hg1)
#out = tf.nn.softmax(hg2)

  
#######################################################################################


# create the model
model = Model(inputs=inp, outputs=out)


# Define the optimizer ( minimization algorithm )
#optim = SGD(learning_rate=0.0001,decay=1e-6)
#optim = Nadam(learning_rate=0.0005)
#optim = Adam(learning_rate=0.0001)
optim = Adam(learning_rate=0.0005)
#optim = Adam()


# Compile the Model
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Model Summary
model.summary()


model.get_layer('tmul_1').set_weights([np.expand_dims(Rr, axis=0)])
model.get_layer('tmul_2').set_weights([np.expand_dims(Rs, axis=0)])
model.get_layer('tmul_3').set_weights([np.expand_dims(np.transpose(Rr), axis=0)])
del Rr,Rs

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


outputdir = 'nconst{}_nbits{}_De{}_NL{}_SE{}_SN{}_batch{}_acc{}_{}'.format(nmax, nbits, De, NL, scale_e, scale_n, args.batch, args.acc, time.strftime("%Y%m%d-%H%M%S"))

print('output dir: ', outputdir)
os.mkdir(outputdir)

patience = args.patience

if args.acc:
    # early stopping callback
    es = EarlyStopping(monitor='val_categorical_accuracy', patience=patience)
    # Learning rate scheduler 
    ls = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=patience)

    chkp = ModelCheckpoint(outputdir+'/model_QInteractionNetwork_nconst_'+str(nmax)+'_nbits_'+str(nbits)+'.h5', 
        monitor='val_loss', 
        verbose=1, save_best_only=True, 
        save_weights_only=False, mode='auto', 
        save_freq='epoch')
else:
    # early stopping callback
    es = EarlyStopping(monitor='val_loss', patience=patience)
    # Learning rate scheduler 
    ls = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience)
    
    # model checkpoint callback
    # this saves our model architecture + parameters into mlp_model.h5
    chkp = ModelCheckpoint(outputdir+'/model_QInteractionNetwork_nconst_'+str(nmax)+'_nbits_'+str(nbits)+'.h5', 
        monitor='val_loss', 
        verbose=1, save_best_only=True, 
        save_weights_only=False, mode='auto', 
        save_freq='epoch')


# Train classifier
history = model.fit( X_train_val , Y_train_val, 
                    epochs=args.epochs, 
                    #batch_size=512, 
                    batch_size=args.batch, #small batch
                    verbose=1,
                    callbacks=[es,ls,chkp], 
                    validation_split=0.3 )   




# Has shape (-1,8,3)
from sklearn.metrics import accuracy_score

y_keras = model.predict(X_test)
accuracy_keras  = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_keras,axis=1)))

accs = np.zeros(2)
accs[0] = accuracy_keras
np.savetxt('{}/acc.txt'.format(outputdir), accs, fmt='%.6f')
print('Keras:\n', accuracy_keras)


# Set model and output name
#arch = 'QInteractionNetwork_Conv1D'
#fname = arch+'_nconst_'+str(nmax)+'_nbits_'+str(nbits)
#print('Saving Model : ',fname)

## Save the model+ weights
#model.save(outputdir+'/model_'+fname+'.h5')

## Save the model weights in a separate file
#model.save_weights(outputdir+'/weights_'+fname+'.h5')

print('output dir: ', outputdir)



