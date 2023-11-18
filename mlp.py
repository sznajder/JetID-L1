

# Define MLP input layer dimension ( NINPUT = neta*nphi )
NINPUT = nconstit*nfeat
NOUTPUT = 5

print("NINPUT = ",NINPUT)
print("NOUTPUT = ",NOUTPUT)


if (nconstit==32) :
# MLP architechture for 32 contituents
#    scale=0.3
#    scale=0.7
#    nhidden1 = int(NINPUT*scale)
#    nhidden2 = int(NINPUT*scale)
#    nhidden3 = int(NINPUT*scale)
#
# OPTUNA Best trialOPTUNA BEST Trial val_ccuracy : 0.6583895087242126

#    '''
#MAXNL = 3
#MAXNEU = 128   # maximum number of neurons per layer for Optuna search
#PATIEN = 20 # maximum pacience for early stop and checkpoint
#REGL1 = 0.0001
    layers = [ 128, 59, 76, 9 ]
    lr = 0.00043296719759933135
    batch = 64
    REGL1 = 0.0001  
    activation="relu" 
#    '''
#
elif (nconstit==16):
# MLP architechture for 16 contituents
#    scale=0.8
#    nhidden1 = int(NINPUT*scale)
#    nhidden2 = int(NINPUT*scale)
#    nhidden3 = int(NINPUT*scale)
#
# OPTUNA BEST Trial :  Accuracy Value:  0.6501372456550598
#MAXNL = 3
#MAXNEU = 128   # maximum number of neurons per layer for Optuna search
#PATIEN = 20 # maximum pacience for early stop and checkpoint
#REGL1 = 0.0001
    layers = [ 128, 28, 79 ]
    lr =  0.00041284843800937004
    batch = 16
    activation="relu" 
#
#
elif (nconstit==8):
    # MLP architechture for 8 contituents
#    scale=1.75
#    nhidden1 = int(NINPUT*scale) 
#    nhidden2 = int(NINPUT*scale) 
#    nhidden3 = int(NINPUT*scale) 
#
# OPTUNA BEST Trial 47 finished with value: 0.6322103142738342 and parameters: {'bsize': 64, 'nlayers': 3, 'nhidden_l0': 152, 'nhidden_l1': 69, 'nhidden_l2': 63, 'learning_rate': 0.00023812077479493001}
#
#
    layers =  [ 152, 69, 63 ]
    lr = 0.0002381
    batch = 64
    activation="relu" 
#
#
else:
    print("Invalid numver of constituents --->",nconstit)
    stop

# Build the model

model = Sequential()

# Define the MLP.
model = Sequential()
model.add( Input(shape=(NINPUT), name = 'inp') )
for i,nhidden in enumerate(layers):
    model.add( QDense(nhidden, name=f'dense_{i}' , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) ) ) 
    model.add( QActivation( activation = qact, name = f'activ_{i}'))
model.add( QDense(5, name=f'dense_out'  , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) ) )
model.add( QActivation(activation='softmax', name = 'activ_out'))

