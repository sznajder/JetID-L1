

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
# OPTUNA Best trial 11 with accuracy:  0.6424840092658997
# Large MLP with MAX neurons=256
#
    '''
    nlayers  = 3
    nhidden1 = 124
    nhidden2 = 32
    nhidden3 = 105
    lr = 0.00021698674235274865
    batch = 32
    REGL1 = 0.0001
    '''
#
# OPTUNA Best trial 47 with accuracy:  0.6329506635665894
# Small MLP with MAX neurons=96
#    nlayers = 3
#    nhidden1 = 84
#    nhidden2 = 45
#    nhidden3 = 38
#    lr = 0.0003322842293803936
#    batch = 16
#
#    '''
    nlayers = 4
    nhidden1 = 128
    nhidden2 = 59
    nhidden3 = 76
    nhidden4 = 9
    lr = 0.00043296719759933135
    batch = 64
    REGL1 = 0.0001
#    '''

elif (nconstit==16):
# MLP architechture for 16 contituents
#    scale=0.8
#    nhidden1 = int(NINPUT*scale)
#    nhidden2 = int(NINPUT*scale)
#    nhidden3 = int(NINPUT*scale)
#
# OPTUNA BEST Trial 48 finished with value:0.655561089515686 and parameters: {'bsize': 64, 'nlayers': 3, 'nhidden_l0': 171, 'nhidden_l1': 30, 'nhidden_l2': 153, 'learning_rate': 0.00025664296269595563}.
#
#
    nhidden1 = 171
    nhidden2 = 30
    nhidden3 = 153
    lr = 0.0002566
    batch = 64


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
    nhidden1 = 152
    nhidden2 = 69
    nhidden3 = 63
    lr = 0.0002381
    batch = 64

else:
    print("Invalid numver of constituents --->",nconstit)
    stop



# Define the input tensor shape
inp  = Input(shape=(NINPUT,), name = 'inp') 

# Instantiate the MLP architechture 
#h = BatchNormalization(name='batchnorm')(inp)

h = QDense( nhidden1, name = 'dense_1', kernel_quantizer=qbits, bias_quantizer=qbits , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) )(inp)
h = QActivation( activation = qact, name = 'activ_1')(h)

h = QDense( nhidden2, name = 'dense_2', kernel_quantizer=qbits, bias_quantizer=qbits , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) )(h)
h = QActivation( activation = qact, name = 'activ_2')(h)

h = QDense( nhidden3, name='dense_3', kernel_quantizer=qbits, bias_quantizer=qbits , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) )(h)
h = QActivation( activation = qact, name = 'activ_3')(h)

out = QDense(NOUTPUT, name = 'dense_out', kernel_quantizer=qbits, bias_quantizer=qbits , kernel_regularizer=regularizers.L1(REGL1), bias_regularizer=regularizers.L1(REGL1) )(h)
out= Activation("softmax",name = 'activ_out')(out)


