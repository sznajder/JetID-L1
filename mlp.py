

# Define MLP input layer dimension ( NINPUT = neta*nphi )
NINPUT = nconstit*nfeat
NOUTPUT = 5

print("NINPUT = ",NINPUT)
print("NOUTPUT = ",NOUTPUT)

if (nconstit==32) :
    # MLP architechture for 32 contituents
    nhidden1 = int(NINPUT/3.3)
    nhidden2 = int(NINPUT/3.3)
    nhidden3 = int(NINPUT/3.3)
elif (nconstit==16):
    # MLP architechture for 16 contituents
    nhidden1 = int(NINPUT*0.8)
    nhidden2 = int(NINPUT*0.8)
    nhidden3 = int(NINPUT*0.8)
elif (nconstit==8):
    # MLP architechture for 8 contituents
    nhidden1 = int(NINPUT*1.75) 
    nhidden2 = int(NINPUT*1.75) 
    nhidden3 = int(NINPUT*1.75) 
else:
    print("Invalid numver of constituents --->",nconstit)
    stop



# Define the input tensor shape
inp  = Input(shape=(NINPUT,), name = 'inp') 

# Instantiate the MLP architechture 
h = BatchNormalization(name='batchnorm')(inp)

h = QDense( nhidden1, name = 'hidden1', kernel_quantizer=qbits, bias_quantizer=qbits )(h)
h = QActivation( activation = qact, name = 'activation1')(h)

h = QDense( nhidden2, name = 'hidden2', kernel_quantizer=qbits, bias_quantizer=qbits )(h)
h = QActivation( activation = qact, name = 'activation2')(h)

h = QDense( nhidden3, name='hidden3', kernel_quantizer=qbits, bias_quantizer=qbits )(h)
h = QActivation( activation = qact, name = 'activation_dense3')(h)

out = QDense(NOUTPUT, name = 'denseout', kernel_quantizer=qbits, bias_quantizer=qbits )(h)
out= Activation("softmax",name = 'activationOut')(out)


