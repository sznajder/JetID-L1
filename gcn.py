# Define GraphConv Model 


# For 32 constituents
if (nconstit==32) :
    nchannels=32
    poolsiz=6
#    nhidden1 = 22            
    nhidden1 = 40            
    nhidden2 = 0
    nhidden3 = 0

# For 16 constituents
elif (nconstit==16):
    nchannels=40              
    poolsiz=6
    nhidden1 = 50
    nhidden2 = 0 
    nhidden3 = 0 

# For 8 constituents
elif (nconstit==8):
    nchannels=40                
    poolsiz=3
    nhidden1 = 50
    nhidden2 = 0 
    nhidden3 = 0 

else:
    print("Invalid numver of constituents --->",nconstit)
    stop

#############################################################################


# Print
print("Trainign with max # of contituents = ", nconstit)
print("Number of node features = ", nfeat)
print("Quantization with nbits=",nbits)


#############################################################################


# Number of target classes
ntargets = 5 

# Instantiate Tensorflow input tensors in Batch mode 
inp = Input(shape=(nconstit,nfeat), name="inp")   # Conv1D input format
#inp = Input(shape=(1,nconstit,nfeat), name="input")    # Conv2D input format


# Input point features BatchNormalization 
#h = BatchNormalization(name='BatchNorm')(inp)

############ Here we can use either CONDV or DENSE layers #############

# Conv1D with kernel_size=1 and stride=1 to get neighbouring nodes embedded features ( nchannels )
#hn = QConv1D(nchannels, kernel_size=1, strides=1, name='conv1D_2', activation=conv_qbits, kernel_quantizer=qbits, bias_quantizer=qbits, use_bias="False" )(h)
hn = QConv1D(nchannels, kernel_size=1, strides=1, name='conv1D_1', kernel_quantizer=qbits, bias_quantizer=qbits, use_bias="False" )(inp)
hn = QActivation( activation = conv_qbits, name = 'activ_conv_1')(hn) # linear convolution activation

# Dense layer to get each neighbouring node  embedded features ( nchannels ). 
# Weights acts only on features (tensor last component) and have the same value for all constituents ( shared weights )
#hn = QDense(nchannels,name='dense_1', activation=qbits,kernel_quantizer=qbits,bias_quantizer=qbits,use_bias="False")(h)

# Conv1D with kernel_size=1 and stride=1 to get each node embedded features ( nchannels )
#   input node features ( pt, eta_rel, phi_rel) ---- mapped into ----> nchannels features 
#h = QConv1D(nchannels, kernel_size=1, strides=1, name='conv1D_1', activation=conv_qbits, kernel_quantizer=qbits, bias_quantizer=qbits, use_bias="True" )(h)
h = QConv1D(nchannels, kernel_size=1, strides=1, name='conv1D_2', kernel_quantizer=qbits, bias_quantizer=qbits, use_bias="True" )(inp)
h = QActivation( activation = conv_qbits, name = 'activ_conv_2')(h) # linear convolution activation

# Dense layer to get each node embedded features ( nchannels ). 
# Weights acts only on features which are the last components and have the same value for all constituents 
#h = QDense(nchannels,name='dense_2', activation=qbits,kernel_quantizer=qbits,bias_quantizer=qbits,use_bias="True")(h)

###################################################################################

# Agregate neighbouring nodes embedded features ( avg over all nodes for fully connected graph )
hn = GlobalAveragePooling1D(name='avgpool_1')(hn)      # sum features over constituents and normalize by nconstit 
#hn = BatchNormalization(name='BatchNorm_2')(hn)


# Add embedded each node features (h) to the aggregated neighbours features (hn)
hn = Reshape( (1,nchannels),name='reshap')(hn)       # reshape tensor to original 3D format (batch, 1, nfeat)
hn = UpSampling1D(size=nconstit,name='upsampl')(hn)    # make #(nconstit) copies of tensor along axis=1
h = Add(name='add1')([h,hn])       # add neighbours average features to each node feature ( W1.x+W2.x_avg_neighb+B1)

# Activate the nodes agreggation
h = QActivation(activation = qact, name = 'activ_aggregation')(h)

# Linear activation to change HLS bitwidth to fix overflow in AveragePooling
h = QActivation(activation='quantized_bits(15,7)', name = 'linear_activ')(h)


# Reduce number of graph nodes by avg. pooling 
h = AveragePooling1D(pool_size=poolsiz,name='avgpool_2')(h)

# Flatten for MLP input ( now each constituent embedded feature gets an independent weight )
h = Flatten(name='Flatten')(h)

# Dense layers for classification
h = QDense(nhidden1, name='dense_3', kernel_quantizer=qbits, bias_quantizer=qbits )(h)
h = QActivation( activation = qact, name = 'activ_dense_1')(h)


out = QDense(ntargets, name='denseOut', kernel_quantizer=qbits, bias_quantizer=qbits )(h)  # (N, num_classes)
out = Activation("softmax", name="softmx")(out)

