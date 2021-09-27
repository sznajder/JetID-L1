#!/usr/bin/env python
# coding: utf-8
import os
import setGPU
# edit depending on where Vivado is installed:
# os.environ['PATH'] = '/<Xilinx installation directory>/Vivado/<version>/bin:' + os.environ['PATH']
os.environ['PATH'] = '/xilinx/Vivado/2019.1/bin:' + os.environ['PATH']
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model


model_file_path = 'model_QInteractionNetwork_nconst_8_nbits_8.h5'


from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)    

model = load_model(model_file_path, custom_objects=co)
model.summary()


import hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

config['Model']['ReuseFactor'] = 1
config['Model']['Strategy'] = 'Latency'
config['Model']['Precision'] = 'ap_fixed<32,16>'
#config['SkipOptimizers'] = ['optimize_pointwise_conv']
for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = False
    if 'input' in layer:
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<32,16>'
    if 'batchnorm' in layer:
        config['LayerName'][layer]['accum_t'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['scale'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<32,16>'
    if 'linear' in layer:
        config['LayerName'][layer]['Precision'] = {}
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<9,6>'
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<9,6>'
    if 'tmul' in layer and 'linear' not in layer:
        config['LayerName'][layer]['Precision']['weight'] = 'ap_uint<1>'
        config['LayerName'][layer]['Precision']['bias'] = 'ap_uint<1>'
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<9,6>'
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<9,6>'
    if 'q_activation' in layer:
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<9,6>'
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<8,3,AP_RND,AP_SAT>'
    if 'q_conv1d' in layer:
        config['LayerName'][layer]['accum_t'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<9,6>'
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<9,6>'
    if 'q_dense' in layer:
        config['LayerName'][layer]['accum_t'] = 'ap_fixed<32,16>'
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<9,6>'
        config['LayerName'][layer]['Precision']['default'] = 'ap_fixed<9,6>'

cfg = hls4ml.converters.create_config(part='xc7z020clg400-1')
cfg['HLSConfig'] = config
cfg['IOType'] = 'io_stream'
cfg['Backend'] = 'Vivado'
cfg['ClockPeriod'] = 5
cfg['KerasModel'] = model
cfg['OutputDir'] = 'hls_output'

print("-----------------------------------")
print(cfg)
print("-----------------------------------")

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()
                                                                                    
X_test = np.random.rand(1, 8, 3)
y_keras = model.predict(X_test)
y_hls = hls_model.predict(X_test).reshape(y_keras.shape)
print('Keras:\n', y_keras)
print('hls4ml:\n', y_hls)
# Synthesize                                                                                                                           
hls_model.build(csim=False)
# Reports                                                                                                                              
hls4ml.report.read_vivado_report(output_dir)