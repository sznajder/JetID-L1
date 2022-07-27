# pip install git+https://github.com/com:fastmachinelearning/hls4ml.git@main

import sys, os
import hls4ml
import pickle
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QActivation, QDense, QConv1D, QConv2D, quantized_bits
from qkeras.autoqkeras.utils import print_qmodel_summary

from pathlib import Path
import pprint 

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

from garnet import GarNet

import matplotlib.pyplot as plt 

from joblib import Parallel, delayed
import time
import shutil

import argparse

import os

os.environ['PATH'] = '/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']

def synthezise(mname,plotpath,ONAME,build=False):

  model = tf.keras.models.load_model('models/{}.h5'.format(mname),
                                     custom_objects={'QDense': QDense,
                                                     'QActivation': QActivation,
                                                     'QConv1D' : QConv1D,
                                                     'QConv2D' : QConv2D,
                                                     'quantized_bits': quantized_bits,
                                                     'GarNet': GarNet
                                                   })                                       
  model.summary()

  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'], rounding_mode='AP_RND', saturation_mode='AP_SAT')
  config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<16,6>')
  config['Model']['Strategy'] = 'Latency'
  
  # Handle large span of numerical values in input
  inputPrecision = 'ap_fixed<20,10,AP_RND,AP_SAT>'
  for layer in model.layers:
    if layer.__class__.__name__ in ['BatchNormalization','InputLayer']:
      config['LayerName'][layer.name]['Precision'] = inputPrecision
    elif layer.__class__.__name__ in ['Permute','Concatenate','Flatten','Reshape']:
      print("Skipping trace for:", layer.name)
    else:  
      config['LayerName'][layer.name]['Trace'] = True  
   
  output_dir = f'{ONAME}/{mname}'
      
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type='io_parallel', part='xcvu9p-flgb2104-2l-e')
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
  hls_model.compile()
  
  # Do plots
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=f'{plotpath}/hls4ml_in_plot_{mname}.png')
  tf.keras.utils.plot_model(model,to_file=f'{plotpath}/keras_in_plot_{mname}.png')
  

  # Has shape (-1,8,3)
  nconst = int(mname.split("_")[-3])
  X_test = np.ascontiguousarray(np.load('/eos/home-t/thaarres/level1_jet_validation_samples/x_test_{}const_QGraphConv.npy'.format(nconst)))
  Y_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/y_test_{}const_QGraphConv.npy'.format(nconst), allow_pickle=True)
  X_test = X_test[:3000]
  Y_test = Y_test[:3000]
  
  if mname.find('QMLP') !=-1:
    X_test = np.reshape(X_test, (-1,24))
  
  y_keras = model.predict(X_test)
  y_hls = hls_model.predict(np.ascontiguousarray(X_test))
    
  accuracy_keras  = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_keras,axis=1)))
  accuracy_hls4ml = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_hls,axis=1)))

  accs = {}
  accs['cpu'] = accuracy_keras
  accs['fpga'] = accuracy_hls4ml

  with open('{}/{}/acc.txt'.format(ONAME,mname), "wb") as fp:
    pickle.dump(accs, fp)
  print('Keras:\n', accuracy_keras)
  print('hls4ml:\n', accuracy_hls4ml)

  # Plot the ROC curves
  colors  = ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']
  labels = ['gluon', 'quark', 'W', 'Z', 'top']
  fpr = {}
  tpr = {}
  auc1 = {}
  fig = plt.figure()
  ax = fig.add_subplot()

  for i, label in enumerate(labels):
          fpr[label], tpr[label], threshold = roc_curve(Y_test[:,i], y_keras[:,i])
          auc1[label] = auc(fpr[label], tpr[label])
          ax.plot(tpr[label],fpr[label],label='%s, auc = %.1f%%'%(label,auc1[label]*100.),c=colors[i])
          fpr[label], tpr[label], threshold = roc_curve(Y_test[:,i], y_hls[:,i])
          auc1[label] = auc(fpr[label], tpr[label])
          ax.plot(tpr[label],fpr[label],label='%s HLS, auc = %.1f%%'%(label,auc1[label]*100.),linestyle='dotted',c=colors[i])
  ax.semilogy()
  ax.set_xlabel("sig. efficiency")
  ax.set_ylabel("bkg. mistag rate")
  ax.set_ylim(0.001,1)
  ax.set_xlim(0.,1.)
  plt.figtext(0.2, 0.83,r'{}'.format(mname))
  #ax.set_grid(True)
  ax.legend(loc='lower right')
  plt.savefig(f'{plotpath}/ROC_keras_{mname}.png')

  # if not 'GarNet' in mname: #TODO! Add profiling for multiple inputs
  profile_plots = hls4ml.model.profiling.numerical(model,hls_model,X_test)
  for i,p in enumerate(profile_plots):
    p.savefig(f"{plotpath}/profile_{mname}_{i}.png")
 
  fig = hls4ml.model.profiling.compare(model,hls_model,X_test)
  fig.savefig(f"{plotpath}/compare_{mname}.png")

  
  print("Running synthesis!")
  if build:
    report = hls_model.build(csim=False, synth=True, vsynth=True)
    print(report['CSynthesisReport'])

def getReports(indir):
  
    with open("{}/acc.txt".format(indir), "rb") as fp:
      acc = pickle.load(fp)
  
    data_ = {}
    if 'Garnet' in indir:
      data_['architecture']   = 'GarNet'
    elif 'GraphConv' in indir:
      data_['architecture']   = 'GCN'  
    elif 'InteractionNetwork' in indir:
      data_['architecture']   = 'IN'
    else:
      data_['architecture']   = 'MLP'

    data_['precision']   = str(indir.split('_')[-1].replace('bit','')).replace('/','')
    data_['acc_ratio']   = round(acc['fpga']/acc['cpu'],2)
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
          lines = np.array(report.readlines())
          lut   = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
          ff    = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
          bram  = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
          dsp   = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
          lut_rel = round(float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5].replace('<','')),1)
          ff_rel  = round(float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5]),1)
          bram_rel= round(float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5]),1)
          dsp_rel = round(float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5]),1)
          
          
          data_['lut']     = "{} ({}\%)".format(lut  ,lut_rel )
          data_['ff']      = "{} ({}\%)".format(ff   ,ff_rel  )
          data_['bram']    = "{} ({}\%)".format(bram ,bram_rel)
          data_['dsp']     = "{} ({}\%)".format(dsp  ,dsp_rel )
        
          
          
        
        with report_csynth.open() as report:
          lines = np.array(report.readlines())
          lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
          data_['latency_clks'] = round(int(lat_line.split('|')[2]))
          data_['latency_ns']   = round(int(lat_line.split('|')[2])*5.0)
          data_['latency_ii']   = round(int(lat_line.split('|')[6]))
    
    return data_


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--create", help="Create projects", action="store_true")
parser.add_argument("-B", "--build", help="Build projects", action="store_true")
parser.add_argument("--plotdir", help="Output path for plots", default="/eos/home-t/thaarres/www/l1_jet_tagging/l1_jet_tagging_hls4ml_dataset/")
parser.add_argument("-o", "--outdir", help="Output path for projects", default="/home/thaarres/HLS_PRJS/")
parser.add_argument("-D", "--debug", help="High verbose", action="store_true")
args = parser.parse_args()
  
   
if __name__ == "__main__":
  

  # List of models to synthesize
  models = [
    "model_QGraphConv_nconst_16_nbits_8",
    "model_QGraphConv_nconst_32_nbits_4",
    "model_QGraphConv_nconst_32_nbits_8",
    "model_QGraphConv_nconst_8_nbits_4",
    "model_QGraphConv_nconst_8_nbits_6",
    "model_QGraphConv_nconst_8_nbits_8",
    "model_QInteractionNetwork_Conv1D_nconst_16_nbits_4",
    "model_QInteractionNetwork_Conv1D_nconst_16_nbits_8",
    "model_QInteractionNetwork_Conv1D_nconst_32_nbits_4",
    "model_QInteractionNetwork_Conv1D_nconst_32_nbits_8",
    "model_QInteractionNetwork_Conv1D_nconst_8_nbits_4",
    "model_QInteractionNetwork_Conv1D_nconst_8_nbits_6",
    "model_QInteractionNetwork_Conv1D_nconst_8_nbits_8",
    "model_QMLP_nconst_16_nbits_8",
    "model_QMLP_nconst_32_nbits_8",
    "model_QMLP_nconst_8_nbits_4",   #TODO! CHANGE INPUT LAYER NAME
    "model_QMLP_nconst_8_nbits_6", #TODO! CHANGE INPUT LAYER NAME
    "model_QMLP_nconst_8_nbits_8"
          ]
  
  PLOTS = args.plotdir
  ONAME = args.outdir
  DEBUG = args.debug
  
  # Generate projects and produce firmware  
  if args.create or args.build:  
    start = time.time()
    Parallel(n_jobs=4, backend='multiprocessing')(delayed(synthezise)(modelname,PLOTS,ONAME,build=args.build) for modelname in models)
    end = time.time()
    print('Ended after {:.4f} s'.format(end-start))
      
    
  # Only read projects
  else:
    
    import pandas
    dataMap = {'architecture':[],'precision':[], 'acc_ratio':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    
    for mname in models:
      print("Reading hls project {}/{}/".format(ONAME,mname))
      
      datai = getReports('{}/{}/'.format(ONAME,mname))
      for key in datai.keys():
         dataMap[key].append(datai[key])
    
    dataPandas = pandas.DataFrame(dataMap)
    print(dataPandas)
    print(dataPandas.to_latex(columns=['architecture','precision','acc_ratio','latency_ns','latency_clks','latency_ii','dsp','lut','ff','bram'],header=['Architecture','Precision ( \# bits )', 'Accuracy Ratio (FPGA/CPU)','Latency [ns]','Latency [clock cycles]','II [clock cycles]','DSP','LUT','FF','BRAM'],index=False,escape=False))
