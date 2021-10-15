# This code requires a special version of hls4ml which includes GarNet and a few minor bug fixes not yet on master.
# You can install it via
# pip install git+https://https://github.com/thaarres/hls4ml.git@jet_tag_paper
# or
# git clone -b jet_tag_paper https://github.com/thaarres/hls4ml.git
# cd hls4ml
# pip install . --user
#
# The current projects can be inspected at https://thaarres.web.cern.ch/thaarres/l1_jet_tagging/l1_jet_tagging_hls4ml_dataset/

import sys, os
import hls4ml
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

def print_dict(d, indent=0):
  align=20
  for key, value in d.items():
    print('  ' * indent + str(key), end='')
    if isinstance(value, dict):
      print()
      print_dict(value, indent+1)
    else:
      print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

            
def synthezise(mname,plotpath,ONAME,build=False):
  
  # Make output directories
  PLOTS = '{}/{}/'.format(plotpath,mname)
  if not os.path.isdir(PLOTS):
    os.mkdir(PLOTS)
    # shutil.copyfile('{}/index.php'.format(plotpath), '{}/index.php'.format(PLOTS)) #Private: for rendering web
  if not os.path.isdir(ONAME):
    os.mkdir(ONAME)
  
  # Load model  
  model = tf.keras.models.load_model('JetTagModels/{}.h5'.format(mname),
                                     custom_objects={'QDense': QDense,
                                                     'QActivation': QActivation,
                                                     'QConv1D' : QConv1D,
                                                     'QConv2D' : QConv2D,
                                                     'quantized_bits': quantized_bits,
                                                     'GarNet': GarNet
                                                   })                                       
    
  if DEBUG:
    model.summary()
  
  # Get softmax layer name
  for layer in model.layers:
    if layer.__class__.__name__ in ['Activation']:
      cfg = layer.get_config()
      if cfg['activation'].find('softmax')!=-1:
        softmax_name = layer.name
        print("{}: Tune hls4ml softmax implementation!".format(layer.name))

  
  # Make more QKeras compatible
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
  
  # make hls config
  config = hls4ml.utils.config_from_keras_model(model, granularity='name',default_reuse_factor=1) #, default_precision='ap_fixed<24,12>'
  config['Model']['Strategy'] = 'Latency'
  config['LayerName'][softmax_name]['exp_table_t'] = 'ap_fixed<18,8>'
  config['LayerName'][softmax_name]['inv_table_t'] = 'ap_fixed<18,4>'
  
  # Handle large span of numerical values in input
  inputPrecision = 'ap_fixed<20,10,AP_RND,AP_SAT>'
  for layer in model.layers:
    if layer.__class__.__name__ in ['BatchNormalization']:
      config['LayerName'][layer.name]['Precision']['scale']   = inputPrecision
      config['LayerName'][layer.name]['Precision']['bias']    = inputPrecision
      config['LayerName'][layer.name]['Precision']['result']  = inputPrecision
    if layer.__class__.__name__ in ['InputLayer']:
      config['LayerName'][layer.name]['Precision']['result'] = inputPrecision
    if layer.__class__.__name__ in ['QConv1D']: # For interaction network
       if 'tmul' in layer.name and 'linear' not in layer.name:
         config['LayerName'][layer.name]['Precision']['weight'] = 'ap_uint<1>'
         config['LayerName'][layer.name]['Precision']['bias'] = 'ap_uint<1>'
  
  # Add tracing to all hls model layers before adding non-traceable layers            
  for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True
  
  if 'InteractionNetwork' in mname:
    config['LayerName'][softmax_name]['Strategy'] = 'Stable'
    config['SkipOptimizers'] = ['reshape_stream']
    config['LayerName']['clone_permute_48'] = {}
    config['LayerName']['clone_permute_48']['Precision'] = inputPrecision
    config['LayerName']['concatenate_25'] = {}
    config['LayerName']['concatenate_25']['Precision'] = inputPrecision
     
  # Bug! Cloned layer gets default precision rather than input precision TODO! Remove this when receive new models from Andre
  if 'QGraphConv' in mname:
    from hls4ml.model.optimizer.optimizer import optimizer_map
    optimizer_map.pop('clone_output')     
      
  #Special cases:      
  changeStrategy = False
  if changeStrategy: #Change strategy if layer is > 4,096. Doesn't work to set strategy per layer for io_stream type models
      for layer in model.layers:
        config['LayerName'][layer.name]['Strategy'] = 'Latency'
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
          print("Layer {} is too large ({}), changing strategy Latency --> Resource".format(layer.name,layersize))
          config['LayerName'][layer.name]['Strategy'] = 'Resource'

  print_dict(config)
  
  # Old hls4ml
  # cfg = hls4ml.converters.create_vivado_config() #Deprecated
  # cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  
  cfg = hls4ml.converters.create_config('xcvu9p-flgb2104-2l-e')
  if 'GraphConv' in mname or 'InteractionNetwork' in mname:
    cfg['IOType']     = 'io_stream'
  else:
    cfg['IOType']     = 'io_parallel'
  cfg['HLSConfig']  = config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '{}/{}'.format(ONAME,mname)
  
  
  print("Convert to hls")
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  print("Compile")
  hls_model.compile()
  
  # Do plots
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/hls4ml_in_plot_{}.png'.format(PLOTS,mname))
  tf.keras.utils.plot_model(model,to_file='{}/keras_in_plot_{}.png'.format(PLOTS,mname))
  
  
  # Has shape (-1,8,3)
  X_test = np.ascontiguousarray(np.load('/eos/home-t/thaarres/level1_jet_validation_samples/x_test_8const_full.npy'))
  Y_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/y_test_8const_full.npy', allow_pickle=True)
  X_test = X_test[:3000]
  Y_test = Y_test[:3000]
    
  if 'GNN_model_' in mname:
    X_test = np.ascontiguousarray(np.load('/eos/home-t/thaarres/level1_jet_validation_samples/x_test_8const_garnet.npy'))
    Y_test = np.load('/eos/home-t/thaarres/level1_jet_validation_samples/y_test_8const_garnet.npy', allow_pickle=True)
    X_test = X_test[:3000]
    Y_test = Y_test[:3000]
    
    vmax = 8
    feat = 3
    V_test = np.ones((X_test.shape[0],1))*vmax
    
    y_hls  = np.array([])
    y_keras= np.array([])
    for i,j in zip(X_test,V_test.astype(np.float64)):
      i = np.expand_dims(i, axis=0)
      j = np.expand_dims(j, axis=0)
      x_hls = [i, j.astype(np.float64)]
      y_keras_ = model.predict(x_hls)
      y_hls_ = hls_model.predict(x_hls).reshape(y_keras_.shape)
      y_hls = np.concatenate([y_hls, y_hls_], axis=0) if y_hls.size else y_hls_
      y_keras = np.concatenate([y_keras, y_keras_], axis=0) if y_keras.size else y_keras_
  
    X_test = [X_test,V_test] 
  elif 'GraphConv' in mname or mname.find('InteractionNetwork')!=-1:
    y_keras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
  else:
    X_test = np.reshape(X_test, (-1,24))
    y_keras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
  accuracy_keras  = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_keras,axis=1)))
  accuracy_hls4ml = float(accuracy_score (np.argmax(Y_test,axis=1), np.argmax(y_hls,axis=1)))

  print('Keras:\n', accuracy_keras)
  print('hls4ml:\n', accuracy_hls4ml)
  print('X_test[0]:\n', X_test[0])
  print('Y_test[0]:\n', Y_test[0])
  print('y_keras[0]:\n', y_keras[0])
  print('y_hls[0]:\n', y_hls[0])

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
  plt.savefig('{}/ROC_keras_{}.png'.format(PLOTS,mname))
  
  if not 'InteractionNetwork' in mname: #TODO! Add profiling for multiple inputs
    wp, wph, ap, aph = hls4ml.model.profiling.numerical(model,hls_model,X_test)
    wp.savefig("{}/wp_{}.png".format(PLOTS,mname))
    wph.savefig("{}/wph_{}.png".format(PLOTS,mname))
    ap.savefig("{}/ap_{}.png".format(PLOTS,mname))
    aph.savefig("{}/aph_{}.png".format(PLOTS,mname))
    fig = hls4ml.model.profiling.compare(model,hls_model,X_test)
    fig.savefig("{}/compare_{}.png".format(PLOTS,mname))

  
  print("Running synthesis!")
  if build:
    hls_model.build(csim=False, synth=True, vsynth=True)

def getReports(indir):
    data_ = {}
    data_['architecture']   = str(indir.split('/')[-2].replace('model_',''))
    data_['precision']   = str(indir.split('_')[-1].replace('bit',''))
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
          lines = np.array(report.readlines())
          data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
          data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
          data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
          data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
          data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
          data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
          data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
          data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
        
        with report_csynth.open() as report:
          lines = np.array(report.readlines())
          lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
          data_['latency_clks'] = int(lat_line.split('|')[2])
          data_['latency_ns']   = int(lat_line.split('|')[2])*5.0
          data_['latency_ii']   = int(lat_line.split('|')[6])
    
    return data_


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--create", help="Create projects", action="store_true")
parser.add_argument("-B", "--build", help="Build projects", action="store_true")
parser.add_argument("--plotdir", help="Output path for plots", default="/eos/home-t/thaarres/www/l1_jet_tagging/l1_jet_tagging_hls4ml_dataset/")
parser.add_argument("-o", "--outdir", help="Output path for projects", default="/mnt/data/thaarres/HLS_PROJECTS")
parser.add_argument("-D", "--debug", help="High verbose", action="store_true")
args = parser.parse_args()
  
   
if __name__ == "__main__":
  

  # List of models to synthesize
  models = [
            "GNN_model_4bit", #TODO! Replace by new models from Arpita, these cannot be loaded from .h5. Tune precision!
            "GNN_model_6bit",
            "GNN_model_8bit",
            "QGraphConv_nconst_8_nbits_4", #TODO! Replace by new models from Andre, changing input layer name
            "QGraphConv_nconst_8_nbits_6",
            "QGraphConv_nconst_8_nbits_8",
            "model_QInteractionNetwork_nconst_8_nbits_4",
            "model_QInteractionNetwork_nconst_8_nbits_6",
            "model_QInteractionNetwork_nconst_8_nbits_8",
            "model_QMLP_nconst_8_nbits_4",
            "model_QMLP_nconst_8_nbits_6",
            "model_QMLP_nconst_8_nbits_8",
          ]
  
  PLOTS = args.plotdir
  ONAME = args.outdir
  DEBUG = args.debug
  
  # Generate projects and produce firmware  
  if args.create:  
    start = time.time()
    Parallel(n_jobs=4, backend='multiprocessing')(delayed(synthezise)(modelname,PLOTS,ONAME,build=args.build) for modelname in models)
    end = time.time()
    print('Ended after {:.4f} s'.format(end-start))
      
    
  # Only read projects
  else:
    
    import pandas
    data = {'architecture':[],'precision':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[],'dsp_rel':[], 'lut_rel':[], 'ff_rel':[],'bram_rel':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    
    for mname in models:
      print("Reading hls project {}/{}/".format(ONAME,mname))
      
      datai = getReports('{}/{}/'.format(ONAME,mname))
      for key in datai.keys():
         data[key].append(datai[key])
    
    dataP = pandas.DataFrame(data)
    print(dataP)
    print(dataP.to_latex(columns=['architecture','precision','latency_clks','latency_ns','latency_ii','dsp','dsp_rel','lut','lut_rel','ff','ff_rel','bram','bram_rel'],header=['Architecture','Precision','Latency [cc]','Latency [ns]','II[cc]','DSP','DSP [%]','LUT','LUT [%]','FF','FF [%]','BRAM','BRAM [%]'],index=False,escape=False))
