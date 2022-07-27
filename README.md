# Jet classification on FPGA

With synthesize.py, you can generate FPGA firmware for Graph Convolutional Networks, Interaction Networks and dense fully connected networks, for the models in https://github.com/sznajder/JetID-L1
You need a Vivado certificate to run (e.g logging into geonosis from lxplus (ssh -XY geonosis)).


Everything is running Tensorflow 2.5, and the hls4ml master branch

Set up the Conda environment:
```
conda env create -f environment.yml
conda activate hls4ml-l1jets
```
This should give you the latest hls4ml main branch (you can install that also via pip install git+https://github.com/com:fastmachinelearning/hls4ml.git@main)
To synthesize models

```
python synthesize.py -C    # only create projects
python synthesize.py -C -B # creaste and synthesise projects
python synthesize.py       # Read reports
```