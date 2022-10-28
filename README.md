# Jet classification on FPGA

With `synthesize.py`, you can generate FPGA firmware for graph convolutional networks, interaction networks, and dense fully connected networks, for the models in https://github.com/sznajder/JetID-L1
You need a Vivado certificate to run (e.g logging into geonosis from lxplus (ssh -XY geonosis)).

Everything is running Tensorflow 2.5, and a custom hls4ml branch

Set up the Conda environment:
```bash
mamba env create -f environment.yml
mamba activate hls4ml-l1jets
```

To synthesize models:
```bash
python synthesize.py -C    # only create projects
python synthesize.py -C -T # create and trace model
python synthesize.py -C -B # create and synthesize projects
python synthesize.py       # Read reports
```