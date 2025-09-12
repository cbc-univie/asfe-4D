## Architecture independent absolute solvation free energy calculations with neural network potentials

This repository contains information and instructions on how to install and run our alchemical transofrmations for NNPs code with the MACE-OFF potential.
We need a combination of several packages. 
This repository only contains required information, input data and sample scripts. The actual code and used packages are stored in two other (forked) repositories:

We forked the openmm-ml and the mace github repositories and made some adjustments. You need both packages to get the entire set-up running.
You will need:

https://github.com/cbc-univie/openmm-ml
https://github.com/cbc-univie/mace

## How to install this:

```
git clone git@github.com:cbc-univie/asfe-4D.git
git clone git@github.com:cbc-univie/openmm-ml.git
git clone git@github.com:cbc-univie/mace.git

mamba create -n alchemical_nnp python=3.12
mamba activate alchemical_nnp
mamba install pytorch=2.5.1 pytorch-gpu openmm-torch cudatoolkit nnpops -c conda-forge

cd openmm-ml/
pip install .
cd ../mace/
pip install .
```


old:
```
git clone git@github.com:cbc-univie/asfe-4D.git
git clone git@github.com:cbc-univie/openmm-ml.git
cd openmm-ml/
pip install .
mamba install -c conda-forge pytorch 
mamba install -c conda-forge openmm 
mamba install -c conda-forge openmm-torch
cd ..
git clone git@github.com:cbc-univie/mace.git
cd mace
pip install .
mamba install -c conda-forge nnpops
```

## Sample data

The data folder contains pdb files for all small solute boxes we used for the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/6812638f50018ac7c5da3dd1).
The script folder contains both the sampling scripts (submit and python script) and the analysis scripts (sibmit and python scripts.)
The submit script writes and runs submit scripts for all lambda states. Execute:

```
sbatch generate_samples.sh
```

If you want to run a single job for one lambda state, you can run (e.g. for ethane):

```
python sample_states.py --lamb "0.0" --pdb ../data/ethane_waterbox/input/ethane_waterbox_equil.pdb
```



Pre-print:

https://chemrxiv.org/engage/chemrxiv/article-details/6812638f50018ac7c5da3dd1
