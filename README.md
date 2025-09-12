To make our alchemical transofrmations for NNPs code run, we need a combination of several packages. 
This repository only contains required information, input data and sample scripts.

We forked the openmm-ml and the mace github repositories and made some adjustments. You need both packages to get the entire set-up running.

You will need:

https://github.com/cbc-univie/openmm-ml

https://github.com/cbc-univie/mace

## how to install this:

```
git clone git@github.com:cbc-univie/asfe-4D.git
git clone git@github.com:cbc-univie/openmm-ml.git
git clone git@github.com:cbc-univie/mace.git

mamba create -n 4D python=3.12
mamba activate 4D
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
und einen sample Job per Hand aufrufen:
```
python sample_states.py --lamb "0.0" --pdb ../data/ethane_waterbox/input/ethane_waterbox.pdb
```



Pre-print:

https://chemrxiv.org/engage/chemrxiv/article-details/6812638f50018ac7c5da3dd1