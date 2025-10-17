## Architecture independent absolute solvation free energy calculations with neural network potentials

This repository contains information and instructions on how to install and run our alchemical transofrmations for NNPs code with the MACE-OFF23 potential.
We need a combination of several packages. 
This repository only contains required information, input data, sample and analysis scripts. The actual code and used packages are stored in two other (forked) repositories:

We forked the openmm-ml and the mace github repositories and made some adjustments. You need both packages to get the entire set-up running.
You will need:

https://github.com/cbc-univie/openmm-ml and 
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

## Sampling

The data folder contains pdb files for all small solute boxes we used for the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/6812638f50018ac7c5da3dd1).
The script folder contains both the sampling scripts (slurm submit and python script) and the analysis scripts (slurm submit and python scripts.)
The submit script writes and runs submit scripts for all lambda states. To run this, go to the asfe-4D/scripts/ folder and execute:

```
sbatch generate_samples.sh
```

If you want to run a single job for one lambda state, you can run (e.g. for ethane):

```
python sample_states.py --lamb 0.0 --pdb ../data/ethane/input/ethane_waterbox_equil.pdb --shifting linear_to_cutoff --model_size small --default_dtype float32 --timestep 0.001 --tag v1
```

In the `generate_samples.sh` script you can define your lambda-schedule, the shifting style, model size, precision and the time step for the simulation.

## Analysis

Before running the analysis, make sure the required dependencies are installed:
```
mamba install -c conda-forge pymbar mdtraj seaborn tqdm
```

The free energy calculation can be performed using either the pairwise BAR or MBAR method (if MBAR does not converge, you can apply the mbar_filtered method, see pre-print and `calculate_asfe.py` for more details)
Free energy calculations can be performed using the provided `run_analysis.sh` script. 
Make sure to set the correct `base` directory path inside the script.

Example usage:

```
sbatch run_analysis.sh --system_name ethane --version v1 --method bar 
sbatch run_analysis.sh --system_name ethane --version v1 --method mbar 
sbatch run_analysis.sh --system_name ethane --version v1 --method mbar_filtered --start_index 6000 --lambda_range 0 4
```

The MBAR analysis produces an overlap matrix and a weight matrix, as well as a .pkl file containing the u_kn matrix and N_k array, which were used for the free energy estimation. The free energy result is printed to a .txt file.
The BAR analysis saves .pkl files for all lambda pairs containing the u_kn matrix and N_k array. The pairwiese free energy differences are printed to a .txt file and need to be summed up to get the final free energy estimate.

Pre-print:

https://chemrxiv.org/engage/chemrxiv/article-details/6812638f50018ac7c5da3dd1
