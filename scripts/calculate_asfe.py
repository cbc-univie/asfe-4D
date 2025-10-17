import numpy as np
import pymbar
from openmm import unit, VerletIntegrator, Context
from openmm.app import PDBFile, DCDFile
from openmmml import MLPotential
from openmm.vec3 import Vec3
import torch
import argparse
import mdtraj as md
from tqdm import tqdm  
import matplotlib.pyplot as plt
import seaborn as sns  
import warnings
from openmm import CustomBondForce
from typing import Optional, Tuple
import pickle
import os
from openmm import unit
warnings.filterwarnings("ignore")

# for kBT to kcal/mole conversion
temperature = 300 * unit.kelvin
kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
kBT = temperature * kB
kBT_kcal = kBT.value_in_unit(unit.kilocalories_per_mole)
        

def write_results_to_file(results, uncertainties, lambda_values, output_file):
    """
    Write the calculated MBAR free energy results to a file.

    Parameters
    ----------
    results : ndarray
        Free energy differences between states.
    uncertainties : ndarray
        Uncertainties of the free energy differences.
    lambda_values : ndarray
        Array of lambda values.
    output_file : str
        Path to the output file.
    """
    with open(output_file, "a+") as f:
        f.write("Lambda, Free Energy (kT), Uncertainty kT)\n")
        for i, lamb in enumerate(lambda_values):
            f.write(f"{lamb:.2f}, {results[0, i]:.2f}, {uncertainties[0, i]:.2f}\n")
    print(f"Results written to {output_file}")

def write_pairwise_results_to_file(pairwise_dGs, pairwise_lambdas, output_file):

    with open(output_file, "w") as f:
        f.write("Lambda_1, Lambda_2, Free Energy (kT), Uncertainty (kT)\n")
        for i, r in enumerate(pairwise_dGs):
            lambda_pair = pairwise_lambdas[i]
            delta_f = r["Delta_f"][0, 1]  # free energy difference
            ddelta_f = r["dDelta_f"][0, 1]  # uncertainty
            
            f.write(f"{lambda_pair[0]:.2f}, {lambda_pair[1]:.2f}, {delta_f:.2f}, {ddelta_f:.2f}\n")
    print(f"Pairwise free energies written to '{output_file}'")

def plot_overlap(overlap_matrix, lambda_values, output_file):
    """
    Plot the overlap matrix as a heatmap.

    Parameters
    ----------
    overlap_matrix : ndarray
        Overlap matrix from MBAR.
    lambda_values : ndarray
        Array of lambda values.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=[f"{lamb:.2f}" for lamb in lambda_values],
        yticklabels=[f"{lamb:.2f}" for lamb in lambda_values],
        cmap="coolwarm",
    )
    plt.title("Overlap Matrix")
    plt.xlabel("Lambda State")
    plt.ylabel("Lambda State")
    plt.savefig(output_file)
    #plt.show()


def plot_weight_matrix(W_nk, lambda_values, sample_count, output_file):
    """
    Plot the weight matrix as a heatmap.

    Parameters
    ----------
    W_nk : ndarray
        Weight matrix from MBAR.
    lambda_values : ndarray
        Array of lambda values.
    sample_count : int
        Total number of samples across all states.
    """
    plt.figure(figsize=(12, 6))
    #print(f"Shape of weight matrix: {W_nk.shape}")
    sns.heatmap(
        #W_nk[8000:,:5], # plot only part of the weight matrix to see high weights
        W_nk,
        cmap="viridis",
        cbar=True,
        #xticklabels=[f"Lambda {i}" for i in range(sample_count // len(lambda_values))],
    )
    plt.title("Weight Matrix (W_nk)")
    plt.ylabel("Sample Index")
    plt.xlabel("Lambda State")
    plt.savefig(output_file)
    #plt.show()

def find_weights_above_threshold(W_nk, threshold=0.0006, start_index=7000, lambda_range=(0, 4)):
    """
    Identify samples with weights above a specified threshold in the MBAR weight matrix.
    
    This function searches through the weight matrix W_nk to find samples that have
    unusually high weights, which can indicate problematic configurations that may
    negatively impact the quality of free energy result. These high-weight
    samples can be excluded from subsequent MBAR analysis to ensure
    convergence.

    """
    print(f"Looking for samples with high weight (>{threshold}) in the weight matrix...")
    W_nk = np.array(W_nk)
    results = []

    for n in range(start_index, W_nk.shape[0]):
        for k in range(lambda_range[0], lambda_range[1] + 1):
            weight = W_nk[n, k]
            if weight > threshold:
                results.append((n, k, weight))

    exclude_indices = sorted(set([n for n, _, _ in results]))

    print(f"Found {len(exclude_indices)} unique samples to exclude: {exclude_indices}")
    return results, exclude_indices

def read_free_energy_from_file(output_file):

    if not os.path.exists(output_file):
        print(f"No cached results found in {output_file}. Proceeding with calculation...")
        return None

    total_pairwise_dG = 0.0

    with open(output_file, "r") as f:
        lines = f.readlines()[1:]  # skip the header

    for line in lines:
        _, _, free_energy, _ = map(float, line.strip().split(", "))
        total_pairwise_dG += free_energy

    print("\n##################################################")
    print(f"Sum of pairwise dGs: {total_pairwise_dG:.2f} kT")
    total_pairwise_dG_kcal = total_pairwise_dG * kBT_kcal
    print(f"Sum of pairwise dGs: {total_pairwise_dG_kcal:.2f} kcal/mol")
    return total_pairwise_dG


def calculate_free_energies(
    pdb_file: str,
    trajectory_template: str,
    lambda_values: np.ndarray,
    method: str,
    size: str,
    shifting_style:str,
    default_dtype: str,
    start_index: Optional[int],
    lambda_range: Optional[Tuple[int, int]],
    temperature: float = 300,
    every_nth_frame: int = 1,  
):
    """
    Calculate free energies using MBAR, BAR, or filtered MBAR.


    Parameters
    ----------
    pdb_file : str
        Path to the PDB file used to define the system topology.
    trajectory_template : str
        Template for the trajectory file names, e.g., 'trajectory_lambda_{:.4f}.dcd'.
    lambda_values : np.ndarray
        Array of lambda values used for the alchemical transformation.
    method : str
        Method for free energy calculation: 'bar', 'mbar', or 'mbar_filtered'.
        If MBAR does not converge (see pre-print), you can apply the filtered MBAR approach (mbar_filtered method). 
        This method filters out off-diagonal high-weight samples before rerunning MBAR. 
        You can specify the lambda range where high weights should be identified (typically near lambda=0).
        A starting index for the sample can also be set 
        (use an index larger than the number of samples used for lambda states close to 0 to avoid filtering valid samples).
    start_index : int or None
        Starting frame index for filtering in 'mbar_filtered' method.
    lambda_range : tuple(int, int) or None
        Inclusive lambda index range (e.g., (0, 4)) for filtering in 'mbar_filtered' method.
    temperature : float, optional
        Simulation temperature in Kelvin (default is 300).
    every_nth_frame : int, optional
        Frequency of frame sampling from trajectories (default is 1, meaning every frame).
    """
    
    print("***************************************************************")
    print(f"Calculating free energies using {method}...")
    print("***************************************************************")
    
    beta = 1.0 / (
        unit.BOLTZMANN_CONSTANT_kB
        * (temperature * unit.kelvin)
        * unit.AVOGADRO_CONSTANT_NA
    )
    # Load the PDB file
    pdb = PDBFile(pdb_file)
    # Define atom groups based on residue names
    atom_groups = torch.zeros(len(pdb.positions), dtype=torch.long)

    for i, atom in enumerate(pdb.topology.atoms()):
        residue_name = atom.residue.name
        if residue_name in {"UNK"}:
            atom_groups[i] = 1  # Group for lignad
        else:
            atom_groups[i] = 0  # Group for solvent

    # Check if there are any entries with value 1
    if not torch.any(atom_groups == 1):
        print("Error: No atoms were assigned to group 1 (e.g., solvent). Exiting.")
        exit()

    if method == 'mbar' or method == 'mbar_filtered':
        # Number of lambda states
        K = len(lambda_values)
    elif method == 'bar':
        # Number of lambda states
        K = 2
        pairwise_dGs = []  
        pairwise_lambdas = [] 

    if method =='mbar' or method=='mbar_filtered':
        # calculate the free energy difference with all available lambda states (samples)
        repeat_dG_calculation = range(1)
    elif method == 'bar':
        # calculate the free energy difference for each lambda pair
        repeat_dG_calculation = range(len(lambda_values) - 1)
    
    compute_ukn = True

    all_samples = []
    all_box_vectors = []
    N_ks = []
        
    # go once through this loop for mbar; for pairwise bar all pairs will be evaluated separately        
    for i in tqdm(repeat_dG_calculation):
        
        if method == "bar":
            # check if the pairwise results file already exists
            pairwise_results_file = "pairwise_results_BAR.txt"
            cached_total_dG = read_free_energy_from_file(pairwise_results_file)

            if cached_total_dG is not None:
                print("BAR results read from file.")
                return  # exit early if cached results exist

            # empty lists for each lambda pair
            all_samples = []
            all_box_vectors = []
            N_ks = []

            # get the current pair of lambdas
            lambda_pair = lambda_values[i:i+2]
            print(f"processing lambda pair {lambda_pair}")
            pairwise_lambdas.append(lambda_pair)
            lambda_values_processed = lambda_pair

        elif method == "mbar" or method == "mbar_filtered":
            lambda_values_processed = lambda_values
        
            # check if u_kn already exists 
            cache_file = "u_kn_nk.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    u_kn = data["u_kn"]
                    print(f"MBAR results found. Loaded u_kn from cache. u_kn shape: {u_kn.shape}")
                    N_k_array = data["N_k_array"]
                    compute_ukn = False
            else:
                compute_ukn = True

        # if no results available, compute ukn matrix        
        if compute_ukn:

            print("No cache found. Computing u_kn and N_k_array...")

            # load trajectories from all lambda states
            for lamb_k in tqdm(lambda_values_processed, desc="Loading trajectories"):
                trajectory_file = trajectory_template.format(lamb_k)
                try:
                    samples = md.load_dcd(trajectory_file, top=pdb_file)
                    frames_to_exclude = len(samples) // 5
                    print(f"Excluding {frames_to_exclude} frames from the beginning of the trajectory.")
                    samples = samples[frames_to_exclude::every_nth_frame]
                    print(f"N_k (number of samples used for dG calculation) : {len(samples)}")  # Number of samples
                    all_samples.append(samples)
                    all_box_vectors.extend(samples.unitcell_vectors)
                    N_ks.append(len(samples))
                except Exception as e:
                    print(f"Error loading trajectory for lambda {lamb_k}: {e}")
                    return None

            # Flatten samples into a single array X
            X = np.concatenate([samples.xyz for samples in all_samples], axis=0)

            # Initialize u_kn matrix
            N_total = np.sum(np.array(N_ks)) # K * N_ks[0]  # Total number of samples
            u_kn = np.zeros((K, N_total))  # Shape (K, K * N_k)
  
            # Populate u_kn
            for k, lamb_k in enumerate(tqdm(lambda_values_processed, desc="Evaluating u_kn")):
                # Load potential for lambda=k
                evaluation_potential = MLPotential(
                    "mace-alch", 
                    atom_groups=atom_groups, 
                    lamb=lamb_k,
                    shifting_style=shifting_style,
                    size=size,
                    default_dtype=default_dtype
                )
                evaluation_system = evaluation_potential.createSystem(pdb.topology)
                integrator = VerletIntegrator(0.001)

                context = Context(evaluation_system, integrator)

                # iterate over all samples and evaluate the energy for the current lambda
                for n, x in enumerate(
                    tqdm(X, desc=f"Evaluating lambda {lamb_k}", leave=False)
                ):
                    # Retrieve the box vectors for the current frame
                    # Set the box vectors
                    vec = all_box_vectors[n]
                    context.setPeriodicBoxVectors(
                        Vec3(*vec[0]), Vec3(*vec[1]), Vec3(*vec[2])
                    )

                    # Get the potential energy
                    context.setPositions(x * unit.nanometers)
                    potential_energy = context.getState(getEnergy=True).getPotentialEnergy()
                    u_kn[k, n] = beta * potential_energy

            # Number of samples for each lambda
            N_k_array = N_ks  
            # save ukn and nk
            if method == "mbar" or method == "mbar_filtered":
                cache_file = f"u_kn_nk.pkl"
                print(f"Saving u_kn and N_k_array to {cache_file}...")
                with open(cache_file, "wb") as f:
                    pickle.dump({"u_kn": u_kn, "N_k_array": N_k_array}, f)
            elif method == "bar":
                cache_file = f"u_kn_Nk_{lambda_pair[0]:.8f}_{lambda_pair[1]:.8f}.pkl"
                print(f"Saving u_kn and N_k_array to {cache_file}...")
                with open(cache_file, "wb") as f:
                    pickle.dump({"u_kn": u_kn, "N_k_array": N_k_array}, f)
                
        try:
            mbar = pymbar.MBAR(u_kn, N_k_array, initialize='BAR', verbose=True)
            if method == "mbar":
                overlap_matrix = mbar.compute_overlap()["matrix"]
                weight_matrix = mbar.W_nk

                print("Plotting overlap matrix...")
                plot_overlap(overlap_matrix, lambda_values, f"overlap_matrix_{method}.png")
                print("Plotting weight matrix...")
                plot_weight_matrix(weight_matrix, lambda_values, weight_matrix.shape[1], f"weight_matrix_{method}.png")
                
                r = mbar.compute_free_energy_differences()
                print(f"Free energy differences: {r['Delta_f']}")
                print(f"Uncertainties: {r['dDelta_f']}")
                write_results_to_file(r["Delta_f"], r["dDelta_f"], lambda_values, f"results_{method}.txt")

                total_dG_kT = np.sum(r["Delta_f"][0,-1])
                total_dG_kcal = total_dG_kT * kBT_kcal  

                print("\n##################################################")
                print(f"dG: {total_dG_kT:.2f} kT")
                print(f"dG: {total_dG_kcal:.2f} kcal/mol")
                return
            elif method == "mbar_filtered":

                weight_matrix = mbar.W_nk
                # find indices that cause high weights
                results, exclude_indices = find_weights_above_threshold(weight_matrix, start_index=start_index, lambda_range=lambda_range)
                
                # exclude samples from u_kn
                u_kn_filtered = np.delete(u_kn, exclude_indices, axis=1)  # delete columns = samples

                # adjust N_k_array (samples per state)
                samples_per_state = u_kn.shape[1] // u_kn.shape[0]  
                N_k_array = np.array([samples_per_state] * u_kn.shape[0])

                for n in exclude_indices:
                    k_state = n // samples_per_state
                    N_k_array[k_state] -= 1  # one sample removed from that state

                mbar = pymbar.MBAR(u_kn_filtered, N_k_array, initialize='BAR', verbose=True)

                overlap_matrix = mbar.compute_overlap()["matrix"]
                weight_matrix = mbar.W_nk

                print("Plotting overlap matrix...")
                plot_overlap(overlap_matrix, lambda_values, f"overlap_matrix_{method}.png")
                print("Plotting weight matrix...")
                plot_weight_matrix(weight_matrix, lambda_values, weight_matrix.shape[1], f"weight_matrix_{method}.png")
                
                r = mbar.compute_free_energy_differences()
                # print(f"Free energy differences: {r['Delta_f']}")
                # print(f"Uncertainties: {r['dDelta_f']}")
                write_results_to_file(r["Delta_f"], r["dDelta_f"], lambda_values, f"results_{method}.txt")
                
                total_dG_kT = np.sum(r["Delta_f"][0,-1])
                total_dG_kcal = total_dG_kT * kBT_kcal  

                print("\n##################################################")
                print(f"dG: {total_dG_kT:.2f} kT")
                print(f"dG: {total_dG_kcal:.2f} kcal/mol")
                return
                
            elif method == "bar":

                r = mbar.compute_free_energy_differences()
                # print(f"Free energy differences: {r['Delta_f']}")
                # print(f"Uncertainties: {r['dDelta_f']}")
                pairwise_dGs.append(r)
            
        except Exception as e:
            raise RuntimeError(f"Error during {method} calculation: {e}")
                
    total_pairwise_dG = sum([r["Delta_f"][0, 1] for r in pairwise_dGs])
    print("\n##################################################")
    print(f"Sum of pairwise dGs: {total_pairwise_dG:.2f} kT")
    write_pairwise_results_to_file(pairwise_dGs, pairwise_lambdas, output_file="pairwise_results_BAR.txt")         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate free energy difference.")

    # Command-line arguments
    parser.add_argument(
        "--pdb_file", type=str, required=True, help="Path to the PDB file."
    )
    parser.add_argument(
        "--trajectory_template",
        type=str,
        required=True,
        help="Template for trajectory file names, e.g., 'trajectory_lambda_{:.1f}.dcd'.",
    )
    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs="+",
        required=True,
        help="List of lambda values, e.g., 0.0 0.1 0.2 ...",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300,
        help="Temperature in Kelvin (default: 300).",
    )
    parser.add_argument(
        "--every_nth_frame",
        type=int,
        default=1,
        help="Take every nth frame for analysis (default: take every frame).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bar", "mbar", "mbar_filtered"],
        required=True,
        help="Method for calculating free energy difference: choose from 'bar', 'mbar', or 'mbar_filtered'.",
    )
    parser.add_argument(
        "--start_index", 
        type=int, 
        default=None, 
        help="Only for mbar_filtered: Start index of sample where filtering will be , e.g., start the filtering process at sample 6000."
    )
    parser.add_argument(
        "--lambda_range", 
        type=int, 
        nargs=2, 
        metavar=("START", "END"),
        default=None,
        help="Only for mbar_filtered: Lambda range (inclusive) where filtering will be applied, e.g., --lambda_range 0 4."
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium"],
        required=True,
        help="Size of the model.",
    )
    parser.add_argument(
        "--shifting_style",
        type=str,
        choices=["linear", "linear_to_cutoff", "4D", "4D_to_cutoff"],
        required=True,
        help="Shifting style.",
    )
    parser.add_argument(
        "--default_dtype",
        type=str,
        choices=["float32", "float64"],
        required=True,
        help="Precision of the model.",
    )

    args = parser.parse_args()

    calculate_free_energies(
        pdb_file=args.pdb_file,
        trajectory_template=args.trajectory_template,
        lambda_values=np.array(args.lambda_values),
        temperature=args.temperature,
        every_nth_frame=args.every_nth_frame,
        method=args.method,
        size=args.size,
        shifting_style=args.shifting_style,
        default_dtype=args.default_dtype,
        start_index=args.start_index if args.method == "mbar_filtered" else None,
        lambda_range=tuple(args.lambda_range) if args.method == "mbar_filtered" else None,
    )
