import numpy as np
import pymbar
from openmm import unit, VerletIntegrator, Context
from openmm.app import PDBFile, DCDFile
from openmmml import MLPotential
from openmm.vec3 import Vec3
import torch
import argparse
import mdtraj as md
from tqdm import tqdm  # For progress visualization
import matplotlib.pyplot as plt
import seaborn as sns  # Optional for nicer heatmaps
import warnings
warnings.filterwarnings("ignore")

def write_results_to_file(results, uncertainties, lambda_values, output_file="results.txt"):
    """
    Write the calculated free energy results to a file.

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


def plot_overlap(overlap_matrix, lambda_values):
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
    plt.savefig("overlap_matrix.png")
    plt.show()


def plot_weight_matrix(W_nk, lambda_values, sample_count):
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
    sns.heatmap(
        W_nk,
        cmap="viridis",
        cbar=True,
        #xticklabels=[f"Lambda {i}" for i in range(sample_count // len(lambda_values))],
    )
    plt.title("Weight Matrix (W_nk)")
    plt.ylabel("Sample Index")
    plt.xlabel("Lambda State")
    plt.savefig("weight_matrix.png")
    plt.show()


def calculate_free_energies(
    pdb_file: str,
    trajectory_template: str,
    lambda_values: np.ndarray,
    temperature: float = 300,
    every_nth_frame: int = 1,
):
    """
    Calculate free energies using MBAR and OpenMM simulations.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    trajectory_template : str
        Template for trajectory file names, e.g., "trajectory_lambda_{:.1f}.dcd".
    lambda_values : np.ndarray
        Array of lambda values.
    atom_groups_definition : list
        List defining alchemical region, e.g., [0, 6].
    temperature : float, optional
        Temperature in Kelvin, default is 300.

    Returns
    -------
    tuple
        Delta free energies and uncertainties (deltaF, ddeltaF).
    """
    print("Calculating free energies using MBAR...")
    print(f"Lambda values: {lambda_values}")
    # Constants
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

    print(f"Topology atoms: {[a.residue.name for a in list(pdb.topology.atoms())]}")
    print(f"Atom groups: {atom_groups}")
    # Check if there are any entries with value 1
    if not torch.any(atom_groups == 1):
        print("Error: No atoms were assigned to group 1 (e.g., solvent). Exiting.")
        exit()

    # Number of lambda states
    K = len(lambda_values)
    # Collect all samples into a single array
    all_samples = []
    all_box_vectors = []
    N_ks = []
    import pickle
    import os

    cache_file = "u_kn.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            u_kn = data["u_kn"]
            N_k_array = data["N_k_array"]
    else:
        # Compute u_kn and N_k_array
        print("No cache found. Computing u_kn and N_k_array...")

        for lamb_k in tqdm(lambda_values, desc="Loading trajectories"):
            trajectory_file = trajectory_template.format(lamb_k)
            try:
                samples = md.load_dcd(trajectory_file, top=pdb_file)
                frames_to_exclude = len(samples) // 5
                samples = samples[frames_to_exclude::every_nth_frame]
                print(f"N_k : {len(samples)}")  # Number of samples
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
        # save pdb file from frame 10 of samples
        # samples[10].save("first_frame.pdb")
        # Populate u_kn
        for k, lamb_k in enumerate(tqdm(lambda_values, desc="Evaluating u_kn")):
            # Load potential for lambda=k
            evaluation_potential = MLPotential(
                "mace", atom_groups=atom_groups, lamb=lamb_k,
            )
            evaluation_system = evaluation_potential.createSystem(pdb.topology)
            integrator = VerletIntegrator(0.001)
            context = Context(evaluation_system, integrator)

            # iterate over all samples
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
        # Save to pickle
        print(f"Saving u_kn and N_k_array to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump({"u_kn": u_kn, "N_k_array": N_k_array}, f)

    # Use pymbar to calculate free energies
    try:
        mbar = pymbar.MBAR(u_kn, N_k_array, initialize='BAR', verbose=True)
        overlap_matrix = mbar.compute_overlap()["matrix"]
        weight_matrix = mbar.W_nk
        # Plot overlap and weight matrix
        print("Plotting overlap matrix...")
        plot_overlap(overlap_matrix, lambda_values)
        print("Plotting weight matrix...")
        plot_weight_matrix(weight_matrix, lambda_values, weight_matrix.shape[1])

        r = mbar.compute_free_energy_differences()
        print(f"Free energy differences: {r['Delta_f']}")
        print(f"Uncertainties: {r['dDelta_f']}")
        write_results_to_file(r["Delta_f"], r["dDelta_f"], lambda_values)
        return r["Delta_f"], r["dDelta_f"]
    except Exception as e:
        raise RuntimeError(f"Error during MBAR calculation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate free energies using MBAR.")

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
        help="Take every nth frame for analysis.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the function
    deltaF, ddeltaF = calculate_free_energies(
        pdb_file=args.pdb_file,
        trajectory_template=args.trajectory_template,
        lambda_values=np.array(args.lambda_values),
        temperature=args.temperature,
        every_nth_frame=args.every_nth_frame,
    )
