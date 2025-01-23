import os
import argparse
import socket  # For getting the hostname
from openmmml import MLPotential
from openmm.app import Simulation, StateDataReporter, PDBFile, DCDReporter
from openmm import LangevinIntegrator, unit, Platform, MonteCarloBarostat
import torch
import warnings
warnings.filterwarnings("ignore")

tag = "v4_1"

# Print the hostname of the node
hostname = socket.gethostname()
print(f"Running on node: {hostname}")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an OpenMM simulation with MLPotential."
)
parser.add_argument(
    "--lamb", type=float, required=True, help="Lambda value for the simulation."
)
parser.add_argument(
    "--pdb", type=str, required=True, help="Path to the input PDB file."
)
args = parser.parse_args()

# Command-line inputs
lamb = args.lamb
pdb_file = args.pdb

# File names
trajectory_filename = f"trajectory_lambda_{lamb:.4f}_{tag}.dcd"
energy_filename = f"output_lambda_{lamb:.4f}_{tag}.csv"

# Check if the trajectory file already exists
if os.path.exists(trajectory_filename):
    print(
        f"Trajectory file {trajectory_filename} already exists. Skipping lambda = {lamb}."
    )
    exit()

# Load the PDB file
pdb = PDBFile(pdb_file)

# Define atom groups for the alchemical region
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

# Simulation parameters
temperature = 300 * unit.kelvin
pressure = 1.0 * unit.atmosphere
friction = 1.0 / unit.picoseconds
timestep = 0.001 * unit.picoseconds
simulation_time = 1.0 * unit.nanoseconds
save_interval = 0.25 * unit.picoseconds

# Initialize the CUDA platform
platform = Platform.getPlatformByName("CUDA")

# Create the MLPotential
print(f"Running simulation for lambda = {lamb}...")
potential = MLPotential(
    "mace",
    # modelPath="alchemical_mace_small_v0.pt",
    atom_groups=atom_groups,
    lamb=lamb,
)

# Create the OpenMM system
system = potential.createSystem(pdb.topology)

# Ensure periodic boundary conditions are enabled
assert (
    system.usesPeriodicBoundaryConditions()
), "Periodic boundary conditions are not enabled."
# Add a barostat for constant pressure
system.addForce(MonteCarloBarostat(pressure, temperature))

# Set up the Langevin integrator
integrator = LangevinIntegrator(temperature, friction, timestep)

# Set up the simulation
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Set up reporters for DCD and energy data
simulation.reporters.append(
    DCDReporter(trajectory_filename, int(save_interval / timestep), enforcePeriodicBox=True)
)
# Run the simulation for 1 ns
simulation_steps = int(simulation_time / timestep)
print(f"Running {simulation_steps} steps ({simulation_time}) with lambda = {lamb}...")


simulation.reporters.append(
    StateDataReporter(
        energy_filename,
        int(save_interval / timestep),
        time=True,
        potentialEnergy=True,
        volume=True,
        density=True,
        speed=True,
        remainingTime=True,
        totalSteps=simulation_steps,
        temperature=True,
        progress=True,
    )
)

simulation.step(simulation_steps)
print("Simulation complete.")

# Get the final potential energy
state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

print(f"Simulation complete for lambda = {lamb}.")
print(f"Trajectory saved to {trajectory_filename}")
print(f"Energy data saved to {energy_filename}")

final_positions = simulation.context.getState(getPositions=True).getPositions()
final_pdb_filename = f"final_state_lambda_{lamb:.4f}_{tag}.pdb"

with open(final_pdb_filename, 'w') as final_pdb_file:
    PDBFile.writeFile(pdb.topology, final_positions, final_pdb_file)

print(f"Final state saved to {final_pdb_filename}")
