#!/bin/bash
#SBATCH --job-name=analysis   # Job name
#SBATCH --output=r.log                # Output log file
#SBATCH --error=r.err                 # Error log file
#SBATCH --partition=long                     # GPU partition name
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=2                    # Number of CPU cores per task
#SBATCH --mem=32G                            # Memory per node
#SBATCH --nice=100                           # Number of tasks
#SBATCH --ntasks=1                           # Number of tasks

# Load any required modules
source /data/shared/projects/mamba/etc/profile.d/conda.sh
conda activate alch

# Input arguments: system name and version
system_name=$1
version=$2

# Print the system name
echo "#################################################"
echo "Running analysis for system: $system_name"
echo "#################################################"

# Check if inputs are provided
if [ -z "$system_name" ] || [ -z "$version" ]; then
    echo "Usage: $0 <system_name> <version>"
    exit 1
fi

# Lock file setup
lock_file="${system_name}_${version}.lock"

# Check if the lock file already exists
if [ -f "$lock_file" ]; then
    echo "System ${system_name} is already being processed. Exiting."
    exit 1
fi

# Create the lock file
touch "$lock_file"

# Trap to remove the lock file in case of script termination
trap "rm -f $lock_file" EXIT

base='/home/mwieder/Work/Projects/asfe/'
# cd to the working directory
mkdir -p ${base}/data/${system_name}_waterbox/analysis_${version}
cd ${base}/data/${system_name}_waterbox/analysis_${version}

# Paths
input_dir="../input"
traj_dir="../trajs"

# PDB file
pdb_file="${input_dir}/${system_name}_waterbox.pdb"

# Trajectory template
trajectory_template="${traj_dir}/trajectory_lambda_{:.4f}_${version}.dcd"

# Lambda values
lamb_values=( 0.0 0.00480769 0.01923077 0.04326923 0.07692308 0.12019231 0.17307692 0.23557692 0.30769231 0.38942308 0.48076923 0.58173077 0.69230769 0.8125 1.0 ) # np.linspace(0, 1, num_states)**2

# Temperature
temperature=300
# Load any required modules
source /data/shared/projects/mamba/etc/profile.d/conda.sh
conda activate alch

# Script path
script_path="../../../scripts/calculate_asfe.py"

# Run the Python script
python "$script_path" \
    --pdb_file "$pdb_file" \
    --trajectory_template "$trajectory_template" \
    --lambda_values "${lamb_values[@]}" \
    --temperature "$temperature" \
    --every_nth_frame 4

# Clean up the lock file on successful completion
rm -f "$lock_file"
