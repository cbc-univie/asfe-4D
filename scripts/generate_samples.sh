#!/bin/bash

# Base directory
base_dir="/home/mwieder/Work/Projects/asfe/data"

# System name
system_name="toluene"

# List of lamb values
lamb_values=( 0.0 0.00480769 0.01923077 0.04326923 0.07692308 0.12019231 0.17307692 0.23557692 0.30769231 0.38942308 0.48076923 0.58173077 0.69230769 0.8125 1.0 ) # np.linspace(0, 1, num_states)**2

# Input PDB file
pdb_file="../input/${system_name}_waterbox.pdb"

# Script path
script_path="../../../scripts/sample_states.py"

# SLURM submission script generation
for lamb in "${lamb_values[@]}"; do
    # Create the output directory if it doesn't exist
    output_dir="${base_dir}/${system_name}_waterbox/trajs/"
    mkdir -p "$output_dir"

    # Change to the output directory
    cd "$output_dir" || exit

    # SLURM job script name
    job_script="job_r${lamb}.slurm"

    # Create SLURM job script
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=r${lamb}_${system_name}   # Job name
#SBATCH --output=r${lamb}.log                # Output log file
#SBATCH --error=r${lamb}.err                 # Error log file
#SBATCH --partition=long                     # GPU partition name
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=2                    # Number of CPU cores per task
#SBATCH --mem=32G                            # Memory per node
#SBATCH --nice=100                           # Number of tasks
#SBATCH --ntasks=1                           # Number of tasks

# Load any required modules
source /data/shared/projects/mamba/etc/profile.d/conda.sh
conda activate alch

# Execute the script
python $script_path --lamb $lamb --pdb $pdb_file
EOF

    # Submit the job to SLURM
    echo $job_script
    sbatch $job_script
done

echo "All jobs have been submitted to SLURM."
