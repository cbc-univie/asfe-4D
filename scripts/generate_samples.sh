#!/bin/bash

# Base directory
base_dir="/site/raid7/anna/4D/asfe-4D/data"

# System name
system_name="methanol"
model_size='small'
default_dtype='float64'
shifting_style="linear_to_cutoff" #('4D' '4D_to_cutoff' 'linear' 'linear_to_cutoff')
run=1
timestep=0.001

# Script path
script_path="${base_dir}/../scripts/sample_states.py"

# Input PDB file
############ for production runs
pdb_file="../../input/${system_name}_waterbox_equil.pdb"
output_dir="${base_dir}/${system_name}/prod_${model_size}_${shifting_style}_${default_dtype}_run${run}/trajs/"
lamb_values=( 0.0 0.05263158 0.10526316 0.15789474 0.21052632
       0.26315789 0.31578947 0.36842105 0.42105263 0.47368421 
       0.57894737 0.68421053
       0.78947368 0.89473684 1.0 )


mkdir -p "$output_dir"
# save settings
save_settings_path="${output_dir}/../"
cat <<EOF > $save_settings_path/save_settings.txt
# Settings file
system_name=$system_name
model_size=$model_size
default_dtype=$default_dtype
shifting_style=$shifting_style
run=$run
pdb=$pdb_file
timestep=$timestep ps
EOF






# SLURM submission script generation
for lamb in "${lamb_values[@]}"; do
    # Change to the output directory
    cd "$output_dir" || exit

    # SLURM job script name
    job_script="job_r${lamb}_${shifting_style}.slurm"

    # Create SLURM job script
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=r${lamb}_${system_name}_${shifting_style}   # Job name
#SBATCH --output=r${lamb}.log                # Output log file
#SBATCH --error=r${lamb}.err                 # Error log file
#SBATCH -p 4090
##SBATCH -p ADA
##SBATCH -p gpu


## SBATCH --partition=long                     # GPU partition name
## SBATCH --gres=gpu:1                         # Request 1 GPU
## SBATCH --cpus-per-task=2                    # Number of CPU cores per task
## SBATCH --mem=32G                            # Memory per node
## SBATCH --nice=100                           # Number of tasks
## SBATCH --ntasks=1                           # Number of tasks

# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 4D_all_shift

# Execute the script
python $script_path --lamb $lamb --pdb $pdb_file --shifting $shifting_style --model_size $model_size --default_dtype $default_dtype --timestep $timestep
EOF

    # Submit the job to SLURM
    echo $job_script
    sbatch $job_script
done

echo "All jobs have been submitted to SLURM."
