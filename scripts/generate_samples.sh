#!/bin/bash

# Base directory
base_dir="../data"

# System name
system_name="ethane"
model_size='small' #medium
default_dtype='float32' #float64
shifting_style="linear_to_cutoff" #('4D' '4D_to_cutoff' 'linear' 'linear_to_cutoff')
run=1
timestep=0.001
tag="v1"

# Script path
script_path="../../../../scripts/sample_states.py"

# Input PDB file
############ for production runs
pdb_file="../../input/${system_name}_waterbox_equil.pdb"
output_dir="${base_dir}/${system_name}/${shifting_style}_${default_dtype}_run${run}/trajs/"
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

cd "$output_dir" || exit

# SLURM submission script generation
for lamb in "${lamb_values[@]}"; do
    # Change to the output directory

    # SLURM job script name
    job_script="job_r${lamb}_${shifting_style}_${tag}.slurm"

    # Create SLURM job script
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=r${lamb}_${system_name}_${shifting_style}_${tag}   # Job name
#SBATCH --output=r${lamb}_${tag}.log                # Output log file
#SBATCH --error=r${lamb}_${tag}.err                 # Error log file
#SBATCH --p 4090

# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alchemical_nnp

# Execute the script
python $script_path --lamb $lamb --pdb $pdb_file --shifting $shifting_style --model_size $model_size --default_dtype $default_dtype --timestep $timestep --tag $tag
EOF

    # Submit the job to SLURM
    echo $job_script
    sbatch $job_script
done

echo "All jobs have been submitted to SLURM."