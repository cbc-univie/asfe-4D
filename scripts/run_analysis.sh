#!/bin/bash
#SBATCH --job-name=methanol_analysis 
#SBATCH --output=methanol_analysis.log
#SBATCH --error=methanol_analysis.err
#SBATCH -p 4090

# Load Conda ########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alchemical_nnp

base_dir='asfe-4D'
shifting_style="linear_to_cutoff"
default_dtype="float32"
run=1

###########################
# Parse labeled arguments #
###########################
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --system_name) system_name="$2"; shift 2 ;;
        --version) version="$2"; shift 2 ;;
        --method) method="$2"; shift 2 ;;
        --start_index) start_index="$2"; shift 2 ;;
        --lambda_range)
            lambda_range_start="$2"
            lambda_range_end="$3"
            shift 3
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

###########################
# Validate required inputs
###########################
if [[ -z "$system_name" || -z "$version" || -z "$method" ]]; then
    echo "Error: --system_name, --version, and --method are required."
    exit 1
fi

if [[ "$method" == "mbar_filtered" ]]; then
    if [[ -z "$start_index" || -z "$lambda_range_start" || -z "$lambda_range_end" ]]; then
        echo "Error: --start_index and --lambda_range are required for method=mbar_filtered."
        exit 1
    fi
fi


###########################
# Setup directories
###########################
echo "#################################################"
echo "Running analysis for system: $system_name"
echo "#################################################"

lock_file="${system_name}_${version}_${method}.lock"
if [ -f "$lock_file" ]; then
    echo "System ${system_name} is already being processed. Exiting."
    exit 1
fi
touch "$lock_file"
trap "rm -f $lock_file" EXIT

mkdir -p ${base_dir}/data/${system_name}/${shifting_style}_${default_dtype}_run${run}/analysis_${version}
cd ${base_dir}/data/${system_name}/${shifting_style}_${default_dtype}_run${run}/analysis_${version}

input_dir="${base_dir}/data/${system_name}/${shifting_style}_${default_dtype}_run${run}/input"
traj_dir="${base_dir}/data/${system_name}/${shifting_style}_${default_dtype}_run${run}/trajs"
pdb_file="${input_dir}/${system_name}_waterbox_equil.pdb"
trajectory_template="${traj_dir}/trajectory_lambda_{:.4f}_${version}.dcd"
lamb_values=( 0.0 0.05263158 0.10526316 0.15789474 0.21052632
       0.26315789 0.31578947 0.36842105 0.42105263 0.47368421 
       0.57894737 0.68421053
       0.78947368 0.89473684 1.0 )

temperature=300

script_path="${base_dir}/scripts/calculate_asfe.py"

###########################
# Build Python command
###########################
cmd=(python "$script_path"
    --pdb_file "$pdb_file"
    --trajectory_template "$trajectory_template"
    --lambda_values "${lamb_values[@]}"
    --temperature "$temperature"
    --every_nth_frame 4
    --method "$method"
    --size "small"
    --shifting_style "$shifting_style"
    --default_dtype "$default_dtype"
)

if [[ "$method" == "mbar_filtered" ]]; then
    cmd+=(--start_index "$start_index" --lambda_range "$lambda_range_start" "$lambda_range_end")
fi

###########################
# Execute Python script
###########################
"${cmd[@]}"
rm -f "$lock_file"