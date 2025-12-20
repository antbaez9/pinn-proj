#!/bin/bash

NUM_TRIALS=10
EPS_VALUE="1e-7"
seeds=(42 43 44 45 46 47 48 49 50 51)

# List of data files to iterate through
data_files=('advection_solution_2d.npy')

# Function to run experiments
run_experiments() {
    local eq_name=$1  # Taking equation name directly
    
    for data_file in "${data_files[@]}"; do
        echo "File: $data_file"
        
        for quantity in "momentum" "energy" "both"; do
            for variant in "base" "sc" "proj"; do

                local filename=""
                local flag=""
                local variant_name=""
                
                if [[ "$variant" == "proj" ]]; then
                    filename="pinn_proj_lbfgs_advection_2d.py"
                    flag="--proj"
                    variant_name="proj"
                elif [[ "$variant" == "sc" ]]; then
                    filename="pinn_sc_lbfgs_advection_2d.py"
                    flag=""
                    variant_name="sc"
                else  # base case
                    filename="pinn_proj_lbfgs_advection_2d.py"
                    flag=""
                    variant_name="base"
                fi

                for i in $(seq 1 $NUM_TRIALS); do
                    # echo "$eq_name $quantity pinn-$variant_name $i/$NUM_TRIALS with seed ${seeds[$((i-1))]} using $data_file..."
                    
                    # Construct and print the CLI command with data file
                    local cmd="SEED=${seeds[$((i-1))]} python $filename --eq \"$eq_name\" --quantity \"$quantity\" $flag --eps $EPS_VALUE --data \"$data_file\""
                    echo "$cmd"
                    
                    # Execute the command with data file
                    SEED=${seeds[$((i-1))]} python "$filename" --eq "$eq_name" --quantity "$quantity" $flag --eps $EPS_VALUE --data "$data_file"
                done
            done
        done
    done
}

run_experiments "advection_2d"