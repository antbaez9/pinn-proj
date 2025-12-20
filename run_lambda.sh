#!/bin/bash

NUM_TRIALS=10
EPS_VALUE="1e-6"
seeds=(42 43 44 45 46 47 48 49 50 51)
lambdas=(0 1 10 100)


# Function to run experiments
run_experiments() {
    local eq_name=$1  # Taking equation name directly
    
    for quantity in "momentum" "energy" "both"; do

        for variant in "sc"; do

            for lambda_val in "${lambdas[@]}"; do

                local filename=""
                local flag=""
                local variant_name=""
                
                if [[ "$variant" == "proj" ]]; then
                    filename="pinn_proj.py"
                    flag="--proj"
                    variant_name="proj"
                elif [[ "$variant" == "sc" ]]; then
                    filename="pinn_sc.py"
                    flag=""
                    variant_name="sc"
                else  # base case
                    filename="pinn_proj.py"
                    flag=""
                    variant_name="base"
                fi

                filename="${filename%.py}_lbfgs.py"
                for i in $(seq 1 $NUM_TRIALS); do
                    # echo "$eq_name $quantity pinn-$variant_name lambda=$lambda_val $i/$NUM_TRIALS with seed ${seeds[$((i-1))]}..."
                    
                    # Construct and print the CLI command
                    local cmd="SEED=${seeds[$((i-1))]} SC_WEIGHT=$lambda_val python $filename --eq \"$eq_name\" --quantity \"$quantity\" $flag --eps $EPS_VALUE"
                    echo "$cmd"
                    
                    # Execute the command
                    SEED=${seeds[$((i-1))]} SC_WEIGHT=$lambda_val python "$filename" --eq "$eq_name" --quantity "$quantity" $flag --eps $EPS_VALUE
                done
            done
        done
    done
}

run_experiments "advection"
run_experiments "wave"
run_experiments "react-diff"
run_experiments "kdv"