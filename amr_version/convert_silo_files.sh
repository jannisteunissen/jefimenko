#!/bin/bash

set -e

if (($# != 3)); then
   echo "usage: $0 path/simulation_name start_index end_index"
   exit 1
fi

afivo_dir=$HOME/git/afivo

vars=('rhs' 'Je_1' 'Je_2' 'Je_3')

for ((i="$2"; i<="$3"; i++)); do
    file_a=$(printf "$1_%06d.silo" $((i-1)))
    file_b=$(printf "$1_%06d.silo" $((i)))
    file_c=$(printf "$1_%06d.silo" $((i+1)))

    for v in "${vars[@]}"; do
        raw_file=$(printf "$1_d1_${v}_%06d.raw" $i)
        ${afivo_dir}/tools/compute_silo_derivative.py \
                    "$file_a" "$file_c" -deriv_type 1st_central \
                    -output "$raw_file" -variable "$v" &

        raw_file=$(printf "$1_d2_${v}_%06d.raw" $i)
        ${afivo_dir}/tools/compute_silo_derivative.py \
                    "$file_a" "$file_b" "$file_c" -deriv_type 2nd_central \
                    -output "$raw_file" -variable "$v" &
        wait
    done
done
