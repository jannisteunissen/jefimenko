#!/bin/bash

if [ $# -lt 2 ]; then
   echo "Number of arguments should at least be 2."
   echo "$0 resolution <all silo files>"
   exit 1
fi

# Arguments: resolution <all silo files>
resolution=$1
shift
silo_files="$@"

script_dir=/export/scratch2/hemadity/afivo-streamer_new_rep_pulses/afivo/tools/
vars="rhs Je_1 Je_2 Je_3"
r_min="0.06 0.06 0.05"
r_max="0.14 0.14 0.2"

(cd "$script_dir";
 parallel -j 4 ./plot_raw_data.py {1} -variable {2} -save_npz \
          {1.}_uniform_${resolution}_{2}.npz \
          -min_pixels $resolution \
          -interpolation nearest \
          -r_min ${r_min} -r_max ${r_max} \
          ::: $silo_files ::: $vars
)
