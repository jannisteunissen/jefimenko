#!/bin/bash

simulation_name=/export/scratch2/teunisse/tmp/hema_radio_3d
silo_files=${simulation_name}_*.silo
script_dir=~/git/afivo/tools/
vars="rhs Je_1 Je_2 Je_3"
resolution=256

(cd "$script_dir";
 parallel -j 6 ./plot_raw_data.py {1} -variable {2} -save_npz \
          {1.}_uniform_${resolution}_{2}.npz \
          -min_pixels $resolution \
          -interpolation nearest \
          -r_min 0.05 0.05 0.05 -r_max 0.15 0.15 0.15 \
          ::: $silo_files ::: $vars
 )

# python combine_data.py ${simulation_name}_*_uniform_${resolution}*.npz \
#        -log_file ${simulation_name}_log.txt -output test_${resolution}.npz
