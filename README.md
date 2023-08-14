## Electric field equation of the Jefimenko's equations.


- `jefimenko.py`: Numerically solve the electric field component of the 
Jefimenko's equations for a given sequence of observers.
Usage: `python jefimenko.py <space-charge-file>.npz -observation_points obs1 ... 

- `generate_test_data.py`: Generate data for 2 test cases of sinusoidally 
oscillating charges.
    TODO: Description of the test cases

- `plot_jefimenko_results.py`: Visualize the electric field components from the
`.csv` files generated from executing the `jefimenko.py` script.

- `analyse_signal.py`: Script to analyise and visualize the main frequency
components of the electric field using continuous wavelet transform.
This script requires the installation of the **scaleogram** pacakge from the
following repository [https://github.com/alsauve/scaleogram.git].
