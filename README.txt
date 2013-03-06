tomography
==========

Breast cancer tomography imaging objective function code.

There is a submodule (and a sub-submodule in the project).  After cloning the git repository, do:

git submodule init
git submodule update

This should checkout the code for the tao package.

Then do:

cd tao

git submodule init
git submodule update

Which will checkout the code for the undvc_common submodule of the tao package.

To compile (make sure cmake and boost are installed):

mkdir build
cd build
cmake ..
make

This will create both fdtd_cpu and fdtd_gpu binaries (for the cpu and gpu respectively).

Then to run:

./fdtd_cpu --search_type <ps, de, anm>

for either particle swarm, differential evolution or asynchronous newton method.

Each will spit out other command line parameters for tweaking how the search runs, eg:

Travis-Desells-MacBook-Pro:build deselt$ ./fdtd_cpu --search_type ps
parsed argument '--search_type' successfully: ps
Argument '--population_size' not specified, using default of 200.
Argument '--maximum_iterations' not specified, could run forever. Hit control-C to quit.
Argument '--maximum_created' not specified, could run forever. Hit control-C to quit.
Argument '--maximum_reported' not specified, could run forever. Hit control-C to quit.
Argument '--inertia <F>' not found, using default of 0.75.
Argument '--global_best_weight <F>' not found, using default of 1.5.
Argument '--local_best_weight <F>' not found, using default of 1.5.
Argument '--initial_velocity_scale <F>' not found, using default of 0.25.
Initialized partilce swarm.
    maximum_iterations: 0
    current_iteration:  0
    inertia:            0.75
    global_best_weight: 1.5
    local_best_weight:  1.5

Travis-Desells-MacBook-Pro:build deselt$ ./fdtd_cpu --search_type de
parsed argument '--search_type' successfully: de
Argument '--population_size' not specified, using default of 200.
Argument '--maximum_iterations' not specified, could run forever. Hit control-C to quit.
Argument '--maximum_created' not specified, could run forever. Hit control-C to quit.
Argument '--maximum_reported' not specified, could run forever. Hit control-C to quit.
Argument '--parent_scaling_factor <F>' not found, using default of 1.0.
Argument '--differential_scaling_factor <F>' not found, using default of 1.0.
Argument '--crossover_rate <F>' not found, using default of 0.5.
Argument '--number_pairs <I>' not found, using default of 1.
Argument '--parent_selection <S>' not found, using default of 'best'.
Argument '--recombination_selection <S>' not found, using default of 'binary'.
Initialized differential evolution. 
    maximum_iterations:          0
    current_iteration:           0
    number_pairs:                1
    parent_selection:            0
    recombination_selection:     0
    parent_scaling_factor:       1
    differential_scaling_factor: 1
    crossover_rate:              0.5
    directional:                 0


It will then spit out the globally best found fitnesses (and the parameters for that fitness) as it finds them.
