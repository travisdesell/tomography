#ifdef CUDA
#include "FDTD_GPU.cuh"
#else
#include "FDTD_CPU.h"
#endif

#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <stdint.h>

#include "examples/benchmarks.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"
#include "asynchronous_algorithms/asynchronous_newton_method.hxx"

//from undvc_common
#include "undvc_common/arguments.hxx"

using namespace std;

int main(int argc, char **argv) {
    vector<string> arguments(argv, argv + argc);

    /*
    srand48(time(NULL));

    vector<double> parameters;
    for (int i = 0; i < 81; i++) {
        parameters.push_back(1.0 + (39.0 * drand48()));

        cout << "parameters[" << i << "]: " << parameters.back() << endl;
    }   

    for (int i = 0; i < 81; i++) {
        parameters.push_back(1.1 * drand48());

        cout << "parameters[" << i + 81 << "]: " << parameters.back() << endl;
    }   

    long start_time = time(NULL);
    cout << "fitness: " << FDTD_GPU(parameters) << endl;
    cout << "took " << (time(NULL) - start_time) << " seconds" << endl;
    */

    uint32_t number_of_parameters = 162;
    vector<double> min_bound(number_of_parameters, 0);
    vector<double> max_bound(number_of_parameters, 0);
    vector<double> radius(number_of_parameters, 0);

    for (uint32_t i = 0; i < number_of_parameters; i++) {        //arrays go from 0 to size - 1 (not 1 to size)
        
        if (i < 81) {
            radius[i] = 1;
            min_bound[i] = 1.0;
            max_bound[i] = 40.0;
        } else {
            radius[i] = 0.05;
            min_bound[i] = 0.0;
            max_bound[i] = 1.1;
        }
    }

    string search_type;
    get_argument(arguments, "--search_type", true, search_type);
    if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
#ifdef CUDA
        ps.iterate(FDTD_GPU);
#else
        ps.iterate(FDTD_CPU);
#endif

    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
#ifdef CUDA
        de.iterate(FDTD_GPU);
#else
        de.iterate(FDTD_CPU);
#endif

    } else if (search_type.compare("anm") == 0) {
        AsynchronousNewtonMethod anm(min_bound, max_bound, radius, arguments);
#ifdef CUDA
        anm.iterate(FDTD_GPU);
#else
        anm.iterate(FDTD_CPU);
#endif

    } else {
        cerr << "Improperly specified search type: '" << search_type.c_str() <<"'" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    de     -       differential evolution" << endl;
        cerr << "    ps     -       particle swarm optimization" << endl;
        exit(1);
    }

}

