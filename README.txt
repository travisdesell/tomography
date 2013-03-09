tomography
===============

Project for optimizing breast tomography imaging.


Compilation instructions:

step 1:
    generate an SSH key for github:

    https://help.github.com/articles/generating-ssh-keys

    you should already have one in .ssh/id_rsa.pub you can add this to your
    git account under account settings.

step 2:
    add cmake to your .bashrc

    alias cmake=/share/apps/usrcmake/bin/cmake ..

    log out and log back in

step 3:
    clone and build the repository:
    git clone ssh://git@github.com/travisdesell/tomography.git

    cd tomography
    git submodule init
    git submodule update
    git checkout master
    cd tao
    git submodule init
    git submodule update
    git checkout master

    cd ..
    mkdir build
    cd build
    make

step 4:
    Run:

    For two processes (master/worker) for testing:
        mpirun -np 2 ./fdtd_cpu --search_type ps

    To run on hodor, you can use the hodor_submit.pbs script.  You'll need to update this for your home directory and own email.
    The tomography directory needs to be in your home directory, eg: ~/tomography

    Then submit with:
        sub hodor_submit.pbs

    You can track the job status with:
        qstat -a

    For more information on the job submission script and things like that,
    see the shale instructions:
    http://und.edu/research/computational-research-center/shale-sub-scripts.cfm

    It will also email you when your job starts running and when it completes.  The output will be in:
    ~/tomography/build/hmmc_out.txt
    and
    ~/tomography/build/hmmc_error.txt
