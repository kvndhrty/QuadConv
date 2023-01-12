#!/bin/bash

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test5.txt
	#BSUB -o mystdout_test5.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test5          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST5=ignition_mesh/paper/pvcae.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs
    IG_MESH_DATA=/usr/workspace/doherty8/data/ignition_mesh

    echo "=== STARTING JOB ==="  
    jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST5 --default_root_dir $SAVE --data_dir $IG_MESH_DATA