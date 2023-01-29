#!/bin/bash

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_sn.txt
	#BSUB -o mystdout_sn.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 6:00                    #walltime in minutes
	#BSUB -J smallnets          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST3=ignition_grid/small_networks/small_network_2.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid

    echo "=== STARTING JOB ==="  
    jsrun -n 1 -r 1 -a 4 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST3 --default_root_dir $SAVE --data_dir $IG_GRID_DATA
