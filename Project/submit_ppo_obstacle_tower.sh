#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=12:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 8    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

Xvfb :3 -screen 0 1024x768x24 & # Create virtual display so the remote machine can run the python script
export DISPLAY=:3               # Redirect Unity to the virtual screen created

python train_rl.py
