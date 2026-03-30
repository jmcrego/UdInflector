#!/bin/bash

#SBATCH --job-name=udinflector
#SBATCH --output=./logs/%u.%x.%j.out
#SBATCH --error=./logs/%u.%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00

### uncomment one of the next ### 
#
#SBATCH -C h100
#SBATCH -A eut@h100
#
##SBATCH -C v100-32g
##SBATCH -A eut@v100

#copy data from macbook:
#scp -v -3 UD-enfr.tsv ujt99zo@jean-zay.idris.fr:/linkhome/rech/genata01/ujt99zo/work/josep/UdInflector/

module purge
#
module load arch/h100 #this module must be loaded before pytorch-gpu                                                                                                                                                                                    
module load pytorch-gpu/py3/2.6.0

MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-32B #Qwen3-32B Qwen3-8B
TSV=resources/UD-fren.tsv

#python udinflector.py resources/UD-enfr.tsv --model $MODEL --lang English
#python udinflector.py resources/UD-fren.tsv --model $MODEL --lang French
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python conjugator.py --tsv $TSV --model $MODEL --language French


