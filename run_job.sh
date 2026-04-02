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
#scp -v -3 ud-enfr_{en,fr}.{dic,xml} ujt99zo@jean-zay.idris.fr:/linkhome/rech/genata01/ujt99zo/work/josep/UdInflector/

# Build glossaries with entries : POS \t term_src \t term_tgt
# python SystranUD2glossary.py resources/ud-enfr_fr.dic resources/ud-enfr_en.dic --oname resources/ud --lang1 fr --lang2 en

module purge
#
module load arch/h100 #this module must be loaded before pytorch-gpu                                                                                                                                                                                    
module load pytorch-gpu/py3/2.6.0

MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-8B #Qwen3-32B Qwen3-8B
NAME=Qwen3-8B

#python inflect.py resources/ud-fren.tsv --model $MODEL --language French --out resources/ud-fren.tsv.$NAME.inflections.tsv
#python inflect.py resources/ud-enfr.tsv --model $MODEL --language English --out resources/ud-enfr.tsv.$NAME.inflections.tsv

# Evaluation
# 
python eval_inflector.py resources/ud-enfr_en.xml resources/ud-enfr.tsv.$NAME.inflections.tsv --verbose > resources/ud-enfr.tsv.$NAME.inflections.tsv.eval
# 
python eval_inflector.py resources/ud-enfr_fr.xml resources/ud-enfr.tsv.$NAME.inflections.tsv --verbose > resources/ud-fren.tsv.$NAME.inflections.tsv.eval
