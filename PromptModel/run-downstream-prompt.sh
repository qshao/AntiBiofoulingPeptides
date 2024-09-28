#!/bin/bash

# V.Gazula 1/8/2019
 
#SBATCH -t 72:00:00                             #Time for the job to run
#SBATCH --job-name=prompt                  #Name of the job
#SBATCH -N 1                                    #Number of nodes required
#SBATCH -n 1
#SBATCH -c 10                            #Number of cores needed for the job
#SBATCH --partition=V4V32_CAS40M192_L                           #Name of the queue
#SBATCH --gres=gpu:1                    #Number of GPU's
#SBATCH --mail-type ALL                         #Send email on start/end
#SBATCH --account=gol_qsh226_uksr               #Name of account to run under
##SBATCH --nodelist=gvnodeb004


module load ccs/anaconda/3
module load ccs/cuda/11.6.0_510.39.01
source activate /project/qsh226_uksr/joint_training

#export CUDA_VISIBLE_DEVICES=1
#export HF_DATASETS_CACHE="/project/qsh226_uksr/huggingface"
export TRANSFORMERS_CACHE=/project/qsh226_uksr/huggingface


export TORCH_HOME=/project/qsh226_uksr/torch_cache/
export HF_HOME=/project/qsh226_uksr/transformers_cache/

for i in {1..10}
#for i in 10
do
{
accelerate launch train_nf.py --config_path nf_config_prompt-esm-$i.yaml
}
done   



