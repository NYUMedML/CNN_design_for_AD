#!/bin/bash
#SBATCH --job-name=adni_preprocess
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12-24:00:00
#SBATCH --output=slurm_%j.out

module load anaconda3

module load matlab
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export MATLAB_HOME=HOME/TO/MATLAB
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab
conda activate clinicaEnv
export FREESURFER_HOME=/gpfs/share/apps/freesurfer/6.0.0/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SPM_HOME= HOME/TO/SPM
clinica run t1-volume './ADNI_converted' './ADNI_processed' 'TRAIN' -tsv './Train_ADNI.tsv' -wd './WD_train' -np 24
