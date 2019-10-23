#!/bin/bash
#SBATCH --job-name=adni_conv
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12-24:00:00
#SBATCH --output=slurm_%j.out
module load anaconda3
conda activate clinicaEnv
export FREESURFER_HOME=/DIR/TO/freesurfer/6.0.0/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export PATH="DIR/TO/mricron_lx:$PATH"
export PATH="DIR/TO/MRIcroGL:$PATH"
clinica -v convert adni-to-bids './ADNI' 'Dir/To/Downloaded/Data' './ADNI_converted' -m T1

