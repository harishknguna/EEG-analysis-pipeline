#!/bin/bash
#SBATCH --job-name=gogait_source
#SBATCH --partition=normal
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/scripts_to_Fanny
#SBATCH --output=file_output_%j.log
#SBATCH --error=error_output_%j.log

module load MNE/1.3.0
mne
ls -l
which python
python
python < 01b_import_ftdata_and_export_epoch_numpy_array_GOODpatients.py