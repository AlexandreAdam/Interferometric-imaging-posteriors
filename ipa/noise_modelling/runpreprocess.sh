#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --constraint=A100
#SBATCH --output=logs/%j.%x.out
#SBATCH --exclude=compute-0-4
#SBATCH --job-name=PreProcess
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk

# Parameterise the docker container.
# Use the gain cal, phase, [...], error generation built into oskar.

echo ">>> start"
source /home/mbowles/.bashrc # Replace with my own.
source activate conda_casa

# Get the list of files matching the ".ms" pattern
file_list=$(ls /share/nas2_5/mbowles/data/alma | grep '_continuum.ms$')

# Iterate over each file in the list
for PATH_TO_MS_ in $file_list; do    # Run the Python script with the current entry
    # Pre process data
    PATH_TO_MS="/share/nas2_5/mbowles/data/alma/$PATH_TO_MS_"
    directory=$(dirname "$PATH_TO_MS")
    filename=$(basename "$PATH_TO_MS")
    extension="${filename##*.}"
    filename="${filename%.*}"
    new_filename="${filename}_processed.${extension}" # File name change. data.ms -> data_processed.ms
    PROCESSED_DATA="${directory}/${new_filename}"
    echo $PATH_TO_MS
    echo $PROCESSED_DATA
    python pre_process.py --ms $PATH_TO_MS --outname $PROCESSED_DATA
done

echo ">>> finish"


#### TODO ####
# 1. Chose a single pointing to work with for now.
# 2. Make a seperate imaging script.
# 3. Use the written functions to generate a noise MS and image it.
# 4. Add iteration and run it a number of times to send to Adam.
