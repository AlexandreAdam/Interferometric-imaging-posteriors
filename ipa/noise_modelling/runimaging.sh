#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --constraint=A100
#SBATCH --output=logs/%j.%x.out
#SBATCH --exclude=compute-0-4
#SBATCH --job-name=Image
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk

# Read in command line options
while getopts ":p:" opt; do
    case $opt in
    p)
        MS="$OPTARG"
        ;;
    \?)
        echo ">>> Invalid option -$OPTARG" >&2
        exit 1
        ;;
    esac

    case $OPTARG in
    -*)
        echo ">>> Option $opt needs a valid argument"
        exit 1
        ;;
    esac
done

echo ">>> Activating environment."
source /home/mbowles/.bashrc # Replace with my own.
source activate conda_casa

# Get the list of files matching the ".ms" pattern
# file_list=$(ls /share/nas2_5/mbowles/data/alma | grep '_continuum.ms$')

# Quick image a file or a folder of MS files.
echo ">>> Imaging all MS in subdir or given path if already an MS."
python quick_imaging.py --ms $MS

echo ">>> finish"

#### TODO ####
# 1. Chose a single pointing to work with for now. -> HTLup_continuum*
# 2. Make a seperate imaging script. -> done.
# 3. Use the written functions to generate a noise MS and image it. -> Underway.
# 4. Add iteration and run it a number of times to send to Adam.
