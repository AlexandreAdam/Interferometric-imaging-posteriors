#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --mem=800G
#SBATCH --constraint=A100
#SBATCH --output=logs/%j.%x.out
#SBATCH --exclude=compute-0-4
#SBATCH --job-name=NoiseSim
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk

# Help description of the shell script
SCRIPT_NAME="$(basename "$0")"
display_help() {
  cat <<EOF
Usage: $SCRIPT_NAME [options] [arguments]

Description: Use a measurement set to generate a number of 
    measurement sets which have false data but the same 
    configuration. Currently fills data with thermal noise 
    estimated from the original data. The data is saved to 
    a 'data' folder in the current directory.

Options:
  -h, --help     Display this help message and exit.
  -n             Number of samples to generate.
  -p             Path to the measurement used as base.
  ...
EOF
}

# Read in command line options
while getopts ":n:p:" opt; do
  case $opt in
  n)
    NO_SAMPLES="$OPTARG"
    ;;
  p)
    PATH_TO_MS="$OPTARG"
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

# Check if N is a valid integer or convert it
if ! [[ $NO_SAMPLES =~ ^-?[0-9]+$ ]]; then
  N=$(expr "$NO_SAMPLES" + 0 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo ">>> Error: -n must be an integer. Setting to 1." >&2
    NO_SAMPLES=1
  fi
fi

# Make necessary output folders
CURRENT_PATH=$(pwd)
mkdir -p data
FOLDER_NAME=$(basename "$PATH_TO_MS")
FOLDER_NAME="${FOLDER_NAME%.*}"
OUT_PATH=$CURRENT_PATH/data/$FOLDER_NAME/
echo ">>> Creating $OUT_PATH folder for output."
mkdir -p $OUT_PATH

echo ">>> Activating environment"
source /home/mbowles/.bashrc # Replace with my own.
source activate conda_casa

echo ">>> Starting noise modelling."
python noise_modelling.py \
  --ms $PATH_TO_MS \
  --outpath $OUT_PATH \
  --number $NO_SAMPLES

echo ">>> finish"
