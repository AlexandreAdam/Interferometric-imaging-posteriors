#!/bin/bash

#SBATCH --time=15:00:00
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
    no_samples="$OPTARG"
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

if [ -z "$PATH_TO_MS" ]; then
  echo "No directory path provided. Please run the script with a -p parameter."
  exit 1
fi

# Check if parameter is a directory
if [ -d "$PATH_TO_MS" ]; then
  echo ">>> $PATH_TO_MS is a directory."
else
  echo ">>> $PATH_TO_MS is not a directory."
  exit 1
fi

# Check if N is a valid integer or convert it
if ! [[ $no_samples =~ ^-?[0-9]+$ ]]; then
  N=$(expr "$no_samples" + 0 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo ">>> Error: -n must be an integer. Setting to 1." >&2
    no_samples=1
  fi
fi

# Make necessary output folders
CURRENT_PATH=$(pwd)
mkdir -p data
FOLDER_NAME=$(basename "$PATH_TO_MS")
FOLDER_NAME="${FOLDER_NAME%.*}"
OUT_PATH=$CURRENT_PATH/data/$FOLDER_NAME
echo ">>> Creating ${OUT_PATH} folder for output."
mkdir -p "$OUT_PATH"

echo ">>> Activating environment"
source /home/mbowles/.bashrc # Replace with my own.
source activate conda_casa

cp -r "$PATH_TO_MS" "${OUT_PATH}/" # Copy for imaging.

# Generate PSF image and dirty image of original
python quick_imaging.py \
  --ms "${OUT_PATH}/"

TMP_MS="${OUT_PATH}/$(basename ${FOLDER_NAME})_noise.ms"
echo ">>> Copying input MS to ${TMP_MS} to be overwritten with noise."
cp -r $PATH_TO_MS $TMP_MS

# Start loop
for idx in $(seq 0 1 $no_samples); do
  echo ">>> Starting noise modelling ${idx}"
  # Model noise - generates fits files for each sampled noise MS.
  python noise_modelling.py \
    --ms $TMP_MS \
    --number $idx \
    --cont

  # Remove residual files
  echo ">>> Removing all residual files in ${OUT_PATH}"
  for file in "$OUT_PATH"/*; do
    # If the file does not end with .fits and is not the temporary MS, then remove it
    if [[ $file != *.fits && $(basename "$file") != $(basename "$TMP_MS") ]]; then
      rm -r "$file"
    fi
  done
done

echo ">>> finish"
