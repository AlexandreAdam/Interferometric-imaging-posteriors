#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --constraint=A100
#SBATCH --output=logs/%j.%x.out
#SBATCH --exclude=compute-0-4
#SBATCH --job-name=NoiseSim
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk

# Parameterise the docker container.
# Use the gain cal, phase, [...], error generation built into oskar.
echo "hello world"
echo getopts

while getopts ":n:p:" opt; do
  case $opt in
    n) NO_SAMPLES="$OPTARG"
    ;;
    p) PATH_TO_MS="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

# Check if N is a valid integer or convert it
if ! [[ $NO_SAMPLES =~ ^-?[0-9]+$ ]]; then
  N=$(expr "$NO_SAMPLES" + 0 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo "Error: -n must be an integer. Setting to 1." >&2
    NO_SAMPLES=1
  fi
fi

runoskar2() {
    singularity exec --bind /share/nas2_5/ --nv /share/nas2_5/mbowles/data/OSKAR-2.8.2-Python3.sif $1 $2
}

runoskarpy2() {
    singularity exec --bind /share/nas2_5/ --nv /share/nas2_5/mbowles/data/OSKAR-2.8.2-Python3.sif python3 $1 $2
}

mkdir -p data

ls /share/nas2_5/mbowles/data/

echo ">>> start"
source /home/mbowles/.bashrc # Replace with my own.
source activate conda_casa

# Pre process data
directory=$(dirname "$PATH_TO_MS")
filename=$(basename "$PATH_TO_MS")
extension="${filename##*.}"
filename="${filename%.*}"
new_filename="${filename}_processed.${extension}" # File name change. data.ms -> data_processed.ms
PROCESSED_DATA="${directory}/${new_filename}"
python pre_process.py $PATH_TO_MS -o $PROCESSED_DATA

echo ">>> run oskar"
for ((i=0; i<$NO_SAMPLES; i++)); do
  echo "Starting the $i'th run"
  # python3 gennoise.py # Generate a random noise file "../visdata/noise_temp"
  runoskarpy2 writevis.py $PROCESSED_DATA # Write a visibility set based on a measurement set and the generated noise_temp file
  runoskarpy2 noiseim.py $i # Image the generated visibility.
done

echo ">>> finish"
