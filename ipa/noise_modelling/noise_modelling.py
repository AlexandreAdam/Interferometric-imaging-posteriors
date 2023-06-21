from casatools import table, ms
from casatasks import split
from pathlib import Path
import numpy as np
import argparse
import quick_imaging


def estimate_system_temperature(ms_path):
    # Create a table tool
    tb_tool = table()

    # Open the MS
    tb_tool.open(ms_path)

    # Get the time, antenna1, antenna2, and visibility data
    time = tb_tool.getcol("TIME")
    antenna1 = tb_tool.getcol("ANTENNA1")
    antenna2 = tb_tool.getcol("ANTENNA2")
    data = tb_tool.getcol("DATA")

    # Close the MS
    tb_tool.close()

    # Get the unique baselines
    baselines = set(zip(antenna1, antenna2))

    diff_over_time = []

    # For each baseline, calculate the diff over time
    for baseline in baselines:
        # Find the indices for this baseline
        indices = np.where((antenna1 == baseline[0]) & (antenna2 == baseline[1]))[0]
        # Sort the indices by time
        sorted_indices = indices[np.argsort(time[indices])]

        # Get the data for this baseline, sorted by time
        sorted_data = data[:, :, sorted_indices]

        # Calculate the diff over time
        diff = np.diff(sorted_data).flatten()

        diff_over_time.append(diff)

    return np.concatenate(diff_over_time).std()


def overwrite_visibilities(ms_path, new_ms_path, sigma_n):
    # Create a copy of the original MS
    split(vis=ms_path, outputvis=new_ms_path, datacolumn="data")

    # Create a table and ms tool
    tb_tool = table()
    ms_tool = ms()

    # Open the new MS
    tb_tool.open(new_ms_path, nomodify=False)
    ms_tool.open(new_ms_path)

    # Get the existing spectral windows
    spw_info = ms_tool.getspectralwindowinfo()
    num_spws = len(spw_info)

    # Close the ms tool
    ms_tool.close()

    # Check the shapes of the existing and new visibilities
    for spw in range(num_spws):
        # Select the rows for this spectral window
        tb_spw = tb_tool.query(f"DATA_DESC_ID=={spw}")

        # Get the existing visibilities
        existing_visibilities = tb_spw.getcol("DATA")

        # Generate data
        real = np.random.normal(0, sigma_n, existing_visibilities.shape)
        imaginary = 1j * np.random.normal(0, sigma_n, existing_visibilities.shape)
        new_visibilities = real + imaginary

        if existing_visibilities.shape != new_visibilities.shape:
            raise ValueError(
                f"Shape of new visibilities {new_visibilities.shape} does not match existing visibilities {existing_visibilities.shape}."
            )

        # Overwrite the visibilities
        tb_spw.putcol("DATA", new_visibilities)

        # Close the spectral window table
        tb_spw.close()

    # Close the MS
    tb_tool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read in a measurement set and format ready for OSKAR to handle."
    )
    parser.add_argument(
        "--ms",
        help="Measurement set file to process.",
        default="/share/nas2_5/mbowles/data/alma/HTLup_continuum.ms",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="The name of the file under which the formatted data is saved.",
        required=False,
        default=None,
    )
    parser.add_argument("--cont", action="store_true", help="Continue processing")

    ### Parse arguments
    args = parser.parse_args()
    ms_file = Path(args.ms)

    # Set out index
    if args.number is not None:
        index = int(args.number)
    else:
        index = 0
    if not args.cont:  # Continue with next index
        index = 0
        # Loop through each file in the folder
        for file in ms_file.parent.glob("*.fits"):
            # Extract the number from the filename
            num = re.findall(r"\d+", str(ms_file.parent))
            if num:
                num = int(num[0])
                # If num is greater than or equal to index, update index
                if num >= index:
                    index = num + 1

    for i in range(int(args.number)):
        outname = (
            outpath
            + str(ms_file.stem)
            + "_noise_{i:05}".format(i=i)
            + str(ms_file.suffix)
        )
        ### Add other noise structures here if destired
        # generate_visibilities()
        # add_phase_errors()
        # add_*()

        ### Write out a new visibility set with the generated
        overwrite_visibilities(
    image_name = quick_imaging.quick_clean(vis=str(ms_file), index=index)
    quick_imaging.export_fits(image_name=image_name)
            sigma_n=sigma_n,  # Use when randomly generating vis from sigma_n
        )
