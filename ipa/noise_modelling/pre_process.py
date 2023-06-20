from casatools import table, ms
from casatasks import mstransform
from pathlib import Path
import argparse
import os

from casatools import table
import numpy as np

from casatools import table
import numpy as np


def find_largest_spw(ms_path):
    # Create a table tool
    tb_tool = table()

    # Open the MAIN table
    tb_tool.open(ms_path)

    # Get the rows for all spectral windows
    spw_rows = tb_tool.getcol("DATA_DESC_ID")

    # Count the number of rows for each spectral window
    spw_counts = {spw: (spw_rows == spw).sum() for spw in set(spw_rows)}

    # Find the spectral window with the most rows
    largest_spw = max(spw_counts, key=spw_counts.get)

    # Close the table tool
    tb_tool.close()

    return largest_spw


def save_largest_spw(ms_path, new_ms_path):
    largest_spw = find_largest_spw(ms_path)

    if largest_spw is not None:
        # Import the task
        from casatasks import split

        # Use the split task to create a new MS with just the largest spectral window
        split(
            vis=ms_path, outputvis=new_ms_path, spw=str(largest_spw), datacolumn="data"
        )


def ensure_single_spectralwindow(ms_path, outname=None):
    """
    Ensure the input measurement set (ms) only has one spectral window.

    Parameters:
    ms_path (str): Path to the measurement set.
    outname (str): The desired name for the output file, if transformation is required.
    """

    # Instantiate the table tool
    tb = table()

    # Open the SPECTRAL_WINDOW sub-table
    tb.open(ms_path)
    tb.clearlocks()
    tb.close()

    tb.open(os.path.join(ms_path, "SPECTRAL_WINDOW"))

    # Check if there is only one spectral window
    if tb.nrows() > 1:
        # Close the table
        tb.close()

        if outname is None:
            raise ValueError(
                f">>> The measurement set at {ms_path} has more than one spectral window and no output name was specified."
            )
        else:
            print(f"{ms_path} has more than one spectral window.")
            print(f">>> Transformation will be performed to create {outname}")

            # Perform the transformation to a single spectral window
            mstransform(
                vis=ms_path,
                outputvis=outname,
                combinespws=True,
                datacolumn="all",
                chanaverage=True,
                chanbin=16,
            )

            return outname
    else:
        # Close the table
        tb.close()
        print(f">>>>> HERE! THIS ONE ACTUALLY WORKED MAYBE!")

        return ms_path


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
        "-o",
        "--outname",
        help="The name of the file under which the formatted data is saved.",
        required=False,
        default=None,
    )

    # Parse arguments
    args = parser.parse_args()
    ms_file = Path(args.ms)
    if args.outname is None:
        outname = ms_file.root + ms_file.stem + "_processed" + ms_file.suffix
    else:
        outname = args.outname
    if Path(outname).is_dir == False:
        save_largest_spw(str(ms_file), str(outname))
