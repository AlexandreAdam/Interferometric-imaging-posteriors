from casatasks import tclean, exportfits
from pathlib import Path
import argparse
import shutil


def quick_clean(vis, overwrite=True, index=None):
    vis = Path(vis)
    image_name = str(vis.parent) + "/" + str(vis.stem)

    if index is not None:
        image_name += f"_{index:06d}"

    if overwrite:
        # List of possible tclean output extensions
        extensions = [".image", ".mask", ".model", ".pb", ".psf", ".residual", ".sumwt"]

        # Remove any existing tclean output files
        for ext in extensions:
            path = Path(image_name + ext)
            if path.exists():
                print(path)
                shutil.rmtree(path)

    # Create dirty image
    tclean(
        str(vis),
        imagename=image_name,
        datacolumn="data",
        imsize=(256, 256),
        cell=["0.01arcsec", "0.01arcsec"],
        niter=0,  # creates dirty image
        calcpsf=True,
    )
    return image_name


def export_fits(image_name, psf=False, overwrite=True):
    if psf:
        path = Path(image_name)
        directory = str(path.parent)
        exportfits(
            imagename=f"{image_name}.psf",
            fitsimage=f"{directory}/psf.fits",
            overwrite=overwrite,
        )
    else:
        exportfits(
            imagename=f"{image_name}.image",
            fitsimage=f"{image_name}.fits",
            overwrite=overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image a set of measurement sets.")
    parser.add_argument(
        "--ms",
        help="Measurement set file or folder of files to image.",
    )

    # Parse arguments
    args = parser.parse_args()
    path = Path(args.ms)
    if path.exists() and path.is_dir() and (str(path) != "."):
        if path.name.endswith(".ms"):
            # List files matching the pattern ".ms"
            files = [str(path)]
        else:
            files = [str(file) for file in path.glob("*.ms") if file.is_dir()]
        for vis in files:
            print(f">>> Starting imaging for vis {vis}")
            image_name = quick_clean(vis=vis)
            export_fits(image_name=image_name)
        export_fits(image_name=image_name, psf=True)

    else:
        raise ValueError(f"Path is not a valid directory (or measurement set): {path}")
