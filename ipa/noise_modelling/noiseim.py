import numpy as np
import oskar
import sys


def main():
    imager = oskar.Imager()
    imager.fov_deg = 1.75
    imager.image_size = 200
    imager.algorithm = "W-projection"
    imager.channel_snapshots = True
    imager.fft_on_gpu = True
    imager.generate_w_kernels_on_gpu = True
    imager.grid_on_gpu = True
    imager.image_type = "I"
    imager.input_file = ["data/noise_vis.ms"]
    imager.root_path = "data/noise_" + sys.argv[1]
    imager.run()

    imager.image_type = "PSF"
    imager.run()
    return


if __name__ == "__main__":
    main()
