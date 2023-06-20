import numpy as np
import argparse
import oskar
import os
from casatools import ms as MS


def estimate_system_temperature(vis_data, ms):
    print("VISIBILITY SHAPE: {vis_data.shape}")
    vis = vis_data.reshape((ms.num_channels, ms.num_baselines, ms.num_pols, -1))
    print("VISIBILITY SHAPE: {vis.shape}")
    T_sys = np.diff(vis, axis=-1).std()
    print("T_sys: {T_sys}")
    print("T_sys: {T_sys.shape}")
    return T_sys.mean()


def random_vis(vis_data, ms):
    T_sys = estimate_system_temperature(vis_data, ms)
    sigma_n = T_sys / (2**0.5)  # sensitivity estimate
    vis = torch.normal(0, sigma_n / np.sqrt(2), vis_data.shape).to(
        torch.complex64
    ) + 1j * torch.normal(0, sigma_n / np.sqrt(2), vis_data.shape).to(torch.complex64)
    return vis.cpu().numpy()


def main(ms_file):
    ms = oskar.MeasurementSet.open(ms_file, readonly=True)
    uu, vv, ww = np.array(ms.read_coords(0, ms.num_rows))
    print(f"uu.shape: {uu.shape}")
    filename = "data/noise_vis.ms"  # output name
    filename = ensure_single_spectralwindow(filename)

    # Initialise output measurement set
    msout = oskar.MeasurementSet.create(
        filename,
        ms.num_stations,
        ms.num_channels,
        ms.num_pols,
        ms.freq_start_hz,
        ms.freq_inc_hz,
    )
    msout.set_phase_centre(ms.phase_centre_ra_rad, ms.phase_centre_dec_rad)

    msout.write_coords(  # Not sure why I do this
        0,  # starting row
        ms.num_rows,  # number of rows
        uu,
        vv,
        ww,
        1,  # exposure seconds
        1,  # interval seconds
        1,  # time stamp
    )

    # Conditional check if 4 stokes
    # use stokes V to estimate system temperature
    # 1-3 = V
    # nrows = npolarisation x n_channels x ntime_steps x n_baselines
    # std(np.diff(n_channels axis of vis)) = sqrt(2)*std(thermal_noise)
    # works for ntime_steps as well. Use whichever has a better resolution.

    vis_data = ms.read_vis(
        start_row=0,
        start_channel=0,
        num_channels=ms.num_channels,
        num_baselines=ms.num_baselines,
    )
    visdata = random_vis(vis_data, ms)

    msout.write_vis(
        0,  # start row
        0,  # start chan
        ms.num_channels,
        ms.num_rows,
        visdata,  # noise data from estimated system temperature
    )


if __name__ == "__main__":
    print(os.listdir())
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Emulate a measurement set with the observations temperature noise."
    )
    parser.add_argument(
        "ms_file",
        help="Measurement set file to process.",
        default="/share/nas2_5/mbowles/data/alma/HTLup_continuum.ms",
    )

    # Parse arguments
    args = parser.parse_args()

    main(args.ms_file)
