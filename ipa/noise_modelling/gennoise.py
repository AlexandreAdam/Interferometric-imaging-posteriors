import numpy as np
from astropy import units, constants
import torch


def main():
    tsysoverae = 1 / 1.235 * units.K / units.m**2
    # cen_ra = 0.0  # deg
    # cen_dec = -27.0  # deg
    num_ch = 66
    f_21 = 1420405751.7667  # in Hz
    # zarr = np.load("zarr.npy")
    # freqarr = f_21 / (1 + zarr)
    # cb = np.diff(freqarr).mean() * units.Hz  # Channel width I think
    channel_bandwidth = 8 * units.Hz
    t_inter = 60 * units.s
    sigma_n = (
        (2 * constants.k_B * tsysoverae / np.sqrt(channel_bandwidth * t_inter))
        .to("Jy")
        .value
    )  # sensitivity estimate

    num_samples = 10478400
    # observation_length = 8 * units.hours
    # num_samples = observation_length / t_inter * baseline_no # Not sure this is actually what it is meant to be
    pol_channels = 1

    vis = (
        torch.normal(0, sigma_n / np.sqrt(2), (num_ch, num_samples, pol_channels)).to(
            torch.complex64
        )
        + torch.normal(0, sigma_n / np.sqrt(2), (num_ch, num_samples, pol_channels)).to(
            torch.complex64
        )
        * 1j
    )
    vis = vis.cpu().numpy()

    np.save("./tmp_noise_vis", vis)


if __name__ == "__main__":
    main()
