import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time


def single_baseline(antenna1, antenna2, frequency=1.5, HA=0, uv=True, d_deg=45):
    """
    Computes the baseline coordinates in the u-v plane for interferometric observations.

    Args:
        antenna1 (array-like): Position vector of the first antenna.
        antenna2 (array-like): Position vector of the second antenna.
        frequency (float, optional): Observing frequency in GHz. Default is 1.5 GHz.
        HA (float, optional): Hour Angle of the observation in hours. Default is 0.
        uv (bool, optional): Flag indicating whether to compute the u-v coordinates. Default is True.
        d_deg (float, optional): Angular separation of the antennas in degrees. Default is 45 degrees.

    Returns:
        tuple: A tuple containing the u and v coordinates of the baseline in units of wavelength.

    """
    c = 299792458  # Speed of light in m/s
    frequency = frequency * 10**9  # Convert frequency to Hz
    baseline = antenna1 - antenna2  # Compute baseline vector

    if uv:
        H_rad = 2 * np.pi * HA / 24  # Convert Hour Angle to radians
        d = 2 * np.pi * d_deg / 360  # Convert angular separation to radians

        # Compute u and v coordinates using the transformation matrix
        baseline_u = (
            (np.sin(H_rad) * baseline[0] + np.cos(H_rad) * baseline[1]) * frequency / c
        )
        baseline_v = (
            (
                -np.sin(d) * np.cos(H_rad) * baseline[0]
                + np.sin(d) * np.sin(H_rad) * baseline[1]
                + np.cos(d) * baseline[2]
            )
            * frequency
            / c
        )
    else:
        # If uv is False, use the original baseline coordinates as u and v
        baseline_u, baseline_v = baseline[0], baseline[1]

    return baseline_u, baseline_v  # Return baseline coordinates in units of wavelength


def compute_clean_response(
    antenna1,
    antenna2,
    sources,
    fluxes,
    obstime,
    freq_samples=64,
    freq_min=1.0,  # Units: [GHz]
    freq_max=2.0,  # Units: [GHz]
    H_samples=60 * 3 * 2,
    H_min=-1.5,  # [hrs]
    H_max=1.5,  # [hrs]
    uv_coverage=False,
):
    """
    Computes the clean response to the sky for a given pair of antennas, frequency and hour angle ranges,
    considering multiple sources with individual fluxes.

    Args:
        antenna1 (tuple, (x,y,z)): Index of the first antenna.
        antenna2 (tuple, (x,y,z)): Index of the second antenna.
        sources (array-like): Array of source coordinates in the form [(ra1, dec1), (ra2, dec2), ...].
        fluxes (array-like): Array of flux values corresponding to each source.
        obstime (str): Observation time in ISO format, e.g., '2023-06-06T00:00:00'.
        freq_samples (int, optional): Number of frequency samples. Default is 64.
        freq_min (float, optional): Minimum frequency in GHz. Default is 1.0 GHz.
        freq_max (float, optional): Maximum frequency in GHz. Default is 2.0 GHz.
        H_samples (int, optional): Number of hour angle samples. Default is 360.
        H_min (float, optional): Minimum hour angle in hours. Default is -1.5 hrs.
        H_max (float, optional): Maximum hour angle in hours. Default is 1.5 hrs.

    Returns:
        array-like: The clean response to the sky.

    """
    # antenna1 = antennae[antenna1No - 1]  # Get the position of the first antenna
    # antenna2 = antennae[antenna2No - 1]  # Get the position of the second antenna

    source_coords = []
    for source in sources:
        source_coords.append(SkyCoord(source[0] * u.deg, source[1] * u.deg))

    fluxes = np.array(fluxes)  # Convert fluxes to numpy array

    # Define Earth location
    location = EarthLocation.from_geocentric(*antenna1, unit=u.m)

    # Create a meshgrid of frequency and hour angle values
    # freq = np.linspace(freq_min, freq_max, freq_samples) * u.GHz
    # H = np.linspace(H_min, H_max, H_samples) * u.hourangle
    # H = H.to(u.deg)
    meshgrid = np.mgrid[
        freq_min : freq_max : freq_samples * 1j, H_min : H_max : H_samples * 1j
    ]
    freq = meshgrid[0]
    H = meshgrid[1]

    # Define observation time
    obstime = Time(obstime)

    # Compute u-v baseline coordinates
    baseline_u, baseline_v = single_baseline(antenna1, antenna2, freq, H)

    # Compute the clean response for each source
    responses = []
    for i, source_coord in enumerate(source_coords):
        response = fluxes[i] * np.exp(
            -2.0j
            * np.pi
            * (baseline_u * source_coord.ra.value + baseline_v * source_coord.dec.value)
        )
        responses.append(response)

    # Combine the responses from individual sources to get the response to the sky
    sky_response = np.sum(responses, axis=0)

    if uv_coverage:
        return np.stack([baseline_u, baseline_v, sky_response], axis=0)
    return sky_response

def add_thermal_noise():
    raise NotImplementedError

def baseline_response():
    raise NotImplementedError

def compute_real_response():
    raise NotImplementedError
