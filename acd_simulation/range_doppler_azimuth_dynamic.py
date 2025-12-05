import numpy as np
import matplotlib.pyplot as plt

def run_radar_simulation():
    # --- PARAMETERS ---
    noADC = 256
    noRx = 2
    noChirps = 32
    f0 = 60e9               
    fs = 12.5e6       
    c = 299792458.0
    wavelength = c / f0

    # Chirp and radar parameters
    chirpTime = noADC / fs
    BW = 4e9
    slp = BW / chirpTime
    PRI = 300e-6
    antennaSpacing = wavelength / 2.0
    rcs = 1.0
    Pt = 0.0158
    Gt = 10.0
    Gr = 10.0

    # FFT sizes
    FFTRNGSIZE = int(2 ** np.ceil(np.log2(noADC)))
    FFTDOPSIZE = int(2 ** np.ceil(np.log2(noChirps)))
    # AOA_FFT_SIZE = FFTDOPSIZE

    # Target motion
    radarPos = np.array([0.0, 0.0])
    startPos = np.array([2.0, 7.0])
    endPos = np.array([1.0, 1.0])
    duration = 10.0
    target_velocity = (endPos - startPos) / duration
    UPDATE_RATE_SEC = 0.1

    # Axes
    f_bins = (np.arange(FFTRNGSIZE) / FFTRNGSIZE) * fs
    rangeAxis = (c * f_bins) / (2.0 * slp)
    fAx = np.fft.fftshift(np.fft.fftfreq(FFTDOPSIZE, d=PRI))
    velocityAxis = fAx * (wavelength / 2.0)
    # aoaAxis_rad = np.linspace(-np.pi/2, np.pi/2, AOA_FFT_SIZE)
    # aoaAxis_deg = np.rad2deg(aoaAxis_rad)

    # --- PLOTTING SETUP ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))

    # Range profile
    ax1 = fig.add_subplot(2, 2, 1)
    line_range_profile, = ax1.plot(rangeAxis, np.zeros(FFTRNGSIZE), linewidth=2)
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Power (dB)")
    ax1.set_title("Range Profile (Antenna 1)")
    ax1.set_ylim(-100, 0)
    ax1.grid(True)

    # Range-Doppler
    ax2 = fig.add_subplot(2, 2, 2)
    RD_dB_init = np.full((FFTRNGSIZE, FFTDOPSIZE), -100.0)
    im = ax2.imshow(
        RD_dB_init,
        aspect="auto",
        origin="lower",
        extent=[velocityAxis[0], velocityAxis[-1], rangeAxis[0], rangeAxis[-1]],
        cmap="jet",
        vmin=-100.0,
        vmax=0.0,
        # interpolation='nearest',
    )
    
    # TODO: correct radial velocity should be ~0.4 (double check the bisection of fft range)
    
    ax2.set_xlabel("Radial velocity (m/s)")
    ax2.set_ylabel("Range (m)")
    ax2.set_title("Range-Doppler")
    colorBar = fig.colorbar(im, ax=ax2, orientation='vertical')
    colorBar.set_label("Power (dB)")
    # hline_range = ax2.axhline(0.0, color="white", linestyle="--", linewidth=1.5)

    # Top-down geometry
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_aspect("equal")
    ax3.grid(True)
    axis_limit = 6.0
    ax3.set_xlim(min(radarPos[0], startPos[0], endPos[0]) - 1,
                 max(radarPos[0], startPos[0], endPos[0]) + axis_limit)
    ax3.set_ylim(min(radarPos[1], startPos[1], endPos[1]) - 1,
                 max(radarPos[1], startPos[1], endPos[1]) + axis_limit)
    ax3.plot(radarPos[0], radarPos[1], "r*", markersize=12)
    target_point, = ax3.plot(startPos[0], startPos[1], "go", markersize=10)
    aoa_line, = ax3.plot([0, 0], [0, 0], "c--", linewidth=1.5)
    ax3.set_title(f"Top-down view (Time: 0.0s)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    plt.tight_layout()
    
    eps = 1e-12
    t_adc = np.arange(noADC) / fs

    # --- SIMULATION LOOP ---
    for t_now in np.arange(0.0, duration + UPDATE_RATE_SEC, UPDATE_RATE_SEC):
        if t_now > duration:
            break

        # Target position at current frame
        objPos = startPos + target_velocity * t_now
        current_x, current_y = objPos[0], objPos[1]

        # Update target
        target_point.set_data([current_x], [current_y])

        # --- RADAR SIGNAL GENERATION ---
        data = np.zeros((noADC, noRx, noChirps), dtype=np.complex128)
        R_at_start = np.linalg.norm(objPos - radarPos)
        az_at_start = np.arctan2(objPos[1] - radarPos[1], objPos[0] - radarPos[0])
        radial_velocity = np.dot(target_velocity, (objPos - radarPos) / (R_at_start + eps))
        fd_cpi = 2.0 * radial_velocity / wavelength
        print("Doppler: ", fd_cpi)
        
        tau_start = 2.0 * R_at_start / c
        fb_term = slp * tau_start
        Pr_m = (Pt * Gt * Gr * (wavelength ** 2) * rcs) / (((4 * np.pi) ** 3) * (R_at_start ** 4) + eps)
        A_m = np.sqrt(np.abs(Pr_m))
        const_phase_m = np.exp(-1j * 2 * np.pi * (f0 * tau_start + 0.5 * slp * tau_start ** 2))

        for m in range(noChirps):
            doppler_factor = np.exp(1j * 2.0 * np.pi * fd_cpi * m * PRI)
            for rx in range(noRx):
                antenna_phase = np.exp(1j * 2.0 * np.pi * (antennaSpacing * rx * np.sin(az_at_start)) / wavelength)
                beat = A_m * const_phase_m * antenna_phase * doppler_factor * np.exp(1j * 2 * np.pi * fb_term * t_adc)
                data[:, rx, m] = beat

        # Window + Range FFT
        hammingWindow = np.hamming(noADC).reshape(noADC, 1, 1)
        windowed = data * hammingWindow
        data_rx1, data_rx2 = windowed[:, 0, :], windowed[:, 1, :]
        range_fft_rx1 = np.fft.fft(data_rx1, FFTRNGSIZE, axis=0)
        range_fft_rx2 = np.fft.fft(data_rx2, FFTRNGSIZE, axis=0)

        # Remove slow-time (per-range) mean to suppress stationary clutter/DC
        range_fft_rx1 = range_fft_rx1 - np.mean(range_fft_rx1, axis=1, keepdims=True)
        range_fft_rx2 = range_fft_rx2 - np.mean(range_fft_rx2, axis=1, keepdims=True)

        # Apply slow-time window across chirps to reduce spectral leakage
        slow_win = np.hamming(noChirps)
        range_fft_rx1 = range_fft_rx1 * slow_win[np.newaxis, :]
        range_fft_rx2 = range_fft_rx2 * slow_win[np.newaxis, :]

        rd_rx1 = np.fft.fftshift(np.fft.fft(range_fft_rx1, FFTDOPSIZE, axis=1), axes=1)
        rd_rx2 = np.fft.fftshift(np.fft.fft(range_fft_rx2, FFTDOPSIZE, axis=1), axes=1)
        RD_dB = 20.0 * np.log10(np.abs(rd_rx1) + eps)

        # DEBUG: inspect slow-time sequence at the detected bright range bin
        try:
            debug_range_idx = np.argmax(np.max(np.abs(range_fft_rx1), axis=1))
            slow_seq = range_fft_rx1[debug_range_idx, :]
            # print(f"DEBUG: bright range bin index = {debug_range_idx}")
            # print("DEBUG: slow-time magnitude (per chirp):", np.round(np.abs(slow_seq), 6))
            # print("DEBUG: slow-time phase (rad, per chirp):", np.round(np.angle(slow_seq), 6))
        except Exception as e:
            print("DEBUG: failed to extract slow-time sequence:", e)

        # AoA calculation (FIXED)
        range_idx, dop_idx = np.unravel_index(np.argmax(RD_dB), RD_dB.shape)
        S1, S2 = rd_rx1[range_idx, dop_idx], rd_rx2[range_idx, dop_idx]
        delta_phi = np.angle(S2 / (S1 + eps))   # phase difference in radians
        d = antennaSpacing
        # delta_phi -> (2π * d * sin(theta))/λ  => sin(theta) = delta_phi * λ / (2π*d)
        sin_theta = (delta_phi * wavelength) / (2.0 * np.pi * d)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        measured_aoa_rad = np.arcsin(sin_theta)
        measured_aoa_deg = np.rad2deg(measured_aoa_rad)
        print("Measure AoA: ", measured_aoa_deg)
        measured_range = rangeAxis[range_idx]
        measured_velocity = velocityAxis[dop_idx]

        # --- PLOTTING UPDATE ---
        line_range_profile.set_ydata(20.0 * np.log10(np.abs(range_fft_rx1[:, 0]) + eps))
        ax1.relim()
        ax1.autoscale_view(scaley=True)

        im.set_data(RD_dB)
        im.set_clim(vmin=np.max(RD_dB) - 60, vmax=np.max(RD_dB))
        # hline_range.set_ydata([measured_range])

        # Top-down AoA line
        L = R_at_start * 1.5
        true_azimuth_rad = np.arctan2(current_y - radarPos[1], current_x - radarPos[0])
        aoa_line.set_data([radarPos[0], radarPos[0] + L * np.cos(measured_aoa_rad)],
                          [radarPos[1], radarPos[1] + L * np.sin(measured_aoa_rad)])
        ax3.set_title(f"Top-down view (Time: {t_now:.1f}s, True Az: {np.rad2deg(true_azimuth_rad):.2f}°)")

        print(f"\n--- TIME: {t_now:.1f}s ---")
        print(f"Target Pos: ({current_x:.3f}, {current_y:.3f}) m | Range: {measured_range:.3f} m | "
              f"Vel: {measured_velocity:.3f} m/s | Measured AoA: {measured_aoa_deg:.2f}° | True Az: {np.rad2deg(true_azimuth_rad):.2f}°)")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_radar_simulation()
