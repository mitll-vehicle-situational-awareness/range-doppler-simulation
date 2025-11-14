import numpy as np
import matplotlib.pyplot as plt

# FMCW Radar Parameters
c = 3e8  # Speed of light (m/s)
f_start = 77e9  # Start frequency (77 GHz)
B = 150e6  # Bandwidth (150 MHz) - reduced for longer range
T_chirp = 40e-6  # Chirp duration (40 μs)
N_samples = 512  # Samples per chirp - increased for better range resolution
N_chirps = 128  # Number of chirps (for angle estimation)

# Derived parameters
fs = N_samples / T_chirp  # Sampling frequency
slope = B / T_chirp  # Chirp slope
range_resolution = c / (2 * B)
max_range = (c * fs) / (2 * slope)

print(f"Range Resolution: {range_resolution:.2f} m")
print(f"Max Range: {max_range:.2f} m")

# Define targets: [range (m), azimuth angle (degrees), RCS (radar cross section)]
targets = [
    [20, -30, 1.0],   # Target 1: 20m away, -30°
    [35, 0, 1.5],     # Target 2: 35m away, 0° (directly ahead)
    [50, 20, 0.8],    # Target 3: 50m away, +20°
    [25, 15, 0.6],    # Target 4: 25m away, +15°
]

# Antenna parameters
wavelength = c / f_start
d = wavelength / 2  # Element spacing (half wavelength)
N_antennas = 2  # Number of receive antennas

# Time arrays
t_fast = np.linspace(0, T_chirp, N_samples)  # Fast time (within chirp)
t_slow = np.arange(N_chirps) * T_chirp  # Slow time (across chirps)

# Initialize radar data cube: [chirps, antennas, samples]
radar_data = np.zeros((N_chirps, N_antennas, N_samples), dtype=complex)

# Generate signal for each target
for target_range, target_angle, rcs in targets:
    # Range delay
    tau = 2 * target_range / c
    
    # Beat frequency (frequency after mixing)
    f_beat = slope * tau
    
    # Doppler frequency (assuming stationary targets for simplicity)
    f_doppler = 0
    
    # Generate signal for each chirp and antenna
    for chirp_idx in range(N_chirps):
        for ant_idx in range(N_antennas):
            # Phase shift due to angle of arrival
            phase_shift = 2 * np.pi * (ant_idx * d / wavelength) * np.sin(np.radians(target_angle))
            
            # Signal amplitude (with some noise variation)
            amplitude = np.sqrt(rcs)
            
            # Generate IF signal
            signal = amplitude * np.exp(1j * (
                2 * np.pi * f_beat * t_fast +
                2 * np.pi * f_doppler * t_slow[chirp_idx] +
                phase_shift
            ))
            
            radar_data[chirp_idx, ant_idx, :] += signal

# Add noise
noise_power = 0.1
noise = np.sqrt(noise_power/2) * (np.random.randn(*radar_data.shape) + 
                                   1j * np.random.randn(*radar_data.shape))
radar_data += noise

# Processing: Range FFT
range_fft = np.fft.fft(radar_data, axis=2)
range_fft = np.fft.fftshift(range_fft, axes=2)

# Average across chirps for single frame
range_profile = np.mean(np.abs(range_fft[:, 0, :]), axis=0)

# Range axis
range_bins = np.linspace(-max_range/2, max_range/2, N_samples)

# Processing: Azimuth FFT (beamforming)
azimuth_fft = np.fft.fft(range_fft, n=256, axis=1)
azimuth_fft = np.fft.fftshift(azimuth_fft, axes=1)

# Average across chirps
range_azimuth_map = np.mean(np.abs(azimuth_fft), axis=0).T

# Azimuth axis (angle in degrees)
angle_bins = np.linspace(-90, 90, azimuth_fft.shape[1])

# Focus on positive range only
positive_range_idx = range_bins >= 0
range_bins_pos = range_bins[positive_range_idx]
range_azimuth_map_pos = range_azimuth_map[positive_range_idx, :]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Range Profile
axes[0].plot(range_bins_pos, 20*np.log10(range_profile[positive_range_idx] + 1e-10))
axes[0].set_xlabel('Range (m)', fontsize=12)
axes[0].set_ylabel('Magnitude (dB)', fontsize=12)
axes[0].set_title('Range Profile (Single Antenna)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 60])

# Add vertical lines at target ranges
for target_range, _, _ in targets:
    axes[0].axvline(x=target_range, color='r', linestyle='--', alpha=0.5, linewidth=1)

# Plot 2: Range-Azimuth Map
range_azimuth_db = 20*np.log10(range_azimuth_map_pos + 1e-10)
im = axes[1].imshow(range_azimuth_db, 
                     aspect='auto',
                     extent=[angle_bins[0], angle_bins[-1], 
                            range_bins_pos[-1], range_bins_pos[0]],
                     cmap='jet',
                     vmin=np.max(range_azimuth_db) - 40,
                     vmax=np.max(range_azimuth_db))

axes[1].set_xlabel('Azimuth Angle (degrees)', fontsize=12)
axes[1].set_ylabel('Range (m)', fontsize=12)
axes[1].set_title('Range-Azimuth Map (Heatmap)', fontsize=14, fontweight='bold')
axes[1].set_ylim([60, 0])
axes[1].set_xlim([-60, 60])

# Add colorbar
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label('Magnitude (dB)', fontsize=11)

# Add markers for true target positions
for target_range, target_angle, _ in targets:
    axes[1].plot(target_angle, target_range, 'w*', markersize=15, 
                markeredgecolor='black', markeredgewidth=1.5)

plt.tight_layout()
plt.savefig('fmcw_radar_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Target Information ===")
for i, (r, a, rcs) in enumerate(targets, 1):
    print(f"Target {i}: Range = {r} m, Azimuth = {a}°, RCS = {rcs}")
print("\nWhite stars (*) on the range-azimuth map indicate true target positions.")