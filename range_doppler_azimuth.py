import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
frameRate = 155.0       # CPI rate (Hz) (not used directly for PRF here)
noFrames = 1
noADC = 256
noRx = 2
noChirps = 32
f0 = 60e9               # carrier (Hz)
# slp = 144.858e12        # chirp slope k (Hz/s)
fs = 12.5e6             # ADC sampling rate (Hz) (samples per chirp)
c = 299792458.0
wavelength = c / f0

# PRI and PRF (set PRI such that PRF < 1/Tchirp; choose PRI >= chirp duration)
chirpTime = noADC / fs    # chirp duration (s)
BW = 4e9                # bandwidth (Hz)
slp = BW / chirpTime    # chirp slope k (Hz/s)
PRI = 300e-6            # Pulse repetition interval (s) - pick > chirpTime
PRF = 1.0 / PRI
# bw = 4e4
# get a range profile --> c / 2 bw
# generate the object radially
# radial = two of them (sin)
# cross-range (does not produce doppler effect)

# shift in freqency is proportional to ratio (velocity / (c / 2) * center frequency) = 

# Derived FFT sizes
FFTRNGSIZE = int(2**np.ceil(np.log2(noADC)))
FFTDOPSIZE = int(2**np.ceil(np.log2(noChirps)))

# ---------- TARGET & RADAR ----------
radarPos = np.array([0.0, 0.0])
objPos = np.array([2.0, 5.0])
rcs = 1.0
target_velocity = 0.0  # stationary

R = np.linalg.norm(objPos - radarPos)
az = np.arctan2(objPos[1], objPos[0])

# Tx/Rx power/gains
Pt = 0.0158 # in watts (12dBm --> 15.8 mW)
Gt = 10.0
Gr = 10.0

# 4ghz = bw

# ---------- AXES ----------
# Range axis from beat frequency bin mapping:
# beat freq per range bin = (bin / N) * fs  -> R = c * f_b / (2k)
f_bins = (np.arange(FFTRNGSIZE) / FFTRNGSIZE) * fs
rAx = (c * f_bins) / (2.0 * slp)

# Proper slow-time frequency axis (centered at zero after fftshift)
fAx = np.fft.fftshift(np.fft.fftfreq(FFTDOPSIZE, d=PRI))   # doppler shift in Hz, ranges [-PRF/2, +PRF/2)
vAx = fAx * (wavelength / 2.0) # v = f * lambda / 2

print(f"Range axis: {rAx[0]:.3f} to {rAx[-1]:.3f} m")
print(f"Velocity axis: {vAx[0]:.6f} to {vAx[-1]:.6f} m/s")

# ---------- DATA CUBE ----------
data = np.zeros((noADC, noRx, noChirps, noFrames), dtype=np.complex128)
t_adc = np.arange(noADC) / fs  # time within chirp

# round-trip time
tau = 2.0 * R / c

# received power (monostatic radar equation)
Pr = (Pt * Gt * Gr * (wavelength**2) * rcs) / (((4*np.pi)**3) * (R**4) + 1e-30)
A = np.sqrt(np.abs(Pr))  # amplitude proportional to sqrt(P_r)

# antenna spacing and phase (linear array along x-axis)
antennaSpacing = wavelength / 2.0

# doppler frequency (Hz) from radial velocity
fd = 2.0 * target_velocity / wavelength  # Hz

# analytic beat frequency
fb = slp * tau

# sanity check: ensure beat frequency under Nyquist to avoid aliasing
if fb >= fs/2:
    print(f"WARNING: beat frequency fb={fb:.3e} Hz >= fs/2={fs/2:.3e} Hz. Increase fs or reduce slope/bandwidth.")
else:
    print(f"fb = {fb:.3e} Hz (fs/2 = {fs/2:.3e} Hz)")

# constant phase term from mixing (scalar)
const_phase = np.exp(-1j * 2*np.pi * (f0*tau + 0.5 * slp * tau**2))

for m in range(noChirps):
    # doppler phase across chirps (slow time)
    doppler_factor = np.exp(1j * 2.0 * np.pi * fd * m * PRI)
    for rx in range(noRx):
        # antenna phase for this RX element
        antenna_phase = np.exp(1j * 2.0 * np.pi * (antennaSpacing * rx * np.sin(az)) / wavelength)
        # beat term: single-tone at fb (linear-in-time)
        beat = A * const_phase * antenna_phase * doppler_factor * np.exp(1j * 2*np.pi * (slp * tau) * t_adc)
        # If tau > chirpTime, move this energy into a later chirp (wrap). For this example tau << chirpTime.
        data[:, rx, m, 0] = beat

# ---------- WINDOW + RANGE FFT ----------
hammingWindow = np.hamming(noADC).reshape(noADC, 1, 1)  # shape (noADC,1,1) to broadcast across Rx & chirps
windowed = data[:, :, :, 0] * hammingWindow  # (noADC, noRx, noChirps)

# pick first RX to visualize range profile
rx0 = 0
range_profiles = windowed[:, rx0, :]  # shape (noADC, noChirps)

# range FFT (along ADC/sample axis)
range_fft = np.fft.fft(range_profiles, FFTRNGSIZE, axis=0)  # (FFTRNGSIZE, noChirps)

# Doppler FFT along chirp axis (axis=1) and fftshift
rd = np.fft.fftshift(np.fft.fft(range_fft, FFTDOPSIZE, axis=1), axes=1)
RD_dB = 20.0 * np.log10(np.abs(rd) + 1e-12)

# ---------- PLOTTING ----------
fig = plt.figure(figsize=(16,10))

# 1) Range profile (first chirp)
ax1 = fig.add_subplot(2,2,1)
range_profile_db = 20.0 * np.log10(np.abs(range_profiles[:,0]) + 1e-12)
ax1.plot(rAx, range_profile_db, linewidth=2)
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Power (dB)")
ax1.set_title("Range Profile (first chirp)")
ax1.grid(True)

# 2) Range-Doppler heatmap
ax2 = fig.add_subplot(2,2,2)
# rd shape = (range_bins, dop_bins)
im = ax2.imshow(RD_dB, aspect='auto', origin='lower',
               extent=[vAx[0], vAx[-1], rAx[0], rAx[-1]],
               cmap='jet', vmin=np.max(RD_dB)-60)
ax2.set_xlabel("Radial velocity (m/s)")
ax2.set_ylabel("Range (m)")
ax2.set_title("Range-Doppler")
plt.colorbar(im, ax=ax2, label='Magnitude (dB)')
ax2.set_ylim(0, rAx[-1])

# 3) Top-down geometry and wavefronts (correct radii)
ax3 = fig.add_subplot(2,2,3)
ax3.set_aspect('equal')
ax3.set_xlim(-1,8) 
ax3.set_ylim(-1,8)
ax3.plot(radarPos[0], radarPos[1], 'r*', markersize=12, label='Radar')
ax3.plot(objPos[0], objPos[1], 'go', markersize=10, label='Target')
ax3.set_title("Top-down view")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")

# times to visualize (0 .. some multiple of 2R/c)
t_vals = np.linspace(0.0, 2.5 * (2*R/c), 6)  # go beyond two-way time to see returns
colors = plt.cm.viridis(np.linspace(0.2,0.9,len(t_vals)))

for t, col in zip(t_vals, colors):
    # outgoing from radar radius:
    r_out = c * t  # one-way radius
    circ_out = plt.Circle(radarPos, r_out, fill=False, linestyle='--', alpha=0.6)
    ax3.add_patch(circ_out)
    # echo visible only after outgoing hits target at t_hit = R/c
    if t >= R/c:
        # echo originates at the target at time t_hit and expands outward from target:
        # radius of echo wave centered at target is r_echo = c * (t - R/c)
        r_echo = c * (t - R/c)
        circ_echo = plt.Circle(objPos, r_echo, fill=False, linestyle=':', alpha=0.6)
        ax3.add_patch(circ_echo)

ax3.legend(loc='upper right')

# 4) Instantaneous beat spectrum (FFT of one beat)
ax4 = fig.add_subplot(2,2,4)
# take beat from first chirp, first rx
beat_sig = range_profiles[:,0]
# compute spectrum
spec = np.fft.fftshift(np.fft.fft(beat_sig, 2048))
freqs = np.fft.fftshift(np.fft.fftfreq(len(spec), 1.0/fs))
ax4.plot(freqs, 20*np.log10(np.abs(spec)+1e-12))
ax4.set_xlim(0, fs/2)
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Magnitude (dB)")
ax4.set_title("Beat Spectrum (first chirp)")
ax4.grid(True)

# DEBUGGING: find max in RD (range x doppler)
range_idx, dop_idx = np.unravel_index(np.argmax(RD_dB), RD_dB.shape)
measured_range = rAx[range_idx]
measured_velocity = vAx[dop_idx]

print(f"Detected peak: range_idx={range_idx}, measured_range={measured_range:.6f} m")
print(f"Detected peak doppler index={dop_idx}, measured_velocity={measured_velocity:.6f} m/s")
ax2.axhline(measured_range, color='white', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.show()