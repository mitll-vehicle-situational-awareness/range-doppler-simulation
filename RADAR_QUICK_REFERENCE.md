# Quick Reference: Radar Data Processing Pipeline

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RADAR SIGNAL PROCESSING FLOW                     │
└─────────────────────────────────────────────────────────────────────┘

STEP 0: PHYSICAL SCENE
┌──────────────────────────────┐
│   4 Cars at different        │
│   positions & velocities     │
│                              │
│   C1: 50m, +15°, -15m/s     │
│   C2: 35m, -20°, -10m/s     │
│   C3: 70m, +5°,  -8m/s      │
│   C4: 20m, -5°,  -20m/s     │
└──────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: RADAR TRANSMISSION & RECEPTION                       │
│                                                               │
│  Transmit: 77 GHz FMCW chirps (40 μs duration, 150 MHz BW)  │
│  Receive:  32-channel array (8 azimuth × 4 elevation)       │
│  Result:   Mixed signal with beat freq, Doppler, phase      │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: DATA CUBE CREATION                                   │
│                                                               │
│   datacube [200 × 128 × 32]                                  │
│            └─┬─┘ └─┬┘  └┬┘                                   │
│              │     │    └─ 32 antenna channels               │
│              │     └────── 128 chirps (slow-time)            │
│              └──────────── 200 samples/chirp (fast-time)     │
│                                                               │
│   Size: ~3 MB complex data                                   │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: RANGE FFT (Fast-Time Processing)                     │
│                                                               │
│   Input:  [200 × 128 × 32]                                   │
│   FFT on dimension 1 → Extract beat frequencies              │
│   Output: [512 × 128 × 32]  (zero-padded)                    │
│                                                               │
│   Beat Frequency → Range:  R = f_beat × c×T / (2×B)         │
│   Resolution: 1 meter                                         │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: DOPPLER FFT (Slow-Time Processing)                   │
│                                                               │
│   Input:  [512 × 128 × 32]                                   │
│   FFT on dimension 2 → Extract Doppler shifts                │
│   Output: [512 × 256 × 32]  (zero-padded)                    │
│                                                               │
│   Doppler → Velocity:  v = f_d × c / (2×fc)                 │
│   Resolution: ~0.5 m/s                                        │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: RESHAPE FOR ANGLE PROCESSING                         │
│                                                               │
│   Separate antenna channels into 2D array:                   │
│   [512 × 256 × 32] → [512 × 256 × 4 × 8]                     │
│                                    └─┬┘ └┬┘                   │
│                                      │   └─ 8 azimuth         │
│                                      └───── 4 elevation       │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 6A: ELEVATION FFT                                       │
│                                                               │
│   Input:  [512 × 256 × 4 × 8]                                │
│   FFT on dimension 3 → Extract elevation angles              │
│   Output: [512 × 256 × 64 × 8]  (zero-padded)                │
│                                                               │
│   Phase difference → Elevation: θ = arcsin(Δφ×λ/(2πd))      │
│   Resolution: ~14°                                            │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 6B: AZIMUTH FFT                                         │
│                                                               │
│   Input:  [512 × 256 × 64 × 8]                               │
│   FFT on dimension 4 → Extract azimuth angles                │
│   Output: [512 × 256 × 64 × 128]  (zero-padded)              │
│                                                               │
│   Phase difference → Azimuth: θ = arcsin(Δφ×λ/(2πd))        │
│   Resolution: ~7°                                             │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: 4D TENSOR                                            │
│                                                               │
│   Complete 4D representation:                                 │
│   [Range × Doppler × Elevation × Azimuth]                    │
│   [512   × 256     × 64        × 128]                        │
│                                                               │
│   Each voxel contains complex radar return                    │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 8: CREATE 3D HEATMAP                                    │
│                                                               │
│   Select zero-velocity slice (Doppler ≈ 0)                   │
│   Take magnitude: |signal|                                    │
│   Convert to dB: 20×log10(magnitude)                         │
│   Normalize to 0 dB peak                                      │
│                                                               │
│   Output: [450 × 64 × 128]                                    │
│           Range × Elevation × Azimuth                         │
│                                                               │
│   3D SPATIAL HEATMAP showing target locations                │
└──────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 9: VISUALIZATIONS                                       │
│                                                               │
│   → 3D Scatter Plot (X,Y,Z in meters)                        │
│   → Range-Azimuth Map (top-down view)                        │
│   → Range-Elevation Map (side view)                          │
│   → Azimuth-Elevation Map (angular view)                     │
│   → Range Profile (1D)                                        │
│   → Angular Profiles (azimuth & elevation)                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Dimensions at Each Stage

```
Stage                   Dimensions              What It Represents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw ADC                 [200×128×32]            Time domain samples
                        
After Range FFT         [512×128×32]            Range resolved
                        └─┬─┘                    
                          └─ 512 range bins     0-100m
                        
After Doppler FFT       [512×256×32]            Range + Velocity resolved
                             └─┬─┘               
                               └─ 256 vel bins  -50 to +50 m/s
                        
After Reshape           [512×256×4×8]           Spatial array separated
                                  └┬┘└┬┘          
                                   │  └─ 8 az    
                                   └──── 4 el    
                        
After Elevation FFT     [512×256×64×8]          Elevation resolved
                                   └─┬─┘          
                                     └─ 64 bins  -90° to +90°
                        
After Azimuth FFT       [512×256×64×128]        Full 4D datacube
                                       └──┬──┘    
                                          └─ 128  -90° to +90°
                        
Final 3D Heatmap        [450×64×128]            Spatial map only
(Doppler integrated)     └┬┘ └┬┘ └─┬─┘          (velocity removed)
                          │   │    └─ Azimuth    
                          │   └────── Elevation  
                          └────────── Range      
```

---

## Key Equations

### 1. Range from Beat Frequency
```
f_beat = (2 × B × R) / (c × T_chirp)

R = (f_beat × c × T_chirp) / (2 × B)

Example: f_beat = 66.67 kHz → R = 50 m
```

### 2. Velocity from Doppler
```
f_doppler = (2 × v × f_c) / c

v = (f_doppler × c) / (2 × f_c)

Example: f_d = -7.7 kHz → v = -15 m/s (approaching)
```

### 3. Angle from Phase
```
Δφ = (2π × d × sin(θ)) / λ

θ = arcsin((Δφ × λ) / (2π × d))

Example: Δφ = 90° between antennas → θ = 15°
```

### 4. Resolution Formulas
```
Range Resolution:     Δr = c / (2 × B)
Velocity Resolution:  Δv = λ / (2 × T_CPI)
Angular Resolution:   Δθ ≈ λ / (N × d)

where T_CPI = coherent processing interval = N_chirps × T_chirp
```

---

## Understanding the Heatmap Values

### What does each number mean?

```
heatmap_3d_db[r, e, a] = X dB

Where:
  r = range bin (0-449)       → 0 to 100 meters
  e = elevation bin (0-63)    → -90° to +90°
  a = azimuth bin (0-127)     → -90° to +90°
  X = power in dB             → relative to peak

Examples:
  heatmap_3d_db[250, 32, 70] = -2 dB   ← Strong return, likely target
  heatmap_3d_db[100, 20, 50] = -40 dB  ← Weak/noise
  heatmap_3d_db[300, 32, 85] = 0 dB    ← Peak (strongest return)
```

### dB Scale Interpretation
```
  0 dB  ─────────  Peak return (normalized)
 -5 dB  ─────────  Very strong (32% of peak power)
-10 dB  ─────────  Strong (10% of peak power)
-20 dB  ─────────  Moderate (1% of peak power)
-30 dB  ─────────  Weak (0.1% of peak power)
-40 dB  ─────────  Very weak / noise floor
```

---

## Coordinate System Conversions

### Spherical → Cartesian
```matlab
% From (Range, Azimuth, Elevation) → (X, Y, Z)

X = R × cos(elevation) × cos(azimuth)     % Forward
Y = R × cos(elevation) × sin(azimuth)     % Lateral (left/right)
Z = R × sin(elevation)                     % Vertical (up/down)
```

### Example Target Positions
```
Car 1: R=50m, Az=+15°, El=0°
  → X = 50×cos(0°)×cos(15°) = 48.3 m
  → Y = 50×cos(0°)×sin(15°) = 12.9 m
  → Z = 50×sin(0°) = 0 m
  Position: (48.3, 12.9, 0) meters

Car 2: R=35m, Az=-20°, El=+1°
  → X = 35×cos(1°)×cos(-20°) = 32.8 m
  → Y = 35×cos(1°)×sin(-20°) = -12.0 m
  → Z = 35×sin(1°) = 0.6 m
  Position: (32.8, -12.0, 0.6) meters
```

---

## Common Analysis Tasks

### 1. Find Targets Above Threshold
```matlab
threshold = -30;  % dB
[r_idx, e_idx, a_idx] = find(heatmap_3d_db > threshold);

% Convert indices to physical coordinates
ranges = range_bins(r_idx);
elevations = elevation_bins(e_idx);
azimuths = azimuth_bins(a_idx);
```

### 2. Extract Peak Detections
```matlab
% Find local maxima
peaks = imregionalmax(heatmap_3d_db);
[pr, pe, pa] = find(peaks & (heatmap_3d_db > -30));

% Get positions of detected targets
detected_ranges = range_bins(pr);
detected_azimuths = azimuth_bins(pa);
detected_elevations = elevation_bins(pe);
```

### 3. Get Range-Azimuth Slice
```matlab
% Maximum projection over elevation (top-down view)
ra_map = squeeze(max(heatmap_3d_db, [], 2));

% Or specific elevation angle (e.g., 0°)
[~, el_idx] = min(abs(elevation_bins - 0));
ra_slice = squeeze(heatmap_3d_db(:, el_idx, :));
```

### 4. Calculate SNR for Detection
```matlab
% Find noise floor (bottom 10% of values)
noise_floor = prctile(heatmap_3d_db(:), 10);

% Calculate SNR for each detection
signal_power = heatmap_3d_db(detection_indices);
SNR_dB = signal_power - noise_floor;
```

---

## Signal Processing Parameters Summary

```
┌─────────────────────────────────────────────────────────┐
│ Parameter              Value          Impact             │
├─────────────────────────────────────────────────────────┤
│ Center Frequency       77 GHz         Wavelength, size   │
│ Bandwidth             150 MHz         Range resolution   │
│ Chirp Duration         40 μs          Max range          │
│ Sample Rate            5 MHz          Nyquist limit      │
│ Number of Chirps       128            Velocity res       │
│ Azimuth Elements       8              Angular res (H)    │
│ Elevation Elements     4              Angular res (V)    │
│ Element Spacing        λ/2            Grating lobes      │
│ Range FFT Size         512            Interpolation      │
│ Doppler FFT Size       256            Velocity bins      │
│ Elevation FFT Size     64             Angle bins (V)     │
│ Azimuth FFT Size       128            Angle bins (H)     │
└─────────────────────────────────────────────────────────┘
```

---

## Memory Usage

```
Data Structure             Size                Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw datacube              200×128×32          ~3 MB
Range FFT                 512×128×32          ~8 MB
Doppler FFT               512×256×32          ~16 MB
4D tensor (full)          512×256×64×128      ~1 GB
3D heatmap (final)        450×64×128          ~3.5 MB

Total workspace:          ~1.1 GB
```

---

## Interpretation Guide

### What makes a good detection?

✅ **Strong Signal:**
- Peak above -20 dB
- Clear local maximum
- SNR > 15 dB

✅ **Spatial Coherence:**
- Forms cluster in 3D space
- Multiple adjacent bins active
- Consistent across angles

✅ **Physical Plausibility:**
- Range < 100m (within radar capability)
- Velocity reasonable for cars (-30 to 0 m/s)
- Elevation near 0° (ground level)

❌ **False Alarm Indicators:**
- Isolated single bins
- Unusual elevation (>10°)
- Inconsistent across views
- Near noise floor

---

## Next Steps / Advanced Topics

1. **CFAR Detection:** Constant False Alarm Rate thresholding
2. **Tracking:** Kalman filtering for multi-frame association
3. **Clustering:** DBSCAN for grouping detections
4. **Classification:** ML models to identify vehicle types
5. **Interference Mitigation:** Multi-radar coexistence
6. **Super-Resolution:** MUSIC, ESPRIT for better angular accuracy

---

This completes the quick reference guide for understanding the radar heatmap data!
