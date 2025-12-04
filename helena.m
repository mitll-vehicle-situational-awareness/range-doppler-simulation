% Automotive Radar Driving Scene Simulation
% Simulates radar data from a realistic driving scenario

clear all; close all; clc;

%% Radar Parameters (Typical 77 GHz Automotive Radar)
fc = 77e9;              % Carrier frequency (77 GHz)
c = 3e8;                % Speed of light (m/s)
lambda = c/fc;          % Wavelength
PRF = 2000;             % Pulse Repetition Frequency (Hz)
PRI = 1/PRF;            % Pulse Repetition Interval
fs = 50e6;              % Sampling frequency (50 MHz)
pulse_width = 2e-6;     % Pulse width (2 microseconds)
num_pulses = 256;       % Number of pulses in coherent processing interval
max_range = 200;        % Maximum range (200 m)
range_resolution = 1;   % Range resolution (m)
azimuth_fov = 60;       % Azimuth field of view (degrees)
azimuth_bins = 64;      % Number of azimuth bins

%% Ego Vehicle Parameters
ego_velocity = 25;      % Ego vehicle speed (25 m/s = 90 km/h)

%% Define Driving Scene Objects
% Format: [range(m), azimuth(deg), velocity(m/s), RCS(m²), type]
% velocity is relative to ground, will be converted to relative velocity
scene_objects = {
    % Leading vehicles
    30,   0,  20,  15, 'car';           % Car ahead, slightly slower
    45,  -5,  18,  12, 'car';           % Car in left lane
    
    % Oncoming traffic
    80,   8, -30,  10, 'car';           % Oncoming car
    95,  10, -28,  12, 'truck';         % Oncoming truck
    
    % Crossing pedestrian
    25,  20,   2, 0.5, 'pedestrian';    % Pedestrian crossing
    
    % Stationary objects
    15, -25,   0,  20, 'guardrail';     % Guardrail
    50,  28,   0,   5, 'sign';          % Road sign
    
    % Vehicles at various distances
    70,  -3,  22,  14, 'car';           % Car in adjacent lane
    120,  0,  15,  18, 'truck';         % Distant truck
    
    % Motorcycle
    35,  15,  24,   3, 'motorcycle';    % Motorcycle
};

num_objects = size(scene_objects, 1);

%% Convert to Radar-Relative Coordinates
fprintf('\n=== Driving Scene Simulation ===\n');
fprintf('Ego Vehicle Speed: %.1f m/s (%.1f km/h)\n\n', ego_velocity, ego_velocity*3.6);

targets = zeros(num_objects, 4);  % [range, azimuth, relative_velocity, RCS]

fprintf('Scene Objects:\n');
fprintf('%-12s | Range(m) | Azimuth(°) | Velocity(m/s) | Rel.Vel(m/s) | RCS(m²)\n', 'Type');
fprintf('-------------|----------|------------|---------------|--------------|--------\n');

for i = 1:num_objects
    range = scene_objects{i, 1};
    azimuth = scene_objects{i, 2};
    velocity = scene_objects{i, 3};
    rcs = scene_objects{i, 4};
    obj_type = scene_objects{i, 5};
    
    % Calculate relative velocity (positive = approaching ego vehicle)
    % For objects ahead: positive velocity = closing (approaching)
    % For objects behind: negative velocity = receding
    % We need the radial component of relative velocity
    relative_velocity = (ego_velocity - velocity) * cosd(azimuth);
    
    targets(i, :) = [range, azimuth, relative_velocity, rcs];
    
    fprintf('%-12s | %8.1f | %10.1f | %13.1f | %12.1f | %6.2f\n', ...
        obj_type, range, azimuth, velocity, relative_velocity, rcs);
end

%% Time Vectors
t_fast = 0:1/fs:2*max_range/c;          % Fast time (range)
t_slow = 0:PRI:(num_pulses-1)*PRI;      % Slow time (Doppler)
num_range_samples = length(t_fast);

%% Generate Radar Data Cube (Range x Doppler x Azimuth)
radar_cube = zeros(num_range_samples, num_pulses, azimuth_bins);

% Azimuth angles
azimuth_angles = linspace(-azimuth_fov/2, azimuth_fov/2, azimuth_bins);

for i = 1:num_objects
    range_target = targets(i, 1);
    azimuth_target = targets(i, 2);
    velocity_rel = targets(i, 3);
    rcs = targets(i, 4);
    
    % Skip if outside FOV
    if abs(azimuth_target) > azimuth_fov/2
        continue;
    end
    
    % Calculate two-way delay
    tau = 2 * range_target / c;
    
    % Calculate Doppler frequency
    fd = 2 * velocity_rel / lambda;
    
    % Range bin
    range_bin = round(tau * fs);
    
    % Azimuth bin (with Gaussian spread for antenna pattern)
    [~, azimuth_bin_center] = min(abs(azimuth_angles - azimuth_target));
    azimuth_spread = 2;  % Gaussian spread in bins
    
    if range_bin > 0 && range_bin < num_range_samples
        % Generate signal for this target
        pulse_samples = round(pulse_width * fs);
        
        for pulse = 1:num_pulses
            % Doppler phase shift
            doppler_phase = 2 * pi * fd * t_slow(pulse);
            amplitude = sqrt(rcs) * exp(1j * doppler_phase);
            
            % Add to radar cube with azimuth pattern
            for az_idx = max(1, azimuth_bin_center-5):min(azimuth_bins, azimuth_bin_center+5)
                % Gaussian azimuth pattern
                azimuth_gain = exp(-((az_idx - azimuth_bin_center)^2) / (2*azimuth_spread^2));
                
                if range_bin + pulse_samples <= num_range_samples
                    radar_cube(range_bin:range_bin+pulse_samples-1, pulse, az_idx) = ...
                        radar_cube(range_bin:range_bin+pulse_samples-1, pulse, az_idx) + ...
                        amplitude * azimuth_gain;
                end
            end
        end
    end
end

%% Add Noise and Clutter
SNR_dB = 15;                            % Signal-to-noise ratio
noise_power = 10^(-SNR_dB/10);
noise = sqrt(noise_power/2) * (randn(size(radar_cube)) + 1j*randn(size(radar_cube)));
radar_cube = radar_cube + noise;

% Add ground clutter (stationary returns at various ranges)
clutter_ranges = [10, 20, 35, 60, 100, 150];
for cr = clutter_ranges
    clutter_bin = round(2*cr/c * fs);
    if clutter_bin < num_range_samples
        clutter_rcs = 0.1;  % Weak clutter
        for az = 1:azimuth_bins
            radar_cube(clutter_bin, :, az) = radar_cube(clutter_bin, :, az) + ...
                sqrt(clutter_rcs) * (randn(1, num_pulses) + 1j*randn(1, num_pulses)) * 0.3;
        end
    end
end

%% Range-Doppler Processing
% Process for center azimuth beam
center_beam = radar_cube(:, :, round(azimuth_bins/2));
range_doppler = fftshift(fft(center_beam, [], 2), 2);

%% Create Axes
range_axis = t_fast * c / 2;            % Range in m
doppler_axis = (-num_pulses/2:num_pulses/2-1) * PRF / num_pulses;
velocity_axis = -doppler_axis * lambda / 2;  % Relative velocity (m/s)

%% Plotting
figure('Position', [50, 50, 1400, 900]);

% Plot 1: Range-Doppler Map (Center Beam)
subplot(2,3,1);
imagesc(velocity_axis, range_axis, 20*log10(abs(range_doppler) + eps));
xlabel('Relative Velocity (m/s)');
ylabel('Range (m)');
title('Range-Doppler Map (Center Beam)');
colorbar;
colormap(jet);
ylim([0 max_range]);
xlim([-50 50]);
caxis([max(20*log10(abs(range_doppler(:))))-60, max(20*log10(abs(range_doppler(:))))]);
grid on;

% Plot 2: Bird's Eye View
subplot(2,3,2);
hold on;
for i = 1:num_objects
    range = targets(i, 1);
    azimuth = targets(i, 2);
    x = range * sind(azimuth);
    y = range * cosd(azimuth);
    
    % Color code by relative velocity
    if targets(i, 3) > 5
        color = 'r';  % Approaching
    elseif targets(i, 3) < -5
        color = 'b';  % Receding
    else
        color = 'k';  % Stationary
    end
    
    plot(x, y, 'o', 'MarkerSize', 8, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k');
end

% Draw ego vehicle
plot(0, 0, 's', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
plot([0 0], [0 10], 'g-', 'LineWidth', 3);  % Direction arrow

xlabel('Lateral Distance (m)');
ylabel('Longitudinal Distance (m)');
title('Bird''s Eye View');
grid on;
axis equal;
xlim([-60 60]);
ylim([0 max_range]);
legend('Approaching', 'Receding', 'Stationary', 'Ego Vehicle', 'Location', 'best');
hold off;

% Plot 3: Range Profile
subplot(2,3,3);
range_profile = mean(abs(center_beam), 2);
plot(range_axis, 20*log10(range_profile + eps), 'LineWidth', 1.5);
xlabel('Range (m)');
ylabel('Magnitude (dB)');
title('Average Range Profile');
grid on;
xlim([0 max_range]);

% Plot 4: Range-Azimuth Map (Integrate over Doppler)
subplot(2,3,4);
range_azimuth = squeeze(mean(abs(radar_cube), 2));
imagesc(azimuth_angles, range_axis, 20*log10(range_azimuth + eps));
xlabel('Azimuth Angle (degrees)');
ylabel('Range (m)');
title('Range-Azimuth Map');
colorbar;
colormap(jet);
ylim([0 max_range]);
caxis([max(20*log10(range_azimuth(:)))-50, max(20*log10(range_azimuth(:)))]);
grid on;

% Plot 5: Velocity Histogram
subplot(2,3,5);
histogram(targets(:, 3), 20, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Relative Velocity (m/s)');
ylabel('Count');
title('Distribution of Target Velocities');
grid on;

% Plot 6: Detection List
subplot(2,3,6);
axis off;
text(0.1, 0.95, 'Detected Objects:', 'FontSize', 12, 'FontWeight', 'bold');
y_pos = 0.85;
for i = 1:min(10, num_objects)
    str = sprintf('%d: R=%.1fm, Az=%.0f°, V=%.1fm/s', ...
        i, targets(i,1), targets(i,2), targets(i,3));
    text(0.1, y_pos, str, 'FontSize', 9, 'FontName', 'FixedWidth');
    y_pos = y_pos - 0.08;
end

sgtitle('Automotive Radar Driving Scene Simulation', 'FontSize', 14, 'FontWeight', 'bold');

%% Export Data Summary
fprintf('\n=== Radar Configuration ===\n');
fprintf('Frequency: %.1f GHz\n', fc/1e9);
fprintf('PRF: %.0f Hz\n', PRF);
fprintf('Max Range: %.0f m\n', max_range);
fprintf('Range Resolution: %.2f m\n', c/(2*fs));
fprintf('Velocity Resolution: %.2f m/s\n', lambda*PRF/(2*num_pulses));
fprintf('Azimuth FOV: ±%.0f degrees\n', azimuth_fov/2);
fprintf('\nSimulation complete!\n');