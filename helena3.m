% Multi-Sensor Simulation: Camera and Dual Radar System
% Simulates N moving sources with video pixel cube and radar data cubes

clear all; close all; clc;

%% Simulation Parameters
N_sources = 3;              % Number of moving sources
T_duration = 5;             % Duration in seconds
frame_rate = 30;            % Video frame rate (fps)
PRF = 1000;                 % Pulse Repetition Frequency (Hz)
dt = 1/frame_rate;          % Time step
N_frames = T_duration * frame_rate;  % Total number of frames
N_pulses = T_duration * PRF;         % Total number of radar pulses

%% Sensor Positions (in meters)
camera_pos = [0, 0, 5];     % Camera position (x, y, z)
radar1_pos = [-10, 0, 3];   % Radar 1 position
radar2_pos = [10, 0, 3];    % Radar 2 position

fprintf('=== Multi-Sensor Simulation Setup ===\n');
fprintf('Number of sources: %d\n', N_sources);
fprintf('Duration: %.1f seconds\n', T_duration);
fprintf('Video frames: %d (at %d fps)\n', N_frames, frame_rate);
fprintf('Radar pulses: %d (at %d Hz PRF)\n\n', N_pulses, PRF);

%% Define Source Trajectories
% Each source has initial position and velocity
source_trajectories = cell(N_sources, 1);

% Source 1: Moving diagonally
source_trajectories{1}.init_pos = [5, 10, 2];
source_trajectories{1}.velocity = [-1.5, -2, 0.1];
source_trajectories{1}.RCS = 10;  % Radar Cross Section (m^2)
source_trajectories{1}.brightness = 0.8;  % Visual brightness

% Source 2: Circular motion
source_trajectories{2}.init_pos = [0, 15, 3];
source_trajectories{2}.omega = 0.5;  % Angular velocity (rad/s)
source_trajectories{2}.radius = 8;
source_trajectories{2}.RCS = 15;
source_trajectories{2}.brightness = 0.9;

% Source 3: Linear motion
source_trajectories{3}.init_pos = [-8, 8, 1.5];
source_trajectories{3}.velocity = [2, 1, -0.05];
source_trajectories{3}.RCS = 8;
source_trajectories{3}.brightness = 0.7;

%% Generate True Source Positions Over Time
time_video = linspace(0, T_duration, N_frames);
time_radar = linspace(0, T_duration, N_pulses);

% Store positions at video frame times
source_positions_video = zeros(N_sources, 3, N_frames);

for i = 1:N_sources
    for t_idx = 1:N_frames
        t = time_video(t_idx);
        
        if i == 2  % Circular motion
            center = source_trajectories{i}.init_pos;
            omega = source_trajectories{i}.omega;
            radius = source_trajectories{i}.radius;
            
            x = center(1) + radius * cos(omega * t);
            y = center(2) + radius * sin(omega * t);
            z = center(3);
        else  % Linear motion
            pos = source_trajectories{i}.init_pos + ...
                  source_trajectories{i}.velocity * t;
            x = pos(1);
            y = pos(2);
            z = pos(3);
        end
        
        source_positions_video(i, :, t_idx) = [x, y, z];
    end
end

% Store positions at radar pulse times
source_positions_radar = zeros(N_sources, 3, N_pulses);

for i = 1:N_sources
    for t_idx = 1:N_pulses
        t = time_radar(t_idx);
        
        if i == 2  % Circular motion
            center = source_trajectories{i}.init_pos;
            omega = source_trajectories{i}.omega;
            radius = source_trajectories{i}.radius;
            
            x = center(1) + radius * cos(omega * t);
            y = center(2) + radius * sin(omega * t);
            z = center(3);
        else  % Linear motion
            pos = source_trajectories{i}.init_pos + ...
                  source_trajectories{i}.velocity * t;
            x = pos(1);
            y = pos(2);
            z = pos(3);
        end
        
        source_positions_radar(i, :, t_idx) = [x, y, z];
    end
end

%% Generate Video Pixel Cube
img_height = 480;
img_width = 640;
video_cube = zeros(img_height, img_width, N_frames);

% Camera intrinsic parameters
focal_length = 500;  % pixels
cx = img_width / 2;
cy = img_height / 2;

fprintf('Generating video pixel cube...\n');

for t_idx = 1:N_frames
    frame = zeros(img_height, img_width);
    
    for i = 1:N_sources
        % Get source position
        pos = squeeze(source_positions_video(i, :, t_idx));
        
        % Transform to camera coordinates
        pos_cam = pos - camera_pos;
        
        % Project to image plane (pinhole camera model)
        if pos_cam(2) > 0  % Source in front of camera
            u = cx + focal_length * pos_cam(1) / pos_cam(2);
            v = cy + focal_length * pos_cam(3) / pos_cam(2);
            
            % Check if within image bounds
            if u >= 1 && u <= img_width && v >= 1 && v <= img_height
                % Add Gaussian blob for the source
                [X, Y] = meshgrid(1:img_width, 1:img_height);
                sigma = 5;  % Blob size
                intensity = source_trajectories{i}.brightness;
                distance = pos_cam(2);
                intensity = intensity / (distance / 10);  % Intensity falloff
                
                blob = intensity * exp(-((X - u).^2 + (Y - v).^2) / (2 * sigma^2));
                frame = frame + blob;
            end
        end
    end
    
    % Add noise
    frame = frame + 0.02 * randn(img_height, img_width);
    frame = max(0, min(1, frame));  % Clip to [0, 1]
    
    video_cube(:, :, t_idx) = frame;
end

fprintf('Video cube generated: %dx%dx%d\n\n', img_height, img_width, N_frames);

%% Generate Radar Data Cubes
% Radar parameters
N_range_bins = 200;
max_range = 50;  % meters
range_resolution = max_range / N_range_bins;
range_bins = linspace(0, max_range, N_range_bins);

N_antennas = 8;  % Number of antenna elements per radar
c = 3e8;  % Speed of light (m/s)
f_carrier = 10e9;  % 10 GHz carrier frequency
wavelength = c / f_carrier;
antenna_spacing = wavelength / 2;

fprintf('Generating radar data cubes...\n');
fprintf('Range bins: %d (resolution: %.2f m)\n', N_range_bins, range_resolution);
fprintf('Antennas: %d\n\n', N_antennas);

% Initialize radar data cubes: (pulse, range, antenna)
radar1_cube = zeros(N_pulses, N_range_bins, N_antennas);
radar2_cube = zeros(N_pulses, N_range_bins, N_antennas);

for pulse_idx = 1:N_pulses
    for i = 1:N_sources
        % Get source position at this pulse time
        pos = squeeze(source_positions_radar(i, :, pulse_idx));
        RCS = source_trajectories{i}.RCS;
        
        % === Radar 1 ===
        % Calculate range and angle
        vec1 = pos - radar1_pos;
        range1 = norm(vec1);
        azimuth1 = atan2(vec1(1), vec1(2));  % Angle in x-y plane
        
        % Find corresponding range bin
        range_bin1 = round(range1 / range_resolution);
        
        if range_bin1 > 0 && range_bin1 <= N_range_bins
            % Signal strength (simplified radar equation)
            signal_strength1 = sqrt(RCS) / (range1^2 + 1);
            
            % Add signal to each antenna with phase progression
            for ant = 1:N_antennas
                phase_shift = 2 * pi * (ant - 1) * antenna_spacing * ...
                             sin(azimuth1) / wavelength;
                
                radar1_cube(pulse_idx, range_bin1, ant) = ...
                    radar1_cube(pulse_idx, range_bin1, ant) + ...
                    signal_strength1 * exp(1i * phase_shift);
            end
        end
        
        % === Radar 2 ===
        % Calculate range and angle
        vec2 = pos - radar2_pos;
        range2 = norm(vec2);
        azimuth2 = atan2(vec2(1), vec2(2));
        
        % Find corresponding range bin
        range_bin2 = round(range2 / range_resolution);
        
        if range_bin2 > 0 && range_bin2 <= N_range_bins
            % Signal strength
            signal_strength2 = sqrt(RCS) / (range2^2 + 1);
            
            % Add signal to each antenna with phase progression
            for ant = 1:N_antennas
                phase_shift = 2 * pi * (ant - 1) * antenna_spacing * ...
                             sin(azimuth2) / wavelength;
                
                radar2_cube(pulse_idx, range_bin2, ant) = ...
                    radar2_cube(pulse_idx, range_bin2, ant) + ...
                    signal_strength2 * exp(1i * phase_shift);
            end
        end
    end
end

% Add noise to radar data
noise_level = 0.01;
radar1_cube = radar1_cube + noise_level * (randn(size(radar1_cube)) + ...
              1i * randn(size(radar1_cube)));
radar2_cube = radar2_cube + noise_level * (randn(size(radar2_cube)) + ...
              1i * randn(size(radar2_cube)));

fprintf('Radar 1 cube generated: %dx%dx%d (complex)\n', ...
        N_pulses, N_range_bins, N_antennas);
fprintf('Radar 2 cube generated: %dx%dx%d (complex)\n\n', ...
        N_pulses, N_range_bins, N_antennas);

%% Save All Data
fprintf('Saving simulation data...\n');

save('/mnt/user-data/outputs/simulation_data.mat', ...
     'N_sources', 'T_duration', 'frame_rate', 'PRF', ...
     'camera_pos', 'radar1_pos', 'radar2_pos', ...
     'source_trajectories', 'source_positions_video', 'source_positions_radar', ...
     'video_cube', 'radar1_cube', 'radar2_cube', ...
     'time_video', 'time_radar', 'range_bins', ...
     'img_height', 'img_width', 'N_range_bins', 'N_antennas', ...
     '-v7.3');

fprintf('Data saved to: simulation_data.mat\n\n');

fprintf('Simulation complete!\n');