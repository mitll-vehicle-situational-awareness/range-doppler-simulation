%% Optimized Range-Azimuth-Elevation Heatmap with Better Visualization
% Enhanced version with stronger signal returns and better processing

clear all; close all; clc;

%% Radar System Parameters
fc = 77e9;              % Center frequency (77 GHz - automotive radar)
c = physconst('LightSpeed');
lambda = c/fc;
rangeres = 1;           % Range resolution (m)

% Radar configuration
Bw = c/(2*rangeres);    % Bandwidth (150 MHz for 1m resolution)
sweeptime = 40e-6;      % Sweep time (40 microseconds)
fs = 5e6;               % Sample rate (5 MHz)

% FMCW Waveform
waveform = phased.FMCWWaveform('SweepTime',sweeptime,'SweepBandwidth',Bw,...
    'SampleRate',fs,'SweepDirection','Up');

% Number of chirps and samples
Nchirp = 128;           % Number of chirps
Nsamp = round(waveform.SampleRate * waveform.SweepTime);

%% Antenna Array Configuration
Nel_az = 8;             % Number of elements in azimuth
Nel_el = 4;             % Number of elements in elevation
d = lambda/2;           % Element spacing

% Create rectangular array
antenna = phased.URA('Size',[Nel_el Nel_az],'ElementSpacing',[d d]);

% Transmitter and Receiver with higher gain
transmitter = phased.Transmitter('PeakPower',0.01,'Gain',40);  % Increased power and gain
receiver = phased.ReceiverPreamp('Gain',35,'NoiseFigure',3,'SampleRate',fs);

% Radiator and Collector
radiator = phased.Radiator('Sensor',antenna,'OperatingFrequency',fc);
collector = phased.Collector('Sensor',antenna,'OperatingFrequency',fc);

%% Target Configuration - Multiple Cars
targets = struct();

targets(1).range = 50;
targets(1).velocity = -15;
targets(1).azimuth = 15;
targets(1).elevation = 0;
targets(1).rcs = 15;      % Increased RCS

targets(2).range = 35;
targets(2).velocity = -10;
targets(2).azimuth = -20;
targets(2).elevation = 1;
targets(2).rcs = 12;

targets(3).range = 70;
targets(3).velocity = -8;
targets(3).azimuth = 5;
targets(3).elevation = -1;
targets(3).rcs = 13;

targets(4).range = 20;
targets(4).velocity = -20;
targets(4).azimuth = -5;
targets(4).elevation = 0.5;
targets(4).rcs = 16;

Ntargets = length(targets);

fprintf('Simulating %d car targets:\n', Ntargets);
for i = 1:Ntargets
    fprintf('  Car %d: R=%.0fm, Az=%+.0f°, El=%+.1f°, V=%.0fm/s, RCS=%.0fdBsm\n', ...
        i, targets(i).range, targets(i).azimuth, targets(i).elevation, ...
        targets(i).velocity, targets(i).rcs);
end

radar_pos = [0; 0; 0];

%% Signal Simulation
fprintf('\nSimulating radar returns...\n');
datacube = zeros(Nsamp, Nchirp, Nel_az*Nel_el);

% Propagation channel
channel = phased.FreeSpace('PropagationSpeed',c,'OperatingFrequency',fc,...
    'TwoWayPropagation',true,'SampleRate',fs);

for chirp_idx = 1:Nchirp
    sig = waveform();
    rxsig_total = zeros(length(sig), Nel_az*Nel_el);
    
    for tgt_idx = 1:Ntargets
        % Create target
        tgt = phased.RadarTarget('MeanRCS',targets(tgt_idx).rcs,...
            'PropagationSpeed',c,'OperatingFrequency',fc);
        
        % Target position
        [tgt_x, tgt_y, tgt_z] = sph2cart(deg2rad(targets(tgt_idx).azimuth), ...
            deg2rad(targets(tgt_idx).elevation), targets(tgt_idx).range);
        tgt_pos = [tgt_x; tgt_y; tgt_z];
        
        % Velocity
        tgt_vel = [targets(tgt_idx).velocity*cosd(targets(tgt_idx).azimuth)*cosd(targets(tgt_idx).elevation); ...
                   targets(tgt_idx).velocity*sind(targets(tgt_idx).azimuth)*cosd(targets(tgt_idx).elevation); ...
                   targets(tgt_idx).velocity*sind(targets(tgt_idx).elevation)];
        current_pos = tgt_pos + tgt_vel * (chirp_idx-1) * sweeptime;
        
        % Angle to target
        [~, ang] = rangeangle(current_pos, radar_pos);
        
        % Transmit
        txsig = transmitter(sig);
        txsig = radiator(txsig, ang);
        
        % Propagate
        txsig = channel(txsig, radar_pos, current_pos, [0;0;0], tgt_vel);
        txsig = tgt(txsig);
        
        % Receive
        rxsig = collector(txsig, ang);
        
        % Doppler
        doppler_shift = 2*targets(tgt_idx).velocity*fc/c;
        t = (0:length(rxsig)-1)'/fs;
        for elem = 1:(Nel_az*Nel_el)
            rxsig(:,elem) = rxsig(:,elem) .* exp(1j*2*pi*doppler_shift*t);
        end
        
        rxsig_total = rxsig_total + rxsig;
    end
    
    rxsig_total = receiver(rxsig_total);
    datacube(:, chirp_idx, :) = rxsig_total;
end

fprintf('Data cube generated: [%d × %d × %d]\n', size(datacube));

%% Apply Windowing
fprintf('Applying windows...\n');
win_range = hamming(Nsamp);
datacube = datacube .* win_range;

win_doppler = hann(Nchirp)';
datacube = datacube .* reshape(win_doppler, 1, Nchirp, 1);

%% Range Processing
fprintf('Range FFT...\n');
Nfft_range = 512;
range_fft = fft(datacube, Nfft_range, 1);

max_range = fs*c/(2*Bw);
range_bins = linspace(0, max_range, Nfft_range);
range_idx = range_bins <= 100;
range_bins = range_bins(range_idx);
range_fft = range_fft(range_idx, :, :);

%% Doppler Processing
fprintf('Doppler FFT...\n');
Nfft_doppler = 256;
range_doppler_fft = fftshift(fft(range_fft, Nfft_doppler, 2), 2);

%% Angle Processing
fprintf('Angle FFT...\n');
rd_map = reshape(range_doppler_fft, size(range_doppler_fft,1), ...
    size(range_doppler_fft,2), Nel_el, Nel_az);

Nfft_el = 64;
el_fft = fftshift(fft(rd_map, Nfft_el, 3), 3);

Nfft_az = 128;
rae_cube = fftshift(fft(el_fft, Nfft_az, 4), 4);

%% Create Heatmap
fprintf('Creating 3D heatmap...\n');
rae_heatmap = abs(rae_cube);
rae_heatmap_db = 20*log10(rae_heatmap + eps);
rae_heatmap_db = rae_heatmap_db - max(rae_heatmap_db(:));

% Angle bins
sin_az = linspace(-1, 1, Nfft_az);
azimuth_bins = asind(sin_az);

sin_el = linspace(-1, 1, Nfft_el);
elevation_bins = asind(sin_el);

% Integrate around zero Doppler for robustness
doppler_center = floor(Nfft_doppler/2) + 1;
doppler_range = max(1,doppler_center-3):min(Nfft_doppler,doppler_center+3);
heatmap_3d = squeeze(mean(rae_heatmap_db(:, doppler_range, :, :), 2));

fprintf('\n3D Heatmap: [%d × %d × %d]\n', size(heatmap_3d,1), size(heatmap_3d,2), size(heatmap_3d,3));

%% Enhanced Visualization
figure('Position', [50 50 1600 1000]);
set(gcf, 'Color', 'w');

threshold_db = -35;  % Adjusted threshold

%% 1. 3D Volumetric Plot
subplot(2,3,1);
[R, Az, El] = meshgrid(range_bins, azimuth_bins, elevation_bins);
R = permute(R, [2 1 3]);
Az = permute(Az, [2 1 3]);
El = permute(El, [2 1 3]);

X = R .* cosd(El) .* cosd(Az);
Y = R .* cosd(El) .* sind(Az);
Z = R .* sind(El);

mask = heatmap_3d > threshold_db;
scatter3(X(mask), Y(mask), Z(mask), 40, heatmap_3d(mask), 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet);
colorbar;
xlabel('X (m)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Y (m)', 'FontWeight', 'bold', 'FontSize', 11);
zlabel('Z (m)', 'FontWeight', 'bold', 'FontSize', 11);
title('3D Volumetric Heatmap', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
view(45, 25);
caxis([threshold_db 0]);

hold on;
for i = 1:Ntargets
    [tx, ty, tz] = sph2cart(deg2rad(targets(i).azimuth), ...
        deg2rad(targets(i).elevation), targets(i).range);
    plot3(tx, ty, tz, 'r*', 'MarkerSize', 25, 'LineWidth', 3);
    text(tx+2, ty, tz+1, sprintf('C%d', i), 'Color', 'red', ...
        'FontWeight', 'bold', 'FontSize', 10);
end

%% 2. Range-Azimuth
subplot(2,3,2);
ra_slice = squeeze(max(heatmap_3d, [], 2));
imagesc(azimuth_bins, range_bins, ra_slice);
axis xy;
colormap(jet);
h = colorbar;
ylabel(h, 'dB', 'FontSize', 10);
xlabel('Azimuth (°)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Range (m)', 'FontWeight', 'bold', 'FontSize', 11);
title('Range-Azimuth Heatmap', 'FontWeight', 'bold', 'FontSize', 12);
caxis([threshold_db 0]);
grid on;

hold on;
for i = 1:Ntargets
    plot(targets(i).azimuth, targets(i).range, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
    text(targets(i).azimuth+3, targets(i).range, sprintf('C%d', i), ...
        'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9, ...
        'BackgroundColor', 'black', 'Margin', 1);
end

%% 3. Range-Elevation
subplot(2,3,3);
re_slice = squeeze(max(heatmap_3d, [], 3));
imagesc(elevation_bins, range_bins, re_slice);
axis xy;
colormap(jet);
h = colorbar;
ylabel(h, 'dB', 'FontSize', 10);
xlabel('Elevation (°)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Range (m)', 'FontWeight', 'bold', 'FontSize', 11);
title('Range-Elevation Heatmap', 'FontWeight', 'bold', 'FontSize', 12);
caxis([threshold_db 0]);
grid on;

hold on;
for i = 1:Ntargets
    plot(targets(i).elevation, targets(i).range, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
end

%% 4. Azimuth-Elevation
subplot(2,3,4);
% Combine slices at target ranges
ae_combined = zeros(length(elevation_bins), length(azimuth_bins));
for i = 1:Ntargets
    [~, ridx] = min(abs(range_bins - targets(i).range));
    ae_combined = ae_combined + squeeze(heatmap_3d(ridx, :, :));
end
ae_combined = ae_combined / Ntargets;

imagesc(azimuth_bins, elevation_bins, ae_combined);
axis xy;
colormap(jet);
h = colorbar;
ylabel(h, 'dB', 'FontSize', 10);
xlabel('Azimuth (°)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Elevation (°)', 'FontWeight', 'bold', 'FontSize', 11);
title('Azimuth-Elevation (Combined Target Ranges)', 'FontWeight', 'bold', 'FontSize', 12);
caxis([threshold_db 0]);
grid on;

hold on;
for i = 1:Ntargets
    plot(targets(i).azimuth, targets(i).elevation, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
    text(targets(i).azimuth+2, targets(i).elevation, sprintf('C%d', i), ...
        'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9, ...
        'BackgroundColor', 'black', 'Margin', 1);
end

%% 5. Range Profile
subplot(2,3,5);
range_profile = squeeze(max(max(heatmap_3d, [], 2), [], 3));
plot(range_bins, range_profile, 'b-', 'LineWidth', 2.5);
grid on;
xlabel('Range (m)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Magnitude (dB)', 'FontWeight', 'bold', 'FontSize', 11);
title('Range Profile', 'FontWeight', 'bold', 'FontSize', 12);
xlim([0 100]);
ylim([threshold_db 5]);

hold on;
for i = 1:Ntargets
    xline(targets(i).range, 'r--', sprintf('C%d', i), ...
        'LineWidth', 2, 'FontSize', 9, 'FontWeight', 'bold', ...
        'LabelVerticalAlignment', 'bottom');
end

%% 6. Angular Profiles
subplot(2,3,6);
% Azimuth profile: max over range and elevation
az_profile = squeeze(max(max(heatmap_3d, [], 1), [], 2));
% Elevation profile: max over range and azimuth
el_profile = squeeze(max(max(heatmap_3d, [], 1), [], 3));

% Ensure proper dimensions (make them row vectors)
az_profile = az_profile(:)';
el_profile = el_profile(:)';

% Debug dimensions
fprintf('Debug Angular Profiles:\n');
fprintf('  azimuth_bins: %d, az_profile: %d\n', length(azimuth_bins), length(az_profile));
fprintf('  elevation_bins: %d, el_profile: %d\n', length(elevation_bins), length(el_profile));

% Plot only if dimensions match
yyaxis left
if length(azimuth_bins) == length(az_profile)
    plot(azimuth_bins, az_profile, 'b-', 'LineWidth', 2.5);
else
    fprintf('Warning: Azimuth dimension mismatch, skipping plot\n');
end
ylabel('Azimuth (dB)', 'FontWeight', 'bold', 'FontSize', 10);
ylim([threshold_db 5]);

yyaxis right
if length(elevation_bins) == length(el_profile)
    plot(elevation_bins, el_profile, 'r-', 'LineWidth', 2.5);
else
    fprintf('Warning: Elevation dimension mismatch, skipping plot\n');
end
ylabel('Elevation (dB)', 'FontWeight', 'bold', 'FontSize', 10);
ylim([threshold_db 5]);

grid on;
xlabel('Angle (°)', 'FontWeight', 'bold', 'FontSize', 11);
title('Angular Profiles', 'FontWeight', 'bold', 'FontSize', 12);
legend('Azimuth', 'Elevation', 'Location', 'best', 'FontSize', 9);

sgtitle(sprintf('Enhanced Range-Azimuth-Elevation Analysis - %d Targets', Ntargets), ...
    'FontSize', 16, 'FontWeight', 'bold');

%% Save Results
fprintf('\nSaving results...\n');
save('/mnt/user-data/outputs/radar_rae_optimized.mat', 'heatmap_3d', ...
    'range_bins', 'azimuth_bins', 'elevation_bins', 'targets', 'datacube');

saveas(gcf, '/mnt/user-data/outputs/radar_rae_optimized.png');
saveas(gcf, '/mnt/user-data/outputs/radar_rae_optimized.fig');

fprintf('\n✓ Simulation Complete!\n');
fprintf('\nTarget Summary:\n');
for i = 1:Ntargets
    fprintf('  Car %d: R=%.0fm, Az=%+.0f°, El=%+.1f°\n', ...
        i, targets(i).range, targets(i).azimuth, targets(i).elevation);
end
fprintf('\n3D Tensor: [%d × %d × %d] (Range × Elevation × Azimuth)\n', ...
    length(range_bins), length(elevation_bins), length(azimuth_bins));