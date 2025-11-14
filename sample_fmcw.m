%% === Clear and Setup ===
clear all %#ok<CLALL>
close all

% --- Radar Hardware Specs ---
f0 = 60e9;                   % Center frequency (Hz)
noRx = 4;                    % Number of receive antennas
noChirps = 32;               % Number of pulses (chirps) in a frame
fs = 12.5e6;                 % ADC sampling rate (Hz)
noADC = 256;                 % Number of ADC samples per chirp
slp = 144.858e12;            % Chirp rate (Hz/s)
PRF = 1 / 200e-6;            % Pulse Repetition Frequency (Hz)

% --- Simulation Control ---
noFrames = 800;              % Number of frames to simulate
frameRate = 155;             % Rate at which frames are collected (Hz)
allFramesData = cell(1, noFrames);  % Cell array for raw echoes

% --- Derived Parameters ---
c = physconst('lightspeed');
tm = noADC / fs;             % Sweep time (s)
bw = slp * tm;               % Sweep bandwidth (Hz)
rangeRes = c / (2 * bw);     % Range resolution (m)
pri = 1 / PRF;               % Pulse Repetition Interval (s)
wavelength = c / f0;         % Wavelength (m)

fprintf('System Range Resolution: %.3f meters\n', rangeRes);

%% === Driving Scenario ===
sampleTime = 1 / frameRate;
stopTime = noFrames * sampleTime;
epsilon = sampleTime / 2;

scenario = drivingScenario('StopTime', stopTime + epsilon, 'SampleTime', sampleTime);
road1 = [50 1 0; 2 0.9 0];
road2 = [27 24 0; 27 -21 0];
laneSpecification = lanespec(2,'Width',4);
road(scenario, road1, 'Lanes', laneSpecification);
road(scenario, road2, 'Lanes', laneSpecification);

egoVehicle = vehicle(scenario,'ClassID',1,'Position',[5 -1 0]);
waypoints = [5 -1 0; 16 -1 0; 40 -1 0];
speed = [30; 0; 30];
waittime = [0; 0.3; 0];
trajectory(egoVehicle, waypoints, speed, waittime);

ambulance = vehicle(scenario,'ClassID',6,'Position',[25 22 0],'PlotColor',[0.466 0.674 0.188],'Name','Ambulance');
waypointsAmb = [25 22 0; 25 13 0; 25 6 0; 26 2 0; 33 -1 0; 45 -1 0];
speedAmb = 25;
trajectory(ambulance, waypointsAmb, speedAmb);
targetVehicle = ambulance;

%% === Preprocessing / System Objects ===
waveform = phased.FMCWWaveform('SweepTime', tm, 'SweepBandwidth', bw, 'SampleRate', fs);

frequencyRange = [f0 - bw/2, f0 + bw/2];
antenna = phased.IsotropicAntennaElement('FrequencyRange', frequencyRange);
rxArray = phased.ULA('NumElements', noRx, 'ElementSpacing', wavelength/2);

transmitter = phased.Transmitter('PeakPower', 0.1, 'Gain', 30);
receiver = phased.ReceiverPreamp('Gain', 20, 'NoiseFigure', 5, 'SampleRate', fs);

radiator = phased.Radiator('Sensor', antenna, 'OperatingFrequency', f0);
collector = phased.Collector('Sensor', rxArray, 'OperatingFrequency', f0);

channel = phased.FreeSpace('PropagationSpeed', c, 'OperatingFrequency', f0, ...
    'SampleRate', fs, 'TwoWayPropagation', false);

target = phased.RadarTarget('MeanRCS', 10, 'Model', 'Nonfluctuating', 'OperatingFrequency', f0);

rdresp = phased.RangeDopplerResponse( ...
    'RangeMethod', 'FFT', ...
    'DopplerOutput', 'Speed', ...
    'SweepSlope', slp, ...
    'SampleRate', fs, ...
    'OperatingFrequency', f0);

%% === Plotting ===
figScenario = figure('Name', 'Driving Scenario', 'Position', [100, 300, 700, 600]);
axScenario = axes(figScenario);
plot(scenario, 'Parent', axScenario);
title(axScenario, 'Live Driving Scenario');

figRD = figure('Name', 'Range-Doppler Map', 'Position', [850, 300, 700, 600]);
axRD = axes(figRD);

[~, range_grid, speed_grid] = rdresp(zeros(noADC, noChirps));
hRD = imagesc(axRD, speed_grid, range_grid, zeros(length(range_grid), length(speed_grid)));
set(axRD, 'YDir', 'normal');
hTitleRD = title(axRD, 'Range-Doppler Map (Initializing...)');
xlabel(axRD, 'Velocity (m/s)');
ylabel(axRD, 'Range (m)');
colorbar(axRD);
clim(axRD, [-20, 40]);

fprintf('Starting simulation...\n');
%% === Preallocate received data ===
rx_puls = zeros(noADC, noRx, noChirps);

% FFT parameters
nRangeFFT = noADC;
nDopplerFFT = noChirps;

% Range vector
rangeVec = (0:nRangeFFT-1)*(c/(2*bw))/nRangeFFT;

% Doppler vector
velVec = (-nDopplerFFT/2:nDopplerFFT/2-1)*(wavelength/(2*pri*nDopplerFFT));

fprintf('Starting simulation...\n');

for frameIdx = 1:noFrames
    % === Scenario info ===
    egoPos = egoVehicle.Position.';
    egoVel = egoVehicle.Velocity.';
    tgtPos = targetVehicle.Position.';
    tgtVel = targetVehicle.Velocity.';

    [~, tgtang] = rangeangle(tgtPos, egoPos);

    % === Loop over chirps ===
    for k = 1:noChirps
        wf = waveform();
        wf = transmitter(wf);
        wf = radiator(wf, tgtang);

        wf = channel(wf, egoPos, tgtPos, egoVel, tgtVel);
        wf = target(wf);
        wf = channel(wf, tgtPos, egoPos, tgtVel, egoVel);

        wf = collector(wf, tgtang);
        rx_puls(:, :, k) = receiver(wf);  % store all antennas
    end

    % === FFT Processing ===
    % Range FFT along 1st dimension (samples)
    rangeFFT = fft(rx_puls, nRangeFFT, 1);  % 256 x 4 x 32
    combinedAnt = mean(rangeFFT, 2);        % average antennas -> 256 x 1 x 32

    % Doppler FFT along 3rd dimension (chirps)
    rdMap = fftshift(fft(combinedAnt, nDopplerFFT, 3), 3); % 256 x 1 x 32
    rdMap = squeeze(rdMap); % 256 x 32 for plotting

    rdMag = mag2db(abs(rdMap));

    % === Peak detection ===
    rdMagClipped = rdMag;
    [~, zeroVelIdx] = min(abs(velVec));
    rdMagClipped(1:5, :) = -Inf; % exclude very close ranges
    velBinsToExclude = [zeroVelIdx-1, zeroVelIdx, zeroVelIdx+1];
    velBinsToExclude = velBinsToExclude(velBinsToExclude > 0 & velBinsToExclude <= numel(velVec));
    rdMagClipped(:, velBinsToExclude) = -Inf;

    [maxVal, maxIdx] = max(rdMagClipped(:));
    TargetPeakThreshold_dB = -100.0;

    if maxVal >= TargetPeakThreshold_dB
        [rangeIdx, velIdx] = ind2sub(size(rdMag), maxIdx);
        detectedRange = rangeVec(rangeIdx);
        detectedVel = velVec(velIdx);
    else
        detectedRange = NaN;
        detectedVel = NaN;
    end

    % === Actual values ===
    relPos = tgtPos(1:2) - egoPos(1:2);
    relVel = tgtVel(1:2) - egoVel(1:2);
    actualRange = norm(relPos);
    actualVel = dot(relVel, relPos) / norm(relPos);

    % === Print comparison ===
    fprintf('Frame %3d | Detected: (%.2f m/s, %.2f m) | Expected: (%.2f m/s, %.2f m)\n', ...
        frameIdx, detectedVel, detectedRange, actualVel, actualRange);

    % === Update RD Map plot ===
    set(hRD, 'CData', rdMag);
    set(hTitleRD, 'String', sprintf('Range-Doppler Map (Frame %d)', frameIdx));
    drawnow;

    % Advance scenario
    advance(scenario);
end

fprintf('Simulation finished.\n');
