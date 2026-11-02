clear; clc;

% Create output folder
output_folder = 'sliced_data';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

dt_orig = 0.005;  % original sampling period in seconds
samplingFreq = 1 / dt_orig;
dt_ds = 0.010;    % downsampled period (10ms)
dsRatio = dt_ds / dt_orig;  % downsampling ratio

gait_speed = 1; % 1 = fast, 2 = normal, 3 = slow
gait_speeds_names = {'fast', 'normal', 'slow'};

if gait_speed == 1
    startIndex1 = 5.5 * samplingFreq;
    endIndex1   = 8.5 * samplingFreq;
    startIndex2 = 2638;
    endIndex2   = 3038;

    ikData = load("AB08\ik\levelground_ccw_fast_01_01.mat");
    ikData = ikData.data;
elseif gait_speed == 2
    startIndex1 = 1250;
    endIndex1   = 1926;
    startIndex2 = 3238;
    endIndex2   = 3900;

    ikData = load("AB08\ik\levelground_ccw_normal_01_01.mat");
    ikData = ikData.data;
elseif gait_speed == 3
    startIndex1 = 1100;
    endIndex1   = 1718;
    startIndex2 = 4574;
    endIndex2   = 5200;

    ikData = load("AB08\ik\levelground_ccw_slow_01_01.mat");
    ikData = ikData.data;
else
    error('Invalid gait speed. Please set to 1 (fast), 2 (normal), or 3 (slow).');
end

% create a single index vector by stitching the two intervals
idx = [round(startIndex1):round(endIndex1), startIndex2:endIndex2];
nSamples = numel(idx);

lowPass = true;
cutoff_freq = 50;  % Hz
saveVideo = false;
generateVideo = false;  % Set to false to skip video generation

%% extract COM/Pelvis position data
pelvis_pos_data = zeros(nSamples, 3);

% Extract original pelvis coordinates from dataset
pelvis_tx = ikData{idx, 'pelvis_tx'};  % forward direction
pelvis_ty = ikData{idx, 'pelvis_ty'};  % height
pelvis_tz = ikData{idx, 'pelvis_tz'};  % lateral direction

% Apply coordinate transformation: x = -z, y = -x, z = y
pelvis_pos_data(:, 1) = -pelvis_tz;
pelvis_pos_data(:, 2) = -pelvis_tx;
pelvis_pos_data(:, 3) = pelvis_ty;

% Make first position zero (relative coordinates)
pelvis_pos_data(:, 1) = pelvis_pos_data(:, 1) - pelvis_pos_data(1, 1);
pelvis_pos_data(:, 2) = pelvis_pos_data(:, 2) - pelvis_pos_data(1, 2);
% keep height as is (relative to ground)

% Handle stitching: flip sign of x in second half and remove discontinuity
switchIndex = round(endIndex1 - startIndex1) + 2;  % index where intervals join
fprintf('Switch index, %.2f\n', switchIndex);

pelvis_pos_data(switchIndex:end, 1) = -pelvis_pos_data(switchIndex:end, 1);
for dim = 1:3
    discontinuity = pelvis_pos_data(switchIndex, dim) - pelvis_pos_data(switchIndex-1, dim);
    fprintf('Dimension %d: Discontinuity = %.4f\n', dim, discontinuity);
    pelvis_pos_data(switchIndex:end, dim) = pelvis_pos_data(switchIndex:end, dim) - discontinuity;
end

%% visualize COM trajectory as video
if generateVideo
    figure('Position', [500, 100, 1000, 800], 'Name', 'COM Trajectory Video', 'NumberTitle', 'off');
    hold on;
    grid on;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('3D Trajectory of Center of Mass (COM)');
    view(45, 30);
    axis equal;

    % Set axis limits
    xRange = [min(pelvis_pos_data(:, 1)), max(pelvis_pos_data(:, 1))];
    yRange = [min(pelvis_pos_data(:, 2)), max(pelvis_pos_data(:, 2))];
    zRange = [min(pelvis_pos_data(:, 3)), max(pelvis_pos_data(:, 3))];

    % Add some padding
    padding = 0.05;
    xRange = xRange + padding * [-1, 1];
    yRange = yRange + padding * [-1, 1];
    zRange = zRange + padding * [-1, 1];

    xlim(xRange);
    ylim(yRange);
    zlim(zRange);

    % Plot complete trajectory in light gray (reference)
    plot3(pelvis_pos_data(:, 1), pelvis_pos_data(:, 2), pelvis_pos_data(:, 3), ...
        'Color', [0.8 0.8 0.8], 'LineWidth', 1.5, 'DisplayName', 'Full Trajectory', 'LineStyle', '--');

    % Create handles for animated elements
    hCOM_point = plot3(pelvis_pos_data(1, 1), pelvis_pos_data(1, 2), pelvis_pos_data(1, 3), ...
        'ro', 'MarkerSize', 10, 'DisplayName', 'Current Position');
    hTrail = animatedline('Color', 'r', 'LineWidth', 2, 'DisplayName', 'Trail');
    hTime = text(xRange(1) + 0.01, yRange(2) - 0.01, zRange(2) - 0.01, '', ...
        'FontSize', 12, 'Color', 'k', 'VerticalAlignment', 'top');

    legend('Location', 'best');

    % Set up video writer
    videoFileName = fullfile(output_folder, ['COM_trajectory_motion_' gait_speeds_names{gait_speed} '.mp4']);

    if saveVideo
        v = VideoWriter(videoFileName, 'MPEG-4');
        v.FrameRate = 60;
        open(v);
    end

    % Animate through all frames
    framesToRecord = 1:dsRatio:nSamples;  % Match downsampling rate
    timeArray = framesToRecord * dt_orig;

    for frameIdx = 1:length(framesToRecord)
        idx_frame = framesToRecord(frameIdx);
        
        % Update current position
        set(hCOM_point, 'XData', pelvis_pos_data(idx_frame, 1), ...
                        'YData', pelvis_pos_data(idx_frame, 2), ...
                        'ZData', pelvis_pos_data(idx_frame, 3));
        
        % Add to trail
        addpoints(hTrail, pelvis_pos_data(idx_frame, 1), ...
                        pelvis_pos_data(idx_frame, 2), ...
                        pelvis_pos_data(idx_frame, 3));
        
        % Update time display
        currentTime = timeArray(frameIdx);
        set(hTime, 'String', sprintf('Time: %.2f s', currentTime));
        
        drawnow;
        
        if saveVideo
            frame = getframe(gcf);
            writeVideo(v, frame);
        end
    end

    if saveVideo
        close(v);
        disp(['Video saved as: ' videoFileName]);
    end
end

%% extract joint angle data
% Joint order: hip_adduction_r, hip_adduction_l, hip_flexion_r, hip_flexion_l, ...
%              knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l
joint_angles = zeros(nSamples, 8);

joint_angles(:, 1:2) = deg2rad(ikData{idx, {'hip_adduction_r', 'hip_adduction_l'}});
joint_angles(:, 3:4) = deg2rad(ikData{idx, {'hip_flexion_r', 'hip_flexion_l'}});
joint_angles(:, 5:6) = -deg2rad(ikData{idx, {'knee_angle_r', 'knee_angle_l'}});
joint_angles(:, 7:8) = deg2rad(ikData{idx, {'ankle_angle_r', 'ankle_angle_l'}});

% Keep pelvis tilt separate from joint/q_dq datasets
pelvis_tilt_angle = deg2rad(ikData{idx, 'pelvis_tilt'});

%% calculate velocities for joint angles
joint_velocities = zeros(nSamples, 8);
for i = 1:8
    joint_velocities(:, i) = [0; diff(joint_angles(:, i)) / dt_orig];
end

% Pelvis tilt velocity (separate variable)
pelvis_tilt_velocity = [0; diff(pelvis_tilt_angle) / dt_orig];

% Pelvis x-y velocity (forward and lateral motion)
pelvis_vel_xy = zeros(nSamples, 2);
pelvis_vel_xy(:, 1) = [0; diff(pelvis_pos_data(:, 1)) / dt_orig];  % x velocity
pelvis_vel_xy(:, 2) = [0; diff(pelvis_pos_data(:, 2)) / dt_orig];  % y velocity

%% apply low-pass filter to joint velocities
if lowPass
    [b, a] = butter(4, cutoff_freq / (samplingFreq / 2));  % 4th order
    for i = 1:8
        joint_velocities(:, i) = filtfilt(b, a, joint_velocities(:, i));
    end
    pelvis_tilt_velocity = filtfilt(b, a, pelvis_tilt_velocity);
    pelvis_vel_xy(:, 1) = filtfilt(b, a, pelvis_vel_xy(:, 1));
    pelvis_vel_xy(:, 2) = filtfilt(b, a, pelvis_vel_xy(:, 2));
end

%% downsample
downsample_idx = 1:dsRatio:nSamples;
nSamples_ds = length(downsample_idx);

joint_angles_ds = joint_angles(downsample_idx, :);
joint_velocities_ds = joint_velocities(downsample_idx, :);
pelvis_pos_ds = pelvis_pos_data(downsample_idx, :);
pelvis_tilt_ds = pelvis_tilt_angle(downsample_idx, :);
pelvis_tilt_velocity_ds = pelvis_tilt_velocity(downsample_idx, :);
pelvis_vel_xy_ds = pelvis_vel_xy(downsample_idx, :);

%% ensure 12-second dataset length
target_duration = 12;  % seconds
target_samples = round(target_duration / dt_ds);

if nSamples_ds < target_samples
    % Pad with last values
    padding_size = target_samples - nSamples_ds;
    
    joint_angles_ds = [joint_angles_ds; repmat(joint_angles_ds(end, :), padding_size, 1)];
    joint_velocities_ds = [joint_velocities_ds; zeros(padding_size, 8)];
    pelvis_pos_ds = [pelvis_pos_ds; repmat(pelvis_pos_ds(end, :), padding_size, 1)];
    pelvis_tilt_ds = [pelvis_tilt_ds; repmat(pelvis_tilt_ds(end, :), padding_size, 1)];
    pelvis_tilt_velocity_ds = [pelvis_tilt_velocity_ds; zeros(padding_size, 1)];
    pelvis_vel_xy_ds = [pelvis_vel_xy_ds; zeros(padding_size, 2)];
    
    nSamples_ds = target_samples;
    fprintf('Padded dataset from %.1f s to %.1f s\n', (nSamples_ds - padding_size) * dt_ds, nSamples_ds * dt_ds);
end

% combine into final matrices
q_dq_data = [joint_angles_ds, joint_velocities_ds];  % 16 columns (no pelvis tilt)
pelvis_data = pelvis_pos_ds;                          % 3 columns
pelvis_tilt_data = [pelvis_tilt_ds, pelvis_tilt_velocity_ds];  % 2 columns: [tilt, tilt_velocity]

% Calculate the stitch point in downsampled data
stitch_idx_ds = round(switchIndex / dsRatio);

%% plot final downsampled data
time_ds = (0:nSamples_ds-1) * dt_ds;

% Joint Angles Figure
figure;
subplot(2, 2, 1);
plot(time_ds, rad2deg(q_dq_data(:, [1, 2])));
title('Hip Adduction');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 2);
plot(time_ds, rad2deg(q_dq_data(:, [3, 4])));
title('Hip Flexion/Extension');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 3);
plot(time_ds, rad2deg(q_dq_data(:, [5, 6])));
title('Knee Flexion/Extension');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 4);
plot(time_ds, rad2deg(q_dq_data(:, [7, 8])));
title('Ankle Flexion/Extension');
xlabel('Time (s)');
ylabel('Angle (degrees)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

sgtitle('Joint Angles');

saveas(gcf, fullfile(output_folder, ['joint_angles_plot_' gait_speeds_names{gait_speed} '.png']));

% Joint Velocities Figure
figure;
subplot(2, 2, 1);
plot(time_ds, q_dq_data(:, [9, 10]));
title('Hip Adduction Velocity');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 2);
plot(time_ds, q_dq_data(:, [11, 12]));
title('Hip Flexion/Extension Velocity');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 3);
plot(time_ds, q_dq_data(:, [13, 14]));
title('Knee Flexion/Extension Velocity');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 2, 4);
plot(time_ds, q_dq_data(:, [15, 16]));
title('Ankle Flexion/Extension Velocity');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
legend('Right', 'Left');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

sgtitle('Joint Velocities');

saveas(gcf, fullfile(output_folder, ['joint_velocities_plot_' gait_speeds_names{gait_speed} '.png']));

% Pelvis tilt (separate from joint/q_dq)
figure;
subplot(2, 1, 1);
plot(time_ds, rad2deg(pelvis_tilt_data(:, 1)));
title('Pelvis Tilt');
xlabel('Time (s)');
ylabel('Angle (degrees)');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 1, 2);
plot(time_ds, pelvis_tilt_data(:, 2));
title('Pelvis Tilt Velocity');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

sgtitle('Pelvis Tilt');
saveas(gcf, fullfile(output_folder, ['pelvis_tilt_plot_' gait_speeds_names{gait_speed} '.png']));

% Pelvis position
figure;
subplot(2, 1, 1);
plot(time_ds, pelvis_data(:, 1:3));
title('Pelvis Position');
xlabel('Time (s)');
ylabel('Position (m)');
legend('X', 'Y', 'Z');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

subplot(2, 1, 2);
plot(time_ds, pelvis_vel_xy_ds(:, 1:2));
title('Pelvis X-Y Velocity');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('X velocity', 'Y velocity');
grid on;
xline(time_ds(stitch_idx_ds), '--k', 'LineWidth', 1);

saveas(gcf, fullfile(output_folder, ['pelvis_position_plot_' gait_speeds_names{gait_speed} '.png']));

%% save data to NPZ file (NumPy)
npz_filename = fullfile(output_folder, ['motion_data_' gait_speeds_names{gait_speed} '.npz']);

try
    np = py.importlib.import_module('numpy');
    np.savez(npz_filename, pyargs( ...
        'q_dq_data', np.array(q_dq_data), ...
        'pelvis_data', np.array(pelvis_data), ...
        'pelvis_vel_xy', np.array(pelvis_vel_xy_ds), ...
        'pelvis_tilt_data', np.array(pelvis_tilt_data), ...
        'time_ds', np.array(time_ds), ...
        'dt_ds', dt_ds));

    disp(['Data saved to: ' npz_filename]);
catch ME
    warning('NPZ export failed (Python/NumPy not available): %s', ME.message);
end