clear; clc;

% Define filepaths
filepaths = {
    'AB08\ik\levelground_ccw_normal_01_01.mat', ...
    'AB08\ik\levelground_ccw_slow_01_01.mat', ...
    'AB08\ik\levelground_ccw_fast_01_01.mat'
};

titles = {
    'Normal Speed', ...
    'Slow Speed', ...
    'Fast Speed'
};

% Create subfolder for saving
output_folder = 'to_be_sliced';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Create main figure with tabbed interface
main_fig = figure('Name', 'Motion Data Comparison', 'NumberTitle', 'off');
set(main_fig, 'Position', [100, 100, 1200, 800]);
tg = uitabgroup(main_fig);

% Process each file
for file_idx = 1:length(filepaths)
    filepath = filepaths{file_idx};
    
    % Load data
    ikData = load(filepath);
    ikData = ikData.data;
    
    % Get filename without extension for title
    [~, filename, ~] = fileparts(filepath);
    title_str = titles{file_idx};
    
    % Get number of samples
    n_samples = height(ikData);
    
    % Extract joint angle data
    % Joint order: hip_adduction_r, hip_adduction_l, hip_flexion_r, hip_flexion_l, ...
    %              knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l
    joint_angles = zeros(n_samples, 8);
    
    joint_angles(:, 1:2) = deg2rad(table2array(ikData(:, {'hip_adduction_r', 'hip_adduction_l'})));
    joint_angles(:, 3:4) = deg2rad(table2array(ikData(:, {'hip_flexion_r', 'hip_flexion_l'})));
    joint_angles(:, 5:6) = -deg2rad(table2array(ikData(:, {'knee_angle_r', 'knee_angle_l'})));
    joint_angles(:, 7:8) = deg2rad(table2array(ikData(:, {'ankle_angle_r', 'ankle_angle_l'})));
    
    % Extract pelvis position data
    pelvis_pos_data = zeros(n_samples, 3);
    
    % Extract original pelvis coordinates from dataset
    pelvis_tx = table2array(ikData(:, 'pelvis_tx'));  % forward direction
    pelvis_ty = table2array(ikData(:, 'pelvis_ty'));  % height
    pelvis_tz = table2array(ikData(:, 'pelvis_tz'));  % lateral direction
    
    % Apply coordinate transformation: x = -z, y = -x, z = y
    pelvis_pos_data(:, 1) = -pelvis_tz;
    pelvis_pos_data(:, 2) = -pelvis_tx;
    pelvis_pos_data(:, 3) = pelvis_ty;
    
    % Make first position zero (relative coordinates)
    % pelvis_pos_data(:, 1) = pelvis_pos_data(:, 1) - pelvis_pos_data(1, 1);
    % pelvis_pos_data(:, 2) = pelvis_pos_data(:, 2) - pelvis_pos_data(1, 2);
    
    % Create x-axis: datapoints/indices
    datapoints = 1:n_samples;
    
    % Initialize best matches
    best_match_indices = [];
    best_match_differences = [];
    manual_marker_indices = [];
    
    % Match joint positions for first two files
    if file_idx == 1
        % First file: extract at 1926, search around 3200 ±200
        reference_idx = 1926;
        reference_joints = joint_angles(reference_idx, :);
        search_region = max(1, 3200-200):min(n_samples, 3200+200);
        manual_marker_indices = [1250, 1926, 3900];
        
        fprintf('\n--- File 1: Normal Speed ---\n');
        fprintf('Reference joint positions at datapoint %d\n', reference_idx);
        
        % Calculate quadratic differences for all points in search region
        diffs = [];
        for search_idx = search_region
            quad_diff = sum((joint_angles(search_idx, :) - reference_joints).^2);
            diffs = [diffs; search_idx, quad_diff];
        end
        
        % Sort by difference and get top 3
        [~, sort_idx] = sort(diffs(:, 2));
        best_match_indices = diffs(sort_idx(1:3), 1);
        best_match_differences = diffs(sort_idx(1:3), 2);
        
        fprintf('Best 3 matches in region [3000-3400]:\n');
        for i = 1:3
            fprintf('  %d: Index %d, Quadratic Diff = %.6f\n', i, best_match_indices(i), best_match_differences(i));
        end
        fprintf('Marked indices: %d, %d, %d\n', manual_marker_indices);
        
    elseif file_idx == 2
        % Second file: extract at 1718, search around 4600 ±200
        reference_idx = 1718;
        reference_joints = joint_angles(reference_idx, :);
        search_region = max(1, 4600-200):min(n_samples, 4600+200);
        manual_marker_indices = [1100, 1718, 5200];
        
        fprintf('\n--- File 2: Slow Speed ---\n');
        fprintf('Reference joint positions at datapoint %d\n', reference_idx);
        
        % Calculate quadratic differences for all points in search region
        diffs = [];
        for search_idx = search_region
            quad_diff = sum((joint_angles(search_idx, :) - reference_joints).^2);
            diffs = [diffs; search_idx, quad_diff];
        end
        
        % Sort by difference and get top 3
        [~, sort_idx] = sort(diffs(:, 2));
        best_match_indices = diffs(sort_idx(1:3), 1);
        best_match_differences = diffs(sort_idx(1:3), 2);
        
        fprintf('Best 3 matches in region [4400-4800]:\n');
        for i = 1:3
            fprintf('  %d: Index %d, Quadratic Diff = %.6f\n', i, best_match_indices(i), best_match_differences(i));
        end
        fprintf('Marked indices: %d, %d, %d\n', manual_marker_indices);
    end
    
    %% Create tab for Joint Angles
    tab_angles = uitab(tg, 'Title', [title_str ' - Angles']);
    
    tcl = tiledlayout(tab_angles, 2, 2, 'Padding', 'compact');
    
    nexttile(tcl);
    ax1 = gca;
    plot(ax1, datapoints, rad2deg(joint_angles(:, [1, 2])));
    title(ax1, 'Hip Adduction');
    xlabel(ax1, 'Datapoint Index');
    ylabel(ax1, 'Angle (degrees)');
    legend(ax1, 'Right', 'Left');
    grid(ax1, 'on');
    
    nexttile(tcl);
    ax2 = gca;
    plot(ax2, datapoints, rad2deg(joint_angles(:, [3, 4])));
    title(ax2, 'Hip Flexion/Extension');
    xlabel(ax2, 'Datapoint Index');
    ylabel(ax2, 'Angle (degrees)');
    legend(ax2, 'Right', 'Left');
    grid(ax2, 'on');
    
    nexttile(tcl);
    ax3 = gca;
    plot(ax3, datapoints, rad2deg(joint_angles(:, [5, 6])));
    title(ax3, 'Knee Flexion/Extension');
    xlabel(ax3, 'Datapoint Index');
    ylabel(ax3, 'Angle (degrees)');
    legend(ax3, 'Right', 'Left');
    grid(ax3, 'on');
    
    nexttile(tcl);
    ax4 = gca;
    plot(ax4, datapoints, rad2deg(joint_angles(:, [7, 8])));
    title(ax4, 'Ankle Flexion/Extension');
    xlabel(ax4, 'Datapoint Index');
    ylabel(ax4, 'Angle (degrees)');
    legend(ax4, 'Right', 'Left');
    grid(ax4, 'on');
    
    % Add vertical lines to the last file's angle plots
    if file_idx == length(filepaths)
        marker_indices = [1100, 2638, 1700, 3038];
        for ax = [ax1, ax2, ax3, ax4]
            hold(ax, 'on');
            for idx = marker_indices
                if idx <= n_samples
                    xline(ax, idx, '--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'Alpha', 0.7);
                end
            end
            hold(ax, 'off');
        end
    elseif file_idx <= 2 && ~isempty(best_match_indices)
        % Add best matches and manual markers for files 1 and 2
        all_indices = [best_match_indices; manual_marker_indices'];
        for ax = [ax1, ax2, ax3, ax4]
            hold(ax, 'on');
            % Best match indices in red dotted
            for idx = best_match_indices'
                if idx <= n_samples
                    xline(ax, idx, ':', 'LineWidth', 2, 'Color', [0.8 0.2 0.2], 'Alpha', 0.8);
                end
            end
            % Manual marker indices in blue dashed
            for idx = manual_marker_indices
                if idx <= n_samples
                    xline(ax, idx, '--', 'LineWidth', 1.5, 'Color', [0.2 0.2 0.8], 'Alpha', 0.7);
                end
            end
            hold(ax, 'off');
        end
    end
    
    %% Create tab for Pelvis Positions
    tab_pelvis = uitab(tg, 'Title', [title_str ' - Pelvis']);
    ax_pelvis = axes('Parent', tab_pelvis);
    
    plot(ax_pelvis, datapoints, pelvis_pos_data(:, 1:3), 'LineWidth', 1.5);
    title(ax_pelvis, ['Pelvis Position - ' title_str]);
    xlabel(ax_pelvis, 'Datapoint Index');
    ylabel(ax_pelvis, 'Position (m)');
    legend(ax_pelvis, 'X', 'Y', 'Z');
    grid(ax_pelvis, 'on');
    
    % Add vertical lines to the last file's pelvis plot
    if file_idx == length(filepaths)
        marker_indices = [1100, 2638, 1700, 3038];
        hold(ax_pelvis, 'on');
        for idx = marker_indices
            if idx <= n_samples
                xline(ax_pelvis, idx, '--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'Alpha', 0.7);
            end
        end
        hold(ax_pelvis, 'off');
    elseif file_idx <= 2 && ~isempty(best_match_indices)
        % Add best matches and manual markers for files 1 and 2 to pelvis plot
        hold(ax_pelvis, 'on');
        % Best match indices in red dotted
        for idx = best_match_indices'
            if idx <= n_samples
                xline(ax_pelvis, idx, ':', 'LineWidth', 2, 'Color', [0.8 0.2 0.2], 'Alpha', 0.8);
            end
        end
        % Manual marker indices in blue dashed
        for idx = manual_marker_indices
            if idx <= n_samples
                xline(ax_pelvis, idx, '--', 'LineWidth', 1.5, 'Color', [0.2 0.2 0.8], 'Alpha', 0.7);
            end
        end
        hold(ax_pelvis, 'off');
    end
    
    % Save figures to file
    % output_filename_angles = fullfile(output_folder, ['joint_angles_' title_str '.png']);
    % exportapp(main_fig, output_filename_angles);
    % 
    % output_filename_pelvis = fullfile(output_folder, ['pelvis_position_' title_str '.png']);
    % exportapp(main_fig, output_filename_pelvis);
    
    fprintf('Processed: %s\n', filename);
end

disp('Done');
