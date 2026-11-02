%% SEA Actuator Dynamics - Discretized Torque Transfer Model
% Based on: Rampeltshammer et al. (2020), IEEE/ASME Trans. Mechatronics

clear; clc;

%% SEA Parameters
Ts = 1e-3; % Sample time [s]

% Open log file for output
log_file = fullfile(fileparts(mfilename('fullpath')), 'sea_models_log.txt');
diary(log_file);

% NOTE: load_joint_parameters.m (actuator parameters of the WE2 exoskeleton)
% is proprietary and not included in this public repository.
load_joint_parameters;

% All joints
all_joints = {'LHA', 'LHF', 'LK', 'LA', 'RHA', 'RHF', 'RK', 'RA'};

% Joint groups: knee vs other legs
knee_joints = {'LK', 'RK'};
leg_joints = {'LHA', 'LHF', 'LA', 'RHA', 'RHF', 'RA'};
representative_joints = {'LK', 'LHA'};  % One from each group

% Extract mechanical parameters for all joints (for reference table)
for i = 1:length(all_joints)
    j = all_joints{i};
    jp = Joint_params.(j);
    
    MechParams.(j).jm = jp.DOBController.SEA_inertia;
    MechParams.(j).bm = jp.DOBController.SEA_damper;
    MechParams.(j).k = jp.SpringStiffness_Nm_rad;
end

%% Print parameters for verification
fprintf('=== Mechanical Parameters for All Joints ===\n');
data = cellfun(@(j) [MechParams.(j).jm, ...
                     MechParams.(j).bm, ...
                     MechParams.(j).k], all_joints, 'UniformOutput', false);

T = array2table(vertcat(data{:}), ...
    'VariableNames', {'jm', 'bm', 'k'}, ...
    'RowNames', all_joints);

disp(T);
fprintf('\nNote: Joints grouped into 2 models:\n');
fprintf('  Knee Model (LK, RK):  Same parameters\n');
fprintf('  Leg Model (LHA, LHF, LA, RHA, RHF, RA): Same parameters\n\n');

clear data T

%% Load-side velocity input (omega_l)
% In simulation/hardware: replace this with your measured signal.
% Here it is defined as a symbolic/placeholder signal for clarity.
% omega_l is treated as an EXTERNAL INPUT to the system.
% The full spring torque output is:
%
%   tau_k = H(s) * tau_m  +  (-Z(s)) * omega_l
%
% Both transfer functions share the same denominator.

%% Initialize models structure
models = struct();
s = tf('s');

%% Loop through representative joints (one from each group) and calculate transfer functions and state-space models
fprintf('%s\n', repmat('=', 1, 80)); 
fprintf('Computing SEA Models for %d Representative Joints\n', length(representative_joints));
fprintf('%s\n', repmat('=', 1, 80));

for i = 1:length(representative_joints)
    j = representative_joints{i};
    jm = MechParams.(j).jm;
    bm = MechParams.(j).bm;
    k = MechParams.(j).k;
    
    fprintf('\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('Joint: %s (Representative for group)\n', j);
    fprintf('%s\n', repmat('=', 1, 80));
    
    % Continuous-time Transfer Functions
    denom = jm*s^2 + bm*s + k;
    
    % Torque transfer: tau_k / tau_m (eq. 2, generalized from eq. 1)
    H_s = k / denom;
    
    % Load disturbance (impedance): tau_k / (-omega_l) (eq. 3)
    % Represents how load-side motion couples into spring torque
    Z_s = k * (jm*s + bm) / denom;
    
    % fprintf('\nContinuous-time Transfer Functions:\n');
    % fprintf('H(s) — Torque transfer (tau_m -> tau_k):\n');
    % disp(H_s);
    % fprintf('Z(s) — Actuator impedance (-omega_l -> tau_k):\n');
    % disp(Z_s);
    
    % Discretization (Tustin method)
    H_d = c2d(H_s, Ts, 'tustin');
    Z_d = c2d(Z_s, Ts, 'tustin');
    
    % fprintf('Discretized Transfer Functions (Ts = %g s):\n', Ts);
    % fprintf('H_d(z) — Torque transfer:\n');
    % disp(H_d);
    % fprintf('Z_d(z) — Actuator impedance:\n');
    % disp(Z_d);
    
    % State-Space Representations
    sys_H = ss(H_d);
    sys_Z = ss(Z_d);
    
    % Determine the prefix based on whether it's a knee or leg joint
    if ismember(j, knee_joints)
        prefix = 'sea_knee';
    else
        prefix = 'sea_leg';
    end
    
    % Output H matrices in Python format
    fprintf('%s_H_A = %s\n', prefix, format_matrix_for_python(sys_H.A));
    fprintf('%s_H_B = %s\n', prefix, format_matrix_for_python(sys_H.B));
    fprintf('%s_H_C = %s\n', prefix, format_matrix_for_python(sys_H.C));
    fprintf('%s_H_D = %s\n\n', prefix, format_matrix_for_python(sys_H.D));
    
    % Output Z matrices in Python format
    fprintf('%s_Z_A = %s\n', prefix, format_matrix_for_python(sys_Z.A));
    fprintf('%s_Z_B = %s\n', prefix, format_matrix_for_python(sys_Z.B));
    fprintf('%s_Z_C = %s\n', prefix, format_matrix_for_python(sys_Z.C));
    fprintf('%s_Z_D = %s\n\n', prefix, format_matrix_for_python(sys_Z.D));
    
    % Store in models structure
    models.(j).H = H_d;
    models.(j).Z = Z_d;
    models.(j).sys_H = sys_H;
    models.(j).sys_Z = sys_Z;
    
    % Store state-space matrices separately for Python import
    models.(j).H_A = sys_H.A;
    models.(j).H_B = sys_H.B;
    models.(j).H_C = sys_H.C;
    models.(j).H_D = sys_H.D;
    models.(j).Z_A = sys_Z.A;
    models.(j).Z_B = sys_Z.B;
    models.(j).Z_C = sys_Z.C;
    models.(j).Z_D = sys_Z.D;
    models.(j).jm = jm;
    models.(j).bm = bm;
    models.(j).k = k;
end

%% Create group mapping for later access
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('Joint Group Mapping:\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('Knee joints (LK, RK) → Use model ''%s''\n', representative_joints{1});
fprintf('Leg joints (LHA, LHF, LA, RHA, RHF, RA) → Use model ''%s''\n', representative_joints{2});

% Create mapping structure
group_mapping = struct();
group_mapping.knee = representative_joints{1};
group_mapping.leg = representative_joints{2};

%% Save models to MAT file for Python import
output_file = fullfile(fileparts(mfilename('fullpath')), 'sea_models.mat');
save(output_file, 'models', 'group_mapping', 'Ts', 'representative_joints', 'knee_joints', 'leg_joints');
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('Models saved to: %s\n', output_file);
fprintf('%s\n', repmat('=', 1, 80));

%% Usage in RL loop:
% In your RL loop, at each timestep k:
%
%   x_H[k+1] = A_H * x_H[k] + B_H * tau_m[k]
%   x_Z[k+1] = A_Z * x_Z[k] + B_Z * omega_l[k]
%   tau_k[k]  = C_H * x_H[k] + D_H * tau_m[k] - (C_Z * x_Z[k] + D_Z * omega_l[k])
%
% omega_l is your measured/simulated load-side velocity.
% Access models via: models.LK.H_A, models.LHA.H_B, etc.
% ==========================================================================

%% Example: Simulate dynamics for a specific joint
% Uncomment to simulate. Use representative joints from: {'LK' (knee), 'LHA' (leg)}
% Or use any joint and it will work as long as it's in the same group.

simulate_joint_dynamics('LK', models, Ts);
% simulate_joint_dynamics('LHA', models, Ts, 'sim_duration', 15, 'sine_amp', 1.5, 'sine_freq', 1.0);

%% Close logging
diary off;
fprintf('\nOutput saved to: %s\n', log_file);

%% Helper function
function str = format_matrix_for_python(matrix)
    % Format a matrix as a Python list string for easy copy-pasting
    [rows, cols] = size(matrix);
    str = '[';
    
    for r = 1:rows
        str = [str, sprintf('\n        [')];
        for c = 1:cols
            if c > 1
                str = [str, ', '];
            end
            % Use %.16g for maximum precision (double precision)
            str = [str, sprintf('%.16g', matrix(r, c))];
        end
        str = [str, '],']; 
    end
    str = [str, sprintf('\n    ]')];
end