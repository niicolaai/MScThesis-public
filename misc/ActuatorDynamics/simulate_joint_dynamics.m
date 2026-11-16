function simulate_joint_dynamics(joint_name, models, Ts, varargin)
% SIMULATE_JOINT_DYNAMICS Simulate and plot SEA dynamics for a single joint
%
% Syntax:
%   simulate_joint_dynamics(joint_name, models, Ts)
%   simulate_joint_dynamics(joint_name, models, Ts, 'sim_duration', 5, 'sine_amp', 1, 'sine_freq', 0.5)
%
% Inputs:
%   joint_name (string)  - Joint identifier (e.g., 'LHA')
%   models (struct)      - Structure containing H_d and Z_d for each joint
%   Ts (double)          - Sample time [s]
%   Optional name-value pairs:
%       'sim_duration'   - Simulation duration (default: 5 s)
%       'sine_amp'       - Sinusoidal load velocity amplitude (default: 1 rad/s)
%       'sine_freq'      - Sinusoidal load velocity frequency (default: 0.5 Hz)

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'sim_duration', 5, @isnumeric);
    addParameter(p, 'sine_amp', 1, @isnumeric);
    addParameter(p, 'sine_freq', 0.5, @isnumeric);
    parse(p, varargin{:});
    
    sim_duration = p.Results.sim_duration;
    sine_amp = p.Results.sine_amp;
    sine_freq = p.Results.sine_freq;
    
    % Extract models for this joint
    if ~isfield(models, joint_name)
        error('Joint ''%s'' not found in models structure.', joint_name);
    end
    
    H_d = models.(joint_name).H;
    Z_d = models.(joint_name).Z;
    
    % Extract parameters to create continuous-time model
    jm = models.(joint_name).jm;
    bm = models.(joint_name).bm;
    k = models.(joint_name).k;
    
    % Create continuous-time transfer functions
    s = tf('s');
    denom = jm*s^2 + bm*s + k;
    H_c = k / denom;
    Z_c = k * (jm*s + bm) / denom;
    
    % Time vector
    t = 0:Ts:sim_duration;
    
    % --- Scenario 1: Zero load velocity ---
    tau_m_step = ones(size(t));  % Step input: 1 Nm at t=0
    tau_m_step(1) = 0;
    omega_l_zero = zeros(size(t));
    
    % --- Scenario 2: Sinusoidal load velocity ---
    omega_l_sine = sine_amp * sin(2*pi*sine_freq*t);
    
    % Plot results - CONTINUOUS SYSTEM
    figure('Position', [100 100 1200 800]);
    sgtitle(sprintf('SEA Actuator Dynamics - CONTINUOUS: Joint %s', joint_name), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    % Simulate continuous system
    tau_k_H_c = lsim(H_c, tau_m_step, t);
    tau_k_Z_zero_c = lsim(Z_c, omega_l_zero, t);
    tau_k_zero_load_c = tau_k_H_c - tau_k_Z_zero_c;
    
    tau_k_H_sine_c = lsim(H_c, tau_m_step, t);
    tau_k_Z_sine_c = lsim(Z_c, omega_l_sine, t);
    tau_k_sine_load_c = tau_k_H_sine_c - tau_k_Z_sine_c;
    
    % Scenario 1: Zero load velocity
    subplot(3,2,1);
    plot(t, tau_m_step, 'b', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Torque [Nm]');
    title('Scenario 1: Motor Torque Input (Step)');
    grid on;
    
    subplot(3,2,3);
    plot(t, omega_l_zero, 'r', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Angular Velocity [rad/s]');
    title('Scenario 1: Load-side Velocity (Zero)');
    grid on;
    
    subplot(3,2,5);
    plot(t, tau_k_zero_load_c, 'g', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Spring Torque [Nm]');
    title('Scenario 1: Spring Torque Output (Zero Load Velocity)');
    grid on;
    
    % Scenario 2: Sinusoidal load velocity
    subplot(3,2,2);
    plot(t, tau_m_step, 'b', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Torque [Nm]');
    title('Scenario 2: Motor Torque Input (Step)');
    grid on;
    
    subplot(3,2,4);
    plot(t, omega_l_sine, 'r', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Angular Velocity [rad/s]');
    title(sprintf('Scenario 2: Load-side Velocity (%.1f Hz Sine, amp=%.1f rad/s)', ...
                  sine_freq, sine_amp));
    grid on;
    
    subplot(3,2,6);
    plot(t, tau_k_sine_load_c, 'g', 'LineWidth', 2);
    hold on;
    plot(t, tau_k_zero_load_c, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Zero load velocity');
    xlabel('Time [s]'); ylabel('Spring Torque [Nm]');
    title('Scenario 2: Spring Torque Output (Sine Load Velocity)');
    legend('With sine load velocity', 'Reference (zero load)');
    grid on;
    
    % Plot results - DISCRETIZED SYSTEM
    figure('Position', [100 100 1200 800]);
    sgtitle(sprintf('SEA Actuator Dynamics - DISCRETIZED (Ts=%.2e s): Joint %s', Ts, joint_name), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    % Simulate discretized system
    tau_k_H_d = lsim(H_d, tau_m_step, t);
    tau_k_Z_zero_d = lsim(Z_d, omega_l_zero, t);
    tau_k_zero_load_d = tau_k_H_d - tau_k_Z_zero_d;
    
    tau_k_H_sine_d = lsim(H_d, tau_m_step, t);
    tau_k_Z_sine_d = lsim(Z_d, omega_l_sine, t);
    tau_k_sine_load_d = tau_k_H_sine_d - tau_k_Z_sine_d;
    
    % Scenario 1: Zero load velocity
    subplot(3,2,1);
    plot(t, tau_m_step, 'b', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Torque [Nm]');
    title('Scenario 1: Motor Torque Input (Step)');
    grid on;
    
    subplot(3,2,3);
    plot(t, omega_l_zero, 'r', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Angular Velocity [rad/s]');
    title('Scenario 1: Load-side Velocity (Zero)');
    grid on;
    
    subplot(3,2,5);
    plot(t, tau_k_zero_load_d, 'g', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Spring Torque [Nm]');
    title('Scenario 1: Spring Torque Output (Zero Load Velocity)');
    grid on;
    
    % Scenario 2: Sinusoidal load velocity
    subplot(3,2,2);
    plot(t, tau_m_step, 'b', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Torque [Nm]');
    title('Scenario 2: Motor Torque Input (Step)');
    grid on;
    
    subplot(3,2,4);
    plot(t, omega_l_sine, 'r', 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Angular Velocity [rad/s]');
    title(sprintf('Scenario 2: Load-side Velocity (%.1f Hz Sine, amp=%.1f rad/s)', ...
                  sine_freq, sine_amp));
    grid on;
    
    subplot(3,2,6);
    plot(t, tau_k_sine_load_d, 'g', 'LineWidth', 2);
    hold on;
    plot(t, tau_k_zero_load_d, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Zero load velocity');
    xlabel('Time [s]'); ylabel('Spring Torque [Nm]');
    title('Scenario 2: Spring Torque Output (Sine Load Velocity)');
    legend('With sine load velocity', 'Reference (zero load)');
    grid on;
    
    % Print summary
    fprintf('\n=== Simulation Results for Joint: %s ===\n', joint_name);
end
