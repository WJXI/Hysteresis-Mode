import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from scipy.linalg import expm
import os
import cmath
from pathlib import Path
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import eigs
import scipy.sparse.linalg as spla
import brian2 as b2
from sklearn.cluster import SpectralClustering
import matplotlib.patches as patches
from collections import deque
import pickle
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe




def run_simulation_core(
    run_id,
    neuron_ids, 
    connectivity_matrix, 
    sugar_ids, 
    threat_ids, 
    mn9_id,
    sugar_freq=100.0,      # Hz
    max_threat_freq=200.0, # Hz
    ramp_duration=5000.0,  # ms (Quasi-static sweep)
    hold_duration=250.0,   # ms
    dt_sim=0.1,            # ms
    input_gain=200.0       # Input gain
):
    """
    RUN SINGLE TRIAL
    """

    print(f"  > Simulating Trial {run_id+1}...", end="\r")
    
    b2.start_scope()
    b2.defaultclock.dt = dt_sim * b2.ms
    
    # --- A. parameters ---
    N = len(neuron_ids)
    id_map = {uid: i for i, uid in enumerate(neuron_ids)}
    
    # 
    tau_m = 20.0 * b2.ms
    tau_syn = 5.0 * b2.ms
    w_base = 0.275 * b2.mV 
    w_input_val = w_base * input_gain

    # 
    t1 = hold_duration
    t2 = t1 + ramp_duration
    t3 = t2 + hold_duration
    t4 = t3 + ramp_duration
    total_time = t4 + 100.0
    
    time_steps = np.arange(0, total_time, dt_sim)
    rate_values = np.zeros_like(time_steps)
    
    # Ramp Up (0 -> Max)
    mask_up = (time_steps >= t1) & (time_steps < t2)
    rate_values[mask_up] = max_threat_freq * (time_steps[mask_up] - t1) / ramp_duration
    # Hold Max
    mask_max = (time_steps >= t2) & (time_steps < t3)
    rate_values[mask_max] = max_threat_freq
    # Ramp Down (Max -> 0)
    mask_down = (time_steps >= t3) & (time_steps < t4)
    rate_values[mask_down] = max_threat_freq * (1.0 - (time_steps[mask_down] - t3) / ramp_duration)
    
    threat_rate_timed = b2.TimedArray(rate_values * b2.Hz, dt=dt_sim*b2.ms)
    
    # LIF Equation
    eqs = '''
    dv/dt = (-v + g) / tau_m : volt (unless refractory)
    dg/dt = -g / tau_syn : volt
    ref_P : second
    '''
    neurons = b2.NeuronGroup(N, eqs, threshold='v > 7.0*mV', reset='v = 0*mV', refractory='ref_P', method='exact')
    neurons.v = 0 * b2.mV
    neurons.g = 0 * b2.mV
    neurons.ref_P = 2.2 * b2.ms 
    

    # Sugar (Constant)
    sugar_indices = np.array([id_map[u] for u in sugar_ids if u in id_map], dtype=int)
    if len(sugar_indices) > 0:
        neurons.ref_P[sugar_indices] = 0 * b2.ms
        P_sugar = b2.PoissonGroup(len(sugar_indices), rates=sugar_freq*b2.Hz)
        S_sugar = b2.Synapses(P_sugar, neurons, on_pre='g += w_input_val')
        S_sugar.connect(i=np.arange(len(sugar_indices)), j=sugar_indices)

    # Threat (Dynamic)
    threat_indices = np.array([id_map[u] for u in threat_ids if u in id_map], dtype=int)
    if len(threat_indices) > 0:
        neurons.ref_P[threat_indices] = 0 * b2.ms
        P_threat = b2.PoissonGroup(len(threat_indices), rates='threat_rate_timed(t)')
        S_threat = b2.Synapses(P_threat, neurons, on_pre='g += w_input_val')
        S_threat.connect(i=np.arange(len(threat_indices)), j=threat_indices)
        
    # --- E. Internal Recurrent ---
    if sp.issparse(connectivity_matrix):
        coo = connectivity_matrix.tocoo()
        sources, targets, weights = coo.col, coo.row, coo.data
    else:
        sources, targets = np.where(connectivity_matrix != 0)
        weights = connectivity_matrix[sources, targets]
        
    syn = b2.Synapses(neurons, neurons, model='w_val : 1', on_pre='g += w_val * w_base', delay=1.8*b2.ms)
    syn.connect(i=sources, j=targets)
    syn.w_val = weights
    
    # --- F. record ---
    idx_mn9 = id_map[mn9_id]
    spikes = b2.SpikeMonitor(neurons)
    
    # namespace=locals() 
    b2.run(total_time * b2.ms, namespace=locals())
    
    # --- G. data analysis ---
    all_spikes = spikes.t / b2.ms
    mn9_spikes = all_spikes[spikes.i == idx_mn9]
    
    #  Rate Curve
    plot_dt = 20.0 
    common_time = np.arange(0, total_time, plot_dt)
    mn9_rate_curve = []
    window_ms = 150.0 
    
    for t in common_time:
        count = np.sum((mn9_spikes >= t - window_ms) & (mn9_spikes < t))
        mn9_rate_curve.append(count / (window_ms/1000.0))
        
    # rebuild Threat Curve 
    threat_curve = []
    for t in common_time:
        if t < t1: val = 0
        elif t < t2: val = max_threat_freq * (t - t1)/ramp_duration
        elif t < t3: val = max_threat_freq
        elif t < t4: val = max_threat_freq * (1.0 - (t - t3)/ramp_duration)
        else: val = 0
        threat_curve.append(val)
        
    return common_time, np.array(threat_curve), np.array(mn9_rate_curve), (t1, t2, t3, t4)




def batch_simulate_and_save(filename, neuron_ids, W, sugar_ids, threat_ids, mn9_id, n_trials=15):
    """
    Run N simulation，save to local .npz 
    """
    print(f"\n=== Starting Batch Simulation (N={n_trials}) ===")
    
    results_rate = []
    common_threat = None
    time_markers = None
    

    for i in range(n_trials):

        _, threat_axis, rate_trace, markers = run_simulation_core(
            i, neuron_ids, W, sugar_ids, threat_ids, mn9_id
        )
        results_rate.append(rate_trace)
        

        if common_threat is None:
            common_threat = threat_axis
            time_markers = markers
            

    rate_matrix = np.vstack(results_rate)
    

    print(f"\nSaving data to {filename}...")
    np.savez(filename, 
             rate_matrix=rate_matrix, 
             threat_axis=common_threat, 
             markers=time_markers)
    print("Done! Simulation data saved.")
# =============================================================================
# 2. plot (Physics Style Helper)
# =============================================================================
def add_bold_arrow(ax, x, y, target_x, direction='right', color='black'):

    idx = (np.abs(x - target_x)).argmin()
    px, py = x[idx], y[idx]
    arrow_span = 20.0 
    
    if direction == 'right':
        xy_tip, xy_tail = (px + arrow_span, py), (px, py)
    else:
        xy_tip, xy_tail = (px - arrow_span, py), (px, py)
        
    ax.annotate('', xy=xy_tip, xytext=xy_tail,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5, mutation_scale=25),
                zorder=10)

def plot_from_saved_data(filename):
    """
    read .npz and plot。
    """
    print(f"Loading data from {filename}...")
    data = np.load(filename)
    
    rate_matrix = data['rate_matrix']
    common_threat = data['threat_axis']
    t1, t2, t3, t4 = data['markers']
    
    N_TRIALS = rate_matrix.shape[0]
    print(f"Loaded {N_TRIALS} trials.")
    

    mean_rate = np.mean(rate_matrix, axis=0)
    std_rate = np.std(rate_matrix, axis=0)
    
    dt_plot = 20.0 
    idx_start_fwd = int(t1 / dt_plot)
    idx_end_fwd   = int(t2 / dt_plot)
    idx_start_bwd = int(t3 / dt_plot)
    idx_end_bwd   = int(t4 / dt_plot)
    
    # Forward Data
    threat_fwd = common_threat[idx_start_fwd : idx_end_fwd]
    mean_fwd   = mean_rate[idx_start_fwd : idx_end_fwd]
    std_fwd    = std_rate[idx_start_fwd : idx_end_fwd]
    
    # Backward Data
    threat_bwd = common_threat[idx_start_bwd : idx_end_bwd]
    mean_bwd   = mean_rate[idx_start_bwd : idx_end_bwd]
    std_bwd    = std_rate[idx_start_bwd : idx_end_bwd]
    
    # find representation
    recovery_thresholds = [] 
    for i in range(N_TRIALS):
        trace = rate_matrix[i][idx_start_bwd : idx_end_bwd]
        try:
            idx_rec = np.where(trace > 50.0)[0][0]
            th_val = threat_bwd[idx_rec]
            recovery_thresholds.append((i, th_val))
        except IndexError:
            recovery_thresholds.append((i, 0)) 
            
    recovery_thresholds.sort(key=lambda x: x[1])
    idx_late   = recovery_thresholds[0][0]
    idx_median = recovery_thresholds[len(recovery_thresholds)//2][0]
    idx_early  = recovery_thresholds[-1][0]
    
    # --- plot ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    

    color_fwd = '#D62728' 
    color_bwd = '#0055A4' 
    
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax1.fill_between(threat_bwd, mean_bwd - std_bwd, mean_bwd + std_bwd, color=color_bwd, alpha=0.15, ec='none', zorder=1)
    ax1.fill_between(threat_fwd, mean_fwd - std_fwd, mean_fwd + std_fwd, color=color_fwd, alpha=0.15, ec='none', zorder=2)
    ax1.plot(threat_bwd, mean_bwd, color=color_bwd, lw=3, label='Recovery (Backward)', zorder=3)
    ax1.plot(threat_fwd, mean_fwd, color=color_fwd, lw=3, label='Suppression (Forward)', zorder=4)
    
    add_bold_arrow(ax1, threat_fwd, mean_fwd, target_x=25, direction='right', color=color_fwd)
    add_bold_arrow(ax1, threat_bwd, mean_bwd, target_x=175, direction='left', color=color_bwd)
    

    baseline_high = np.max(mean_fwd)
    try: idx_off = np.where(mean_fwd < baseline_high * 0.5)[0][0]; th_off = threat_fwd[idx_off]
    except: th_off = 130 
    try: idx_on = np.where(mean_bwd > baseline_high * 0.5)[0][0]; th_on = threat_bwd[idx_on]
    except: th_on = 80 
    
    if th_off > th_on:
        ax1.axvspan(th_on, th_off, color='silver', alpha=0.2, zorder=0)
        mid_y = baseline_high * 0.45
        center_x = (th_on + th_off) / 2
        ax1.axvline(th_off, color=color_fwd, linestyle=':', alpha=0.6, lw=1.5)
        ax1.axvline(th_on, color=color_bwd, linestyle=':', alpha=0.6, lw=1.5)
        ax1.annotate('', xy=(th_on, mid_y), xytext=(th_off, mid_y), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax1.text(center_x, mid_y + 15, 'Hysteresis\nZone', ha='center', va='bottom', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax1.set_title("(A) Macroscopic Hysteresis Loop", fontsize=16, fontweight='bold', loc='left')
    ax1.set_xlabel("Threat Input (Hz)", fontsize=14)
    ax1.set_ylabel("Output MN9 Firing Rate (Hz)", fontsize=14)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-10, 460)
    ax1.legend(loc='lower left', frameon=True, fontsize=11, framealpha=0.9)
    

    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax2.plot(threat_bwd, mean_bwd, color='lightsteelblue', lw=4, alpha=0.5, label='Ensemble Average')
    ax2.fill_between(threat_bwd, mean_bwd - std_bwd, mean_bwd + std_bwd, color='lightsteelblue', alpha=0.15, ec='none')
    
    styles = [
        (idx_early,  'dashed',   2.0, 'navy',      'Early Recovery'),
        (idx_median, 'dashdot',  2.0, 'royalblue', 'Median Recovery'),
        (idx_late,   'dotted',   2.5, '#6495ED',   'Late Recovery') 
    ]
    
    for idx, ls, lw, col, lab in styles:
        trace = rate_matrix[idx][idx_start_bwd : idx_end_bwd]
        ax2.plot(threat_bwd, trace, linestyle=ls, color=col, linewidth=lw, label=lab,
                 path_effects=[pe.Stroke(linewidth=lw+1.5, foreground='white', alpha=0.7), pe.Normal()])
        
    ax2.set_title("(B) Stochastic Recovery Dynamics", fontsize=16, fontweight='bold', loc='left')
    ax2.set_xlabel("Threat Input (Hz)", fontsize=14)
    ax2.set_xlim(0, 160)
    ax2.set_ylim(-10, 460)
    

    ax2.text(130, 80, "Trapped in\nSilent State", color='gray', fontsize=11, style='italic',
             ha='center', va='bottom', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
             
    ax2.annotate("Abrupt Jump", xy=(th_on, 50), xytext=(th_on+35, 150),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=11, fontweight='bold')

    ax2.legend(loc='lower right', frameon=True, fontsize=10, framealpha=0.9)
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig('Figure1_Final_PhysicsStyle.pdf', format='pdf', bbox_inches='tight')
    plt.show()















def brian2_dynamic_hysteresis_complete(
    neuron_ids, 
    connectivity_matrix, 
    sugar_ids, 
    threat_ids, 
    mn9_id,
    sugar_freq=100.0,      
    max_threat_freq=200.0, 
    ramp_duration=5000.0,  
    hold_duration=250.0,   
    dt_sim=0.1,            
    input_gain=200.0       
):
    print(f"=== Brian2 Simulation: Dynamic Hysteresis ===")
    
    b2.start_scope()
    b2.defaultclock.dt = dt_sim * b2.ms
    
    N = len(neuron_ids)
    id_map = {uid: i for i, uid in enumerate(neuron_ids)}
    

    tau_m = 20.0 * b2.ms
    tau_syn = 5.0 * b2.ms
    v_th = 7.0 * b2.mV
    v_reset = 0.0 * b2.mV
    w_base = 0.275 * b2.mV 
    

    w_input_val = w_base * input_gain
    print(f"  > Input Gain: {input_gain}x (Amp ~ {w_input_val})")


    t1 = hold_duration
    t2 = t1 + ramp_duration
    t3 = t2 + hold_duration
    t4 = t3 + ramp_duration
    total_time = t4 + 100.0
    
    time_steps = np.arange(0, total_time, dt_sim)
    rate_values = np.zeros_like(time_steps)
    
    # Ramp Logic
    mask_up = (time_steps >= t1) & (time_steps < t2)
    rate_values[mask_up] = max_threat_freq * (time_steps[mask_up] - t1) / ramp_duration
    mask_max = (time_steps >= t2) & (time_steps < t3)
    rate_values[mask_max] = max_threat_freq
    mask_down = (time_steps >= t3) & (time_steps < t4)
    rate_values[mask_down] = max_threat_freq * (1.0 - (time_steps[mask_down] - t3) / ramp_duration)
    
    threat_rate_timed = b2.TimedArray(rate_values * b2.Hz, dt=dt_sim*b2.ms)
    

    eqs = '''
    dv/dt = (-v + g) / tau_m : volt (unless refractory)
    dg/dt = -g / tau_syn : volt
    ref_P : second
    '''
    
    neurons = b2.NeuronGroup(N, eqs, 
                             threshold='v > v_th', 
                             reset='v = v_reset', 
                             refractory='ref_P', 
                             method='exact')
    
    neurons.v = 0 * b2.mV
    neurons.g = 0 * b2.mV
    neurons.ref_P = 2.2 * b2.ms 
    

    
    # Sugar Input
    sugar_indices = np.array([id_map[u] for u in sugar_ids if u in id_map], dtype=int)
    if len(sugar_indices) > 0:
        neurons.ref_P[sugar_indices] = 0 * b2.ms
        P_sugar = b2.PoissonGroup(len(sugar_indices), rates=sugar_freq*b2.Hz)
        

        S_sugar = b2.Synapses(P_sugar, neurons, on_pre='g += w_input_val')
        S_sugar.connect(i=np.arange(len(sugar_indices)), j=sugar_indices)
        print(f"  > Connected {len(sugar_indices)} Sugar inputs.")

    # Threat Input
    threat_indices = np.array([id_map[u] for u in threat_ids if u in id_map], dtype=int)
    if len(threat_indices) > 0:
        neurons.ref_P[threat_indices] = 0 * b2.ms
        P_threat = b2.PoissonGroup(len(threat_indices), rates='threat_rate_timed(t)')
        

        S_threat = b2.Synapses(P_threat, neurons, on_pre='g += w_input_val')
        S_threat.connect(i=np.arange(len(threat_indices)), j=threat_indices)
        print(f"  > Connected {len(threat_indices)} Threat inputs.")
        
    # Recurrent 
    print("  > Building Recurrent Synapses...")
    if sp.issparse(connectivity_matrix):
        coo = connectivity_matrix.tocoo()
        sources = coo.col
        targets = coo.row
        weights = coo.data
    else:
        sources, targets = np.where(connectivity_matrix != 0)
        weights = connectivity_matrix[sources, targets]
        

    syn = b2.Synapses(neurons, neurons, model='w_val : 1', 
                      on_pre='g += w_val * w_base', 
                      delay=1.8*b2.ms)
    syn.connect(i=sources, j=targets)
    syn.w_val = weights
    

    if mn9_id not in id_map:
        print(f"Error: MN9 ID {mn9_id} not found.")
        return
    idx_mn9 = id_map[mn9_id]
    
    spikes = b2.SpikeMonitor(neurons)
    
    print(f"  > Running Simulation ({total_time} ms)...")
    

    b2.run(total_time * b2.ms, namespace=locals())
    
    # --- analysis ---
    print("  > Analyzing Results...")
    
    all_spikes_t = spikes.t / b2.ms
    all_spikes_i = spikes.i
    mn9_spike_times = all_spikes_t[all_spikes_i == idx_mn9]
    
    window_ms = 100.0
    plot_dt = 10.0
    plot_times = np.arange(0, total_time, plot_dt)
    mn9_rates = []
    
    for t in plot_times:
        t_start = max(0, t - window_ms)
        count = np.sum((mn9_spike_times >= t_start) & (mn9_spike_times < t))
        rate = count / (window_ms / 1000.0)
        mn9_rates.append(rate)
        
    threat_profile_x = []
    for t in plot_times:
        if t < t1: val = 0
        elif t < t2: val = max_threat_freq * (t - t1)/ramp_duration
        elif t < t3: val = max_threat_freq
        elif t < t4: val = max_threat_freq * (1.0 - (t - t3)/ramp_duration)
        else: val = 0
        threat_profile_x.append(val)
        
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(plot_times, threat_profile_x, 'k--', alpha=0.3, label='Threat Input')
    ax1.plot(plot_times, mn9_rates, 'g-', lw=2, label='MN9 Output')
    ax1.set_title("Time Course")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Hz")
    ax1.legend()
    
    ax2 = fig.add_subplot(1, 2, 2)
    mask_fwd = (plot_times >= t1) & (plot_times <= t2)
    mask_bwd = (plot_times >= t3) & (plot_times <= t4)
    
    if np.any(mask_fwd):
        ax2.plot(np.array(threat_profile_x)[mask_fwd], np.array(mn9_rates)[mask_fwd], 'r-', label='Forward')
    if np.any(mask_bwd):
        ax2.plot(np.array(threat_profile_x)[mask_bwd], np.array(mn9_rates)[mask_bwd], 'b-', label='Backward')
        
    ax2.set_title("Hysteresis Loop")
    ax2.set_xlabel("Threat (Hz)")
    ax2.set_ylabel("MN9 (Hz)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()










            





def run_flux_mapping_simulation_time_based(
    neuron_ids, W_matrix, sugar_ids, threat_ids, mn9_id,
    # 模拟参数
    sugar_freq=100.0, 
    max_threat_freq=200.0, 
    ramp_duration=5000.0, 
    hold_duration=500.0, 
    dt_sim=0.1, 
    input_gain=200.0,
    # 采样参数
    dt_window=100.0,
    # 【关键修改】直接使用时间区间 (ms)
    feed_time_range=(0, 1500),      # 默认：Sugar 预热阶段 (0-2000ms内)
    kill_time_range=(6500, 7500)    # 默认：Threat 高锋阶段 (2500+4000 ~ 2500+5000)
):
    """
    Time-Based Flux Mapping:
    直接指定时间窗口提取 F_feed 和 F_kill，不再依赖频率反推。
    这允许提取 Sugar Ramp 阶段 (Threat=0) 的数据。
    """
    print(f"=== Step 1: Running Full Simulation (Time-Based Selection) ===")
    
    # --- 1. 动态时间轴构建 ---
    duration_sugar_ramp = 2000.0 
    duration_hold = hold_duration
    duration_threat_ramp = ramp_duration
    
    # 关键时间点
    t1 = duration_sugar_ramp                  # 2000
    t2 = t1 + duration_hold                   # 2500 (Threat Start)
    t3 = t2 + duration_threat_ramp            # 7500 (Threat End)
    total_time = t3 + 100.0
    
    print(f"  > Timeline Reference:")
    print(f"    [0    - {t1:.0f} ms] : Sugar Ramp (0->100Hz), Threat=0")
    print(f"    [{t1:.0f} - {t2:.0f} ms] : Hold State (Sugar=100), Threat=0")
    print(f"    [{t2:.0f} - {t3:.0f} ms] : Threat Ramp (0->{max_threat_freq}Hz)")
    
    print(f"  > Selected Extraction Windows:")
    print(f"    F_feed: {feed_time_range[0]}-{feed_time_range[1]} ms")
    print(f"    F_kill: {kill_time_range[0]}-{kill_time_range[1]} ms")
    
    # 构建时间步
    time_steps = np.arange(0, total_time, dt_sim)
    sugar_vals = np.zeros_like(time_steps)
    threat_vals = np.zeros_like(time_steps)
    
    # A. Sugar Profile
    mask_sugar_ramp = time_steps < t1
    sugar_vals[mask_sugar_ramp] = sugar_freq * (time_steps[mask_sugar_ramp] / duration_sugar_ramp)
    sugar_vals[~mask_sugar_ramp] = sugar_freq
    
    # B. Threat Profile
    mask_threat_ramp = (time_steps >= t2) & (time_steps < t3)
    threat_vals[mask_threat_ramp] = max_threat_freq * ((time_steps[mask_threat_ramp] - t2) / duration_threat_ramp)
    threat_vals[time_steps >= t3] = max_threat_freq
    
    sugar_timed = b2.TimedArray(sugar_vals*b2.Hz, dt=dt_sim*b2.ms)
    threat_timed = b2.TimedArray(threat_vals*b2.Hz, dt=dt_sim*b2.ms)
    
    # --- 2. 显式网络构建 ---
    b2.start_scope()
    b2.defaultclock.dt = dt_sim * b2.ms
    net = b2.Network()
    
    N = len(neuron_ids)
    id_map = {uid: i for i, uid in enumerate(neuron_ids)}
    tau_m = 20*b2.ms; w_val = 0.275*b2.mV * input_gain
    
    eqs = '''
    dv/dt=(-v+g)/(20*ms):volt(unless refractory)
    dg/dt=-g/(5*ms):volt
    ref_P:second
    '''
    neurons = b2.NeuronGroup(N, eqs, threshold='v>7*mV', reset='v=0*mV', refractory='ref_P', method='exact')
    neurons.v=0; neurons.g=0; neurons.ref_P=2.2*b2.ms; net.add(neurons)
    
    s_idx = [id_map[u] for u in sugar_ids if u in id_map]
    if len(s_idx)>0: 
        neurons.ref_P[s_idx]=0; P_s=b2.PoissonGroup(len(s_idx), rates='sugar_timed(t)'); 
        S_s=b2.Synapses(P_s,neurons,on_pre='g+=w_val'); S_s.connect(i=np.arange(len(s_idx)), j=s_idx); net.add(P_s, S_s)
    
    t_idx = [id_map[u] for u in threat_ids if u in id_map]
    if len(t_idx)>0: 
        neurons.ref_P[t_idx]=0; P_t=b2.PoissonGroup(len(t_idx), rates='threat_timed(t)'); 
        S_t=b2.Synapses(P_t,neurons,on_pre='g+=w_val'); S_t.connect(i=np.arange(len(t_idx)), j=t_idx); net.add(P_t, S_t)
    
    if sp.issparse(W_matrix):
        coo=W_matrix.tocoo(); S_rec=b2.Synapses(neurons,neurons,'w:1',on_pre='g+=w*0.275*mV',delay=1.8*b2.ms)
        S_rec.connect(i=coo.col, j=coo.row); S_rec.w=coo.data; W_csr=W_matrix.tocsr(); net.add(S_rec)
    else: return None
    
    spikes = b2.SpikeMonitor(neurons)
    net.add(spikes)
    
    print("  > Executing Brian2 run...")
    net.run(total_time * b2.ms, namespace=locals())
    
    # --- 3. 数据提取 ---
    print("  > Processing Flux Data...")
    all_t = spikes.t/b2.ms
    all_i = spikes.i
    
    def compute_flux_matrix(t_start, t_end):
        mask = (all_t >= t_start) & (all_t < t_end)
        duration = (t_end - t_start) / 1000.0
        if np.sum(mask) == 0 or duration <= 0: return sp.csr_matrix((N, N))
        counts = np.bincount(all_i[mask], minlength=N)
        nu = counts / duration
        return W_csr @ sp.diags(nu)

    # --- A. 直接使用传入的时间窗口提取特征矩阵 ---
    print("  > Extracting Static Phase Maps...")
    F_feed = compute_flux_matrix(feed_time_range[0], feed_time_range[1])
    F_kill = compute_flux_matrix(kill_time_range[0], kill_time_range[1])
    
    # --- B. 生成全时段动态序列 ---
    # 依然生成从 t1 (2000ms) 开始的序列用于画图，因为通常我们只关心 Threat 介入后的变化
    print(f"  > Generating Time-Series Flux (Start from {t1}ms)...")
    time_points = np.arange(t1, total_time, dt_window)
    
    flux_list = []
    threat_list = []
    time_list = []
    
    for t in time_points:
        t_win_start = t - dt_window
        F_t = compute_flux_matrix(t_win_start, t)
        flux_list.append(F_t)
        time_list.append(t)
        
        # 记录频率 (用于画图 X 轴)
        if t < t2: th = 0
        elif t < t3: th = max_threat_freq * (t - t2) / duration_threat_ramp
        else: th = max_threat_freq
        threat_list.append(th)
        
    print(f"  > Generated {len(flux_list)} time slices.")

    sim_data = {
        'F_feed': F_feed,
        'F_kill': F_kill,
        'W_csr': W_csr,
        'flux_list': flux_list,
        'threat_list': threat_list,
        'time_list': time_list,
        'neuron_ids': neuron_ids,
        'id_map': id_map,
        'mn9_idx': id_map.get(mn9_id),
        'sugar_ids': sugar_ids,
        'threat_ids': threat_ids,
        'time_markers': (t1, t2, t3, total_time) # 记录关键时间点
    }
    return sim_data















def extract_skeleton_adaptive_threshold(
    sim_data,
    # 
    highway_width=25,      
    interceptor_width=10,  
    trace_depth=4,         
    include_loops=True,    
    prune_dangling=True,
    threshold_scale=0.5,
    min_firing_rate=0.0,
    # 
    interceptor_target_depth=2  # only find n level  (必须 <= trace_depth)
):
    print(f"\n=== Skeleton Extraction (Limited Inhibition Range) ===")
    print(f"  > Inhibition Target Depth: Top {interceptor_target_depth} layers closest to MN9")
    
    
    F_feed_raw = sim_data['F_feed']
    F_kill_raw = sim_data['F_kill']
    W_csr = sim_data['W_csr']
    mn9_idx = sim_data['mn9_idx']
    id_map = sim_data['id_map']
    
    sugar_indices = set([id_map[u] for u in sim_data['sugar_ids'] if u in id_map])
    threat_indices = set([id_map[u] for u in sim_data['threat_ids'] if u in id_map])
    
    if mn9_idx is None: return None, None

    # =========================================================================
    # PHASE 0: exclude low-rate neurons
    # =========================================================================
    print(f"  > [Phase 0] Pruning neurons with activity < {min_firing_rate} Hz...")
    
    w_out_sum = np.array(np.abs(W_csr).sum(axis=0)).flatten()
    f_feed_out_sum = np.array(np.abs(F_feed_raw).sum(axis=0)).flatten()
    f_kill_out_sum = np.array(np.abs(F_kill_raw).sum(axis=0)).flatten()
    
    valid_mask = w_out_sum > 1e-9
    nu_feed_est = np.zeros_like(w_out_sum); nu_kill_est = np.zeros_like(w_out_sum)
    nu_feed_est[valid_mask] = f_feed_out_sum[valid_mask] / w_out_sum[valid_mask]
    nu_kill_est[valid_mask] = f_kill_out_sum[valid_mask] / w_out_sum[valid_mask]
    
    max_nu = np.maximum(nu_feed_est, nu_kill_est)
    is_active = max_nu >= min_firing_rate
    protected_nodes = list(sugar_indices) + list(threat_indices) + [mn9_idx]
    is_active[protected_nodes] = True
    
    D_mask = sp.diags(is_active.astype(float))
    F_feed = D_mask @ F_feed_raw @ D_mask
    F_kill = D_mask @ F_kill_raw @ D_mask
    
    # 矩阵准备
    F_feed_abs = np.abs(F_feed)
    F_feed_pos = F_feed.copy(); F_feed_pos.data = np.where(F_feed_pos.data > 0, F_feed_pos.data, 0); F_feed_pos.eliminate_zeros()
    F_kill_neg = F_kill.copy(); F_kill_neg.data = np.where(F_kill_neg.data < -0.1, np.abs(F_kill_neg.data), 0); F_kill_neg.eliminate_zeros()
    F_kill_pos = F_kill.copy(); F_kill_pos.data = np.where(F_kill_pos.data > 0, F_kill_pos.data, 0); F_kill_pos.eliminate_zeros()
    F_kill_abs = np.abs(F_kill)

    # =========================================================================
    # PHASE A: circuit cutoff
    # =========================================================================
    print("  > [Phase A] Backward Search...")
    
    # --- 1. feeding (Highway) ---
    highway_candidates = set([mn9_idx])
    

    highway_targets_for_inhibition = set([mn9_idx]) 
    
    # 1.1 Layer 1 
    inputs_layer1 = F_feed_abs[mn9_idx, :].toarray().flatten()
    best_layer1 = np.argsort(inputs_layer1)[::-1][:highway_width]
    valid_layer1 = [src for src in best_layer1 if inputs_layer1[src] > 0.1]
    
    if len(valid_layer1) > 0:
        base_flux_val = inputs_layer1[valid_layer1[-1]]
        deep_threshold = base_flux_val * threshold_scale
        current_layer = valid_layer1
        highway_candidates.update(valid_layer1)
        
        # check Layer 1
        if interceptor_target_depth >= 1:
            highway_targets_for_inhibition.update(valid_layer1)
    else:
        current_layer = []
        deep_threshold = 0
    
    # 1.2 Layer 2+ 
    for d in range(trace_depth - 1): 

        next_layer = set()
        
        for target in current_layer:
            inputs = F_feed_abs[target, :].toarray().flatten()
            strong_candidates = np.where(inputs >= deep_threshold)[0]
            if len(strong_candidates) == 0: continue
            
            candidate_fluxes = inputs[strong_candidates]
            top_k_local = np.argsort(candidate_fluxes)[::-1][:highway_width]
            best_sources = strong_candidates[top_k_local]
            
            for src in best_sources:
                if src not in highway_candidates:
                    next_layer.add(src)
                    highway_candidates.add(src)
        

        # d=0 is Layer 2, d=1 is Layer 3...
        # Layer Index = d + 2
        if (d + 2) <= interceptor_target_depth:
            highway_targets_for_inhibition.update(next_layer)
            
        current_layer = list(next_layer)
        if not current_layer: break
            
    print(f"    Highway Candidates: {len(highway_candidates)}")
    print(f"    Nodes targeted for Inhibition: {len(highway_targets_for_inhibition)} (Top {interceptor_target_depth} Layers)")
    
    # --- 2. escape (Find Interceptors) ---
    interceptors = set()
    # only search in highway_targets_for_inhibition
    for node in highway_targets_for_inhibition:
        inhibitors = F_kill_neg[node, :].toarray().flatten()
        best_inh = np.argsort(inhibitors)[::-1][:interceptor_width]
        for src in best_inh:
            if inhibitors[src] > 0.1 and src not in highway_candidates:
                interceptors.add(src)
    print(f"    Interceptors: {len(interceptors)}")
    
    # --- 3.  (Backtrace Interceptors) ---

    backtrace_candidates = set()
    all_inh_inputs = []
    for target in interceptors:
        inputs = F_kill_abs[target, :].toarray().flatten()
        best = np.argsort(inputs)[::-1][:5]
        for src in best:
            if inputs[src] > 0.1: all_inh_inputs.append(inputs[src])
    
    bt_threshold = np.median(all_inh_inputs) * threshold_scale if len(all_inh_inputs)>0 else 0.1
    
    current_layer = list(interceptors)
    for d in range(2): 
        next_layer = set()
        for target in current_layer:
            inputs = F_kill_abs[target, :].toarray().flatten()
            cutoff = bt_threshold if d > 0 else 0.1 
            strong_candidates = np.where(inputs >= cutoff)[0]
            if len(strong_candidates) == 0: continue
            
            candidate_fluxes = inputs[strong_candidates]
            top_k_local = np.argsort(candidate_fluxes)[::-1][:5]
            best_sources = strong_candidates[top_k_local]
            
            for src in best_sources:
                if src not in highway_candidates and src not in interceptors and src not in backtrace_candidates:
                    backtrace_candidates.add(src)
                    next_layer.add(src)
        current_layer = list(next_layer)
    print(f"    Backtrace Candidates: {len(backtrace_candidates)}")

    # =========================================================================
    # PHASE B & C: check circuit connectivity
    # =========================================================================
    final_highway = set(highway_candidates)
    final_inhibition = set(interceptors) | set(backtrace_candidates)
    
    if prune_dangling:
        print("\n  > [Phase B] Pruning dangling branches...")
        def get_reachable(start_nodes, pool, adj):
            reachable = set()
            queue = deque(start_nodes)
            pool_set = set(pool) | set(start_nodes)
            visited = set(start_nodes)
            while queue:
                u = queue.popleft()
                reachable.add(u)
                outputs = adj[:, u].nonzero()[0]
                for v in outputs:
                    if v in pool_set and v not in visited:
                        visited.add(v); queue.append(v)
            return reachable

        valid_feeding = get_reachable(list(sugar_indices), highway_candidates, F_feed_pos)
        final_highway = (highway_candidates & valid_feeding) | {mn9_idx}
        
        inhibition_pool = interceptors | backtrace_candidates
        valid_inhibition = get_reachable(list(threat_indices), inhibition_pool, F_kill_pos)
        final_inhibition = inhibition_pool & valid_inhibition
        
        print(f"    Kept {len(final_highway)} Highway nodes, {len(final_inhibition)} Inhibition nodes.")

    skeleton_indices = set()
    skeleton_indices.update(final_highway)
    skeleton_indices.update(final_inhibition)
    
    if include_loops:
        current_nodes = list(skeleton_indices)
        for node in current_nodes:
            f_in = F_feed_pos[node, :].toarray().flatten()
            f_out = F_feed_pos[:, node].toarray().flatten()
            resonance = f_in * f_out
            best = np.argmax(resonance)
            if resonance[best] > 1.0 and best not in skeleton_indices:
                skeleton_indices.add(best)

    skeleton_indices.update(sugar_indices)
    skeleton_indices.update(threat_indices)
    
    final_indices = sorted(list(skeleton_indices))
    final_ids = np.array(sim_data['neuron_ids'])[final_indices]
    
    indices_in_orig = [id_map[uid] for uid in final_ids]
    W_skeleton = W_csr[np.ix_(indices_in_orig, indices_in_orig)]
    
    print(f"=== Final Skeleton: {len(final_ids)} Neurons ===")
    return final_ids, W_skeleton * 1.0





def load_skeleton_data(filename="skeleton_data.pkl"):
    """
    import circuit
    back skeleton: (skel_ids, skel_W)
    """
    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    if not os.path.exists(filename):
        print(f"❌ Error: File not found at {filename}")
        return None, None
        
    try:
        with open(filename, 'rb') as f:
            data_packet = pickle.load(f)
            
        ids = data_packet['skel_ids']
        W = data_packet['skel_W']
        
        print(f"✅ Successfully loaded skeleton data from: {filename}")
        print(f"   - IDs count: {len(ids)}")
        print(f"   - W matrix: {W.shape}")
        
        return ids, W
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None



def save_skeleton_data(ids, W, filename="skeleton_data.pkl"):

    data_packet = {
        'skel_ids': ids,
        'skel_W': W
    }
    

    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data_packet, f)
        print(f"✅ Successfully saved skeleton data to: {os.path.abspath(filename)}")
        print(f"   - IDs shape: {np.shape(ids)}")
        print(f"   - W shape: {W.shape} (Format: {type(W).__name__})")
    except Exception as e:
        print(f"❌ Error saving file: {e}")