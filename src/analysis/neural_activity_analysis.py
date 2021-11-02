from __future__ import division

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm

from .. import train
import math
fs = 10
def distance(x,y,z):
    d = math.sqrt(int(x)**2+int(y)**2+int(z)**2)
    return d

def generate_test_trial(model_serial_idx=0,
                         batch_size=1,
                         c_colors=np.array([0.02]),
                         c_motions=0,
                         gamma_bar_colors=0.0,
                         gamma_bar_motions=0.0,
                         cues=0,
                         noise_on=False):

    rule_trains = "contextInteg_decision_making"
    model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)
    print('model_dir', model_dir)

    trial_sample, run_step, rnn_network = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False,noise_on=noise_on,
                                          batch_size=batch_size,
                                          c_color=c_colors,
                                          c_motion=c_motions,
                                          gamma_bar_color=gamma_bar_colors,
                                          gamma_bar_motion=gamma_bar_motions,
                                          cue=cues)

    return trial_sample, run_step, rnn_network

def plot_Explained_Ratio(concate_neural_activity=0,name_axis=0):

    variance_explained_list = []
    pca_exp = PCA(n_components=5)
    pca_exp.fit(concate_neural_activity)
    variance_explained_list.append(pca_exp.variance_explained_[np.newaxis, :])

    variance_explained_list = np.concatenate(variance_explained_list, axis=0)

    variance_explained_mean = np.mean(variance_explained_list, axis=0)
    variance_explained_sem = np.std(variance_explained_list, axis=0) / np.sqrt(
        variance_explained_list.shape[0])

    fig = plt.figure(figsize=(2, 2.2))
    ax = fig.add_axes([0.4, 0.2, 0.5, 0.75])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.plot(np.arange(1, 1 + len(variance_explained_mean)), variance_explained_mean, color='black',
             marker='o')
    print(len(variance_explained_mean), variance_explained_mean)
    plt.gca().set_xlabel('PC', fontsize=fs)
    plt.gca().set_ylabel('Explained Var. Ratio', fontsize=fs-2)
    ax.set_title(name_axis, fontsize='large', fontweight='bold')

    plt.xticks([1, 3, 5], fontsize=fs-2)
    plt.xlim(0, 5.6)
    plt.yticks(fontsize=fs-2)

    plt.show()
    return fig

def explained_variance(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.06, 0.08]),
                             gamma_bar_motion=0.8,
                             gamma_bar_color=0.8,
                             noise_on=False):

    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    trial_sample_0, run_step_0,_ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=0,
                                                      noise_on=noise_on)

    start_time_0, _ = trial_sample_0.epochs['integrate']
    start_time_0 = np.zeros_like(start_time_0) + 5 * np.ones_like(start_time_0)
    end_time_0 = np.zeros_like(start_time_0) + 60 * np.ones_like(start_time_0)

    neural_activity_cue_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1,_ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.2,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_cue_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1), axis=0)

    #plot variancs
    pca = PCA(n_components=5)
    pca.fit(concate_neural_activity)
    variance_explained_list.append(pca.variance_explained_[np.newaxis, :])
    variance_explained_list = np.concatenate(variance_explained_list, axis=0)

    variance_explained_mean = np.mean(variance_explained_list, axis=0)
    variance_explained_sem = np.std(variance_explained_list, axis=0) / np.sqrt(
        variance_explained_list.shape[0])

    fig = plt.figure(figsize=(2, 2.2))
    ax = fig.add_axes([0.4, 0.2, 0.5, 0.75])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(np.arange(1, 1 + len(variance_explained_mean)), variance_explained_mean, color='black',
             marker='o')
    print(len(variance_explained_mean), variance_explained_mean)
    plt.gca().set_xlabel('PC', fontsize=fs)
    plt.gca().set_ylabel('Explained Var. Ratio', fontsize=fs-2)
    plt.xticks([1, 3, 5], fontsize=fs-2)
    plt.xlim(0,5.6)
    plt.yticks(fontsize=fs-2)
    plt.show()
    return fig

def find_specific_neuron(model_serial_idx=0,batch_size=0,c_colors=0.02,c_motions=0.02,gamma_bar_motions=0.0,
                                      gamma_bar_colors=0.0,
                                      cues=0,
                                      noise_on=True):

    rule_trains = "contextInteg_decision_making"
    model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)

    trial_sample, run_step = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False,
                                          noise_on=noise_on,
                                          batch_size=batch_size,
                                          c_color=c_colors,
                                          c_motion=c_motions,
                                          gamma_bar_color=gamma_bar_colors,
                                          gamma_bar_motion=gamma_bar_motions,
                                          cue=cues)
    start_time, _ = trial_sample.epochs['cue']
    _, end_time = trial_sample.epochs['integrate']
    start_time = np.zeros_like(end_time)

    batch_idx = 0
    neural_activity = run_step.activity_shaped[start_time[batch_idx]:end_time[batch_idx], batch_idx, :].detach().cpu().numpy()
    cue_0_specific = []
    choice_specific = []

    for i in range(256):
        if (np.max(neural_activity[5:50, i]) > 5) * (np.max(neural_activity[65:105, i]) <3):
            cue_0_specific.append(i)
        elif (neural_activity[25, i] < 3)*(neural_activity[40,i] < 3)*(neural_activity[100, i] > 5):
            choice_specific.append(i)

    return cue_0_specific,choice_specific


def activity_peak_order_complex(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([0.02]),
                             gamma_bar_motion=1,
                             gamma_bar_color=1,
                             noise_on=False):
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    func_activity_threshold = 3

    trial_sample, run_step, rnn_network = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.05, 0.08]),
                                                      c_motions=0.02,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=0,
                                                      noise_on=noise_on)


    start_time, _ = trial_sample.epochs['cue_delay']
    _, end_time = trial_sample.epochs['cue_delay'] + 0 * np.ones_like(start_time)

    batch_idx = 0
    data = run_step.activity_shaped[start_time[batch_idx]:end_time[batch_idx], batch_idx, :].detach().cpu().numpy()
    max_neural_activity = np.max(data, axis=0)
    pick_idx = np.argwhere(max_neural_activity > func_activity_threshold).squeeze()

    data = data[:, pick_idx]
    peak_time = np.argmax(data, axis=0)
    peak_order = np.argsort(peak_time, axis=0)
    data = data[:, peak_order]
    # normalize
    for i in range(0, data.shape[1]):
        if np.max(data[:, i])<0:
            data[:, i] = 0
        else:
            data[:, i] = data[:, i] / np.max(data[:, i])

    X, Y = np.mgrid[0:data.shape[0]*20:20, 0:data.shape[1]]

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    plt.gca().set_xlabel('Time (ms)', fontsize=fs+1)
    plt.gca().set_ylabel('Neuron (Sorted)', fontsize=fs+1)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    # Make the plot
    cmap = plt.get_cmap('viridis')#
    plt.pcolormesh(X, Y, data,cmap=cmap)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, aspect=15)
    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs+1)
    cbar.ax.set_title('Normalized\n activity', fontsize=fs+1)
    plt.show()
    return fig

def activity_peak_order_complex_matdata(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([0.02]),
                             gamma_bar_motion=1,
                             gamma_bar_color=1,
                             noise_on=False):
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    func_activity_threshold = 3
    trial_sample, run_step, rnn_network = generate_test_trial(model_serial_idx=model_serial_idx,
                                                              batch_size=batch_size,
                                                              c_colors=np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.05, 0.08]),
                                                              c_motions=0.02,
                                                              gamma_bar_colors=gamma_bar_colors,
                                                              gamma_bar_motions=gamma_bar_motions,
                                                              cues=0,
                                                              noise_on=noise_on)




    start_time, _ = trial_sample.epochs['cue_delay']
    _, end_time = trial_sample.epochs['cue_delay'] + 0 * np.ones_like(start_time)

    batch_idx = 0
    data = run_step.activity_shaped[start_time[batch_idx]:end_time[batch_idx], batch_idx, :].detach().cpu().numpy()
    data_save = data

    max_neural_activity = np.max(data, axis=0)
    pick_idx = np.argwhere(max_neural_activity > func_activity_threshold).squeeze()

    data = data[:, pick_idx]
    peak_time = np.argmax(data, axis=0)
    peak_order = np.argsort(peak_time, axis=0)
    data = data[:, peak_order]
    # normalize
    for i in range(0, data.shape[1]):
        if np.max(data[:, i])<0:
            data[:, i] = 0
        else:
            data[:, i] = data[:, i] / np.max(data[:, i])

    X, Y = np.mgrid[0:data.shape[0]*20:20, 0:data.shape[1]]

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])

    plt.gca().set_xlabel('Time (ms)', fontsize=fs+1)
    plt.gca().set_ylabel('Neuron (Sorted)', fontsize=fs+1)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    # Make the plot
    cmap = plt.get_cmap('viridis')
    plt.pcolormesh(X, Y, data,cmap=cmap)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, aspect=15)
    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs+1)
    cbar.ax.set_title('Normalized\n activity', fontsize=fs+1)
    plt.show()
    return data_save



def activity_peak_order_plot_untrain(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([0.02]),
                             gamma_bar_motion=1,
                             gamma_bar_color=1,
                             noise_on=False):
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    func_activity_threshold = 3

    trial_sample, run_step, rnn_network = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.02,
                                                      c_motions=0.02,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=0,
                                                      noise_on=noise_on)

    start_time, _ = trial_sample.epochs['cue_delay']
    _, end_time = trial_sample.epochs['cue_delay']

    batch_idx = 0
    data = run_step.activity_shaped[start_time[batch_idx]:end_time[batch_idx], batch_idx, :].detach().cpu().numpy()
    max_neural_activity = np.max(data, axis=0)
    pick_idx = np.argwhere(max_neural_activity > func_activity_threshold).squeeze()

    data = data[:, pick_idx]
    peak_time = np.argmax(data, axis=0)
    peak_order = np.argsort(peak_time, axis=0)
    data = data[:, peak_order]

    # normalize
    for i in range(0, data.shape[1]):
        if np.max(data[:, i])<0:
            data[:, i] = 0
        else:
            data[:, i] = (data[:, i] / np.max(data[:, i]))

    X, Y = np.mgrid[0:data.shape[0]*20:20, 0:data.shape[1]]

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])

    plt.gca().set_xlabel('Time (ms)', fontsize=fs+1)
    plt.gca().set_ylabel('Neuron (Sorted)', fontsize=fs+1)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    # Make the plot
    plt.pcolormesh(X, Y, data)
    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, aspect=15)
    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs+1)
    cbar.ax.set_title('Normalized\n activity', fontsize=fs+1)

    plt.show()
    return fig


def plot_weight_zoom_in(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([0.02]),
                             gamma_bar_motion=1,
                             gamma_bar_color=1,
                             noise_on=False):
    batch_size = len(c_motions)
    _color_list = ['blue']*256
    fs = 6


    rule_trains = "contextInteg_decision_making"
    model_serial_idx = 'cue_20_delay_40/model_' + str(7) + '/finalResult'
    model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)

    _, _, rnn_network = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False, noise_on=noise_on)

    weight_hh = rnn_network.weight_hh.detach().cpu().numpy()
    weight_hh_diag = []

    for j in range(256):
        weight_hh_diag.append(weight_hh[j][j])
    weight_hh_nondiag = weight_hh * (np.ones((256,256))-np.eye(256,dtype=int))

    #plot
    fig1 = plt.figure(figsize=(3, 2.2))
    ax1 = fig1.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(weight_hh_nondiag, 50,color=_color_list,alpha=0.5)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax1.set_ylabel('P. D. F', fontsize=fs +5 , color='black')
    ax1.set_xlabel('Recurrent weight', fontsize=fs + 5)
    plt.xticks(fontsize=fs + 5)
    plt.yticks(fontsize=fs + 5)
    plt.ylim(0, 28)

    # diag
    fig3 = plt.figure(figsize=(3, 2.2))
    ax3 = fig3.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(weight_hh_diag, 50, color='green',alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax3.set_ylabel('P. D. F', fontsize=fs + 5, color='black')
    ax3.set_xlabel('Recurrent weight', fontsize=fs + 5)
    plt.xlim(0.9, 1.01)
    plt.xticks(fontsize=fs + 5)
    plt.yticks(fontsize=fs + 5)
    ax3.set_xticks([0.9, 0.95, 1.0])
    plt.show()

    return fig1,fig3



