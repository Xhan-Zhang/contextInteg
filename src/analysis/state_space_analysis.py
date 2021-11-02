from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math

from figtools import Figure
from .. import train

fs = 12 # font size
def distance(x,y,z):
    d = math.sqrt(int(x)**2+int(y)**2+int(z)**2)
    return d

def circular_hist(ax, x, color, alpha = 1, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax

    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, color=color,fill=True,linewidth=1, bottom=1, alpha=alpha)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    ax.set_yticks([1,3,5,7])
    ax.set_ylim([0,5.4])
    ax.tick_params(labelsize=14, labelbottom=False, labelleft=False)

    return n, bins, patches


def plot_task(rule_trains,trial):
    from matplotlib import pyplot as plt
    if rule_trains == 'contextInteg_decision_making':

        plt.figure()
        for i in range(len(trial.x[0, 0, :])):

            data = trial.x[:, :, i]

            if i == 0:
                plt.plot(data, color='royalblue')
            elif i == 1:
                plt.plot(data, color='darkblue')

            elif i == 2:
                plt.plot(data, color='violet')
            elif i == 3:
                plt.plot(data, color='purple')
            elif i == 4:
                plt.plot(data, linewidth='3', color='blue', label='cue_color_blue')
            elif i == 5:
                plt.plot(data, linewidth='3', color='red', label='cue_motion_red')

        for i in range(len(trial.y[0, 0, :])):
            data = trial.y[:, 0, i]

            if i == 0:
                plt.plot(data, '--', color='royalblue', label='out_color_left')  #
            elif i == 1:
                plt.plot(data, '--', color='darkblue', label='out_color_right')  #
            elif i == 2:
                plt.plot(data, '--', color='violet', label='out_motion_left')  #
            elif i == 3:
                plt.plot(data, '--', color='purple', label='out_motion_right')  #

        plt.show()

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

    trial_sample, run_step, rnn_network = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False, noise_on=noise_on,
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
    variance_explained_sem = np.std(variance_explained_list, axis=0) / np.sqrt(variance_explained_list.shape[0])

    fig = plt.figure(figsize=(2, 2.2))
    ax = fig.add_axes([0.4, 0.2, 0.5, 0.75])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(np.arange(1, 1 + len(variance_explained_mean)), variance_explained_mean, color='black', marker='o')
    plt.gca().set_xlabel('PC', fontsize=fs)
    plt.gca().set_ylabel('Explained Var. Ratio', fontsize=fs-2)
    plt.xticks([1, 3, 5], fontsize=fs-2)
    plt.xlim(0,5.6)
    plt.yticks(fontsize=fs-2)
    plt.show()
    return fig

def PCA_plot_3D_105_120(model_serial_idx=0,
                             c_colors=0.02,
                             c_motions=np.array([-0.08, -0.06, -0.04, -0.02,0.02, 0.04, 0.05, 0.08]),
                             gamma_bar_motion=0.8,
                             gamma_bar_color=0.8,
                             noise_on=False):

    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=np.array([ -0.08, -0.06, -0.04, -0.02,0.02, 0.04, 0.05, 0.08]),
                                                  c_motions=0.02,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=0,
                                                  noise_on=noise_on)
    start_projection=105
    end_projection = 120
    start_time_0, _ = trial_sample_0.epochs['integrate']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    neural_activity_cue_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    neural_activity_list_0 = list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_neural_activity_0 = np.concatenate(neural_activity_list_0, axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=c_colors,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1=start_time_0
    end_time_1 = end_time_0


    neural_activity_cue_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    neural_activity_list_1 = list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_neural_activity_1 = np.concatenate(neural_activity_list_1, axis=0)


    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)

    _alpha_list = [1, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 1, 1, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 1]

    pca = PCA(n_components=3)
    pca.fit(concate_neural_activity)
    concate_neural_activity_transform = pca.transform(concate_neural_activity)

    time_size = end_time-start_time
    delim = np.cumsum(time_size)
    concate_transform_split = np.split(concate_neural_activity_transform, delim[:-1], axis=0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(0, len(concate_transform_split)):

        if i<batch_size:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], linewidth='2.5',color='#1f77b4', alpha=_alpha_list[i])
            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  s=80, marker='*', color='#1f77b4', alpha=_alpha_list[i])
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  s=60,marker='o', color='#1f77b4', alpha=_alpha_list[i])
        else:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], linewidth='2.5',color='grey', alpha=_alpha_list[i])
            ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  s=80,marker='*', color='grey', alpha=_alpha_list[i])
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  s=60,marker='o', color='grey', alpha=_alpha_list[i])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.view_init(20, 120)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()
    return fig



def return_choice_axis_color(model_serial_idx=0,start_projection=105,end_projection=120,compont_i=0,cue=0,noise_on=False):

    c_motions = np.array([  -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08])
    gamma_bar_motion = 0.8
    gamma_bar_color = 0.8
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                  c_motions=0.02,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=0,
                                                  noise_on=noise_on)

    start_time_0, _ = trial_sample_0.epochs['integrate']
    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    neural_activity_cue_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.02,
                                                      c_motions=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=0,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_cue_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)
    # plot state space
    pca_1 = PCA(n_components=2)
    pca_1.fit(concate_neural_activity)
    choice_axis = pca_1.components_[compont_i]

    # plot variancs
    #fig_choice = plot_Explained_Ratio(concate_neural_activity,name_axis = 'cue_'+ str(cue))

    return choice_axis, concate_neural_activity


def return_choice_axis_motion(model_serial_idx=0,start_projection=105,end_projection=120,compont_i=0,cue=0,noise_on=False):

    c_motions = np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08])
    gamma_bar_motion = 0.8
    gamma_bar_color = 0.8
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=np.array([-0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                  c_motions=0.02,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=1,
                                                  noise_on=noise_on)


    start_time_0, _ = trial_sample_0.epochs['integrate']
    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    neural_activity_cue_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.02,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_cue_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)
    # plot state space
    pca_1 = PCA(n_components=2)
    pca_1.fit(concate_neural_activity)
    choice_axis = pca_1.components_[compont_i]

    # plot variancs
    #fig_choice = plot_Explained_Ratio(concate_neural_activity,name_axis = 'cue_'+ str(cue))

    return choice_axis, concate_neural_activity


def return_cue_axis(model_serial_idx=0,start_projection=5,end_projection=25,compont_i=0,cue=0,noise_on=False):

    c_motions = np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.06, 0.08])
    c_colors = np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.06, 0.08])
    gamma_bar_motion = 0.8
    gamma_bar_color = 0.8
    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=c_colors,
                                                  c_motions=c_motions,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=cue,
                                                  noise_on=noise_on)


    start_time_0, _ = trial_sample_0.epochs['integrate']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    neural_activity_cue_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=c_colors,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=cue,
                                                      noise_on=noise_on)

    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_cue_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)
    # plot state space
    pca_cue = PCA(n_components=2)
    pca_cue.fit(concate_neural_activity)
    cue_axis = pca_cue.components_[compont_i]

    # plot variancs
    #fig_choice = plot_Explained_Ratio(concate_neural_activity,name_axis = 'cue_motion'+ str(cue))

    return cue_axis


def projection_state_space_cue_stim(model_serial_idx=0,
                                    start_projection = 0,
                                    end_projection = 0,
                                     c_colors=0.02,
                                     c_motions=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                     gamma_bar_motion=0.8,
                                     gamma_bar_color=0.8,
                                     noise_on=False):

    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    color_choice_axis, _ = return_choice_axis_color(model_serial_idx, start_projection=65, end_projection=105,compont_i=1, cue=0)
    motion_choice_axis, _ = return_choice_axis_motion(model_serial_idx, start_projection=65, end_projection=105,compont_i=1, cue=1)
    color_cue_axis, _ = return_choice_axis_color(model_serial_idx, start_projection=start_projection,
                                             end_projection=end_projection, compont_i=0, cue=0)
    motion_cue_axis, _ = return_choice_axis_motion(model_serial_idx, start_projection=start_projection,
                                             end_projection=end_projection, compont_i=0, cue=1)

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                  c_motions=0.02,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=0,
                                                  noise_on=noise_on)

    start_time_0, _ = trial_sample_0.epochs['integrate']
    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    neural_activity_0 = run_step_0.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.02,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)

    # plot state space
    pca = PCA(n_components=4)
    pca.fit(concate_neural_activity)
    concate_neural_activity_transform = pca.transform(concate_neural_activity)
    covariance = pca.get_covariance()

    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    concate_transform_split = np.split(concate_neural_activity_transform, delim[:-1], axis=0)

    #plot
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    _alpha_list_1=[0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8]
    basecolor = Figure.colors('#1f77b4')

    color = '#1f77b4'
    k0=0
    k1=1
    ##projection

    ax.plot([-30 * np.sum(pca.components_[k0] * color_cue_axis), 40 * np.sum(pca.components_[k0] * color_cue_axis)],
            [-30 * np.sum(pca.components_[k1] * color_cue_axis), 40 * np.sum(pca.components_[k1] * color_cue_axis)],
            '-', color='steelblue', alpha=0.8, linewidth=2.5)
    ax.plot([-30 * np.sum(pca.components_[k0] * motion_cue_axis), 40 * np.sum(pca.components_[k0] * motion_cue_axis)],
            [-30 * np.sum(pca.components_[k1] * motion_cue_axis), 40 * np.sum(pca.components_[k1] * motion_cue_axis)],
            '-', color='grey', alpha=0.8, linewidth=2.5)

    #plot
    for i in range(0, len(concate_transform_split)):
        if i<batch_size:
            ax.plot(concate_transform_split[i][:, k0], concate_transform_split[i][:, k1],linewidth='1.8',color=color,alpha=_alpha_list_1[i])
            ax.scatter(concate_transform_split[i][0, k0], concate_transform_split[i][0, k1], s=40,marker='*', color=color,alpha=_alpha_list_1[i])
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1], s=40,marker='o', color=color,alpha=_alpha_list_1[i])
        else:
            ax.plot(concate_transform_split[i][:, k0], concate_transform_split[i][:, k1],linewidth='1.8',color='grey',alpha=_alpha_list_1[i])
            ax.scatter(concate_transform_split[i][0, k0], concate_transform_split[i][0, k1], s=40,marker='*', color='grey',alpha=_alpha_list_1[i])
            ax.scatter(concate_transform_split[i][-1, k0], concate_transform_split[i][-1, k1], s=40,marker='o', color='grey',alpha=_alpha_list_1[i])

    ax.set_xlabel('cue-PC1', fontsize=fs+3)
    ax.set_ylabel('cue-PC2', fontsize=fs+3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()
    return fig

def covariance_matrix(model_serial_idx=0,
                                    start_projection = 0,
                                    end_projection = 0,
                                    number_activate=0,
                                     c_colors=0.02,
                                     c_motions=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                     gamma_bar_motion=0.8,
                                     gamma_bar_color=0.8,
                                     noise_on=False):

    batch_size = len(c_motions)
    gamma_bar_colors = np.array([gamma_bar_color] * batch_size)
    gamma_bar_motions = np.array([gamma_bar_motion] * batch_size)
    variance_explained_list = list()

    color_cue_axis, _ = return_choice_axis_color(model_serial_idx, start_projection=start_projection,
                                             end_projection=end_projection, compont_i=0, cue=0)
    motion_cue_axis, _ = return_choice_axis_motion(model_serial_idx, start_projection=start_projection,
                                             end_projection=end_projection, compont_i=0, cue=1)
    color_choice_axis,_ = return_choice_axis_color(model_serial_idx, start_projection=start_projection, end_projection=end_projection,compont_i=0, cue=0)
    motion_choice_axis,_ = return_choice_axis_motion(model_serial_idx, start_projection=start_projection, end_projection=end_projection,compont_i=0, cue=1)

    #
    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                  batch_size=batch_size,
                                                  c_colors=np.array([ -0.02, -0.04, -0.06, -0.08,0.02, 0.04, 0.06, 0.08]),
                                                  c_motions=0.02,
                                                  gamma_bar_colors=gamma_bar_colors,
                                                  gamma_bar_motions=gamma_bar_motions,
                                                  cues=0,
                                                  noise_on=noise_on)

    start_time_0, _ = trial_sample_0.epochs['integrate']
    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)
    neural_activity_0 = run_step_0.activity_shaped.detach().cpu().numpy()

    neural_activity_list = []
    for j in range(8):
        max_rate_idx = np.argsort(np.mean(neural_activity_0[:,j,:], axis=0))[::-1][0:number_activate]
        neural_activity = neural_activity_0[:,j, max_rate_idx]
        neural_activity_list.append(neural_activity)

    neural_activity_0 = np.array(neural_activity_list).reshape(120,8,number_activate)
    concate_neural_activity_0 = np.concatenate(list(neural_activity_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.02,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0


    neural_activity_1 = run_step_1.activity_shaped.detach().cpu().numpy()
    neural_activity_list_1 = []
    for j in range(8):
        max_rate_idx = np.argsort(np.mean(neural_activity_1[:, j, :], axis=0))[::-1][0:number_activate]
        neural_activity = neural_activity_1[:, j, max_rate_idx]
        neural_activity_list_1.append(neural_activity)

    neural_activity_1 = np.array(neural_activity_list_1).reshape(120, 8, number_activate)
    concate_neural_activity_1 = np.concatenate(list(neural_activity_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)

    # plot state space
    pca = PCA(n_components=4)
    pca.fit(concate_neural_activity)
    concate_neural_activity_transform = pca.transform(concate_neural_activity)
    covariance = pca.get_covariance()

    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    concate_transform_split = np.split(concate_neural_activity_transform, delim[:-1], axis=0)

    return concate_neural_activity, covariance

