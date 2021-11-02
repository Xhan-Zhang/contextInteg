from __future__ import division

from sklearn.decomposition import PCA

import math
import time
import autograd.numpy as np
from autograd import grad
import numpy as np
from matplotlib import pyplot as plt
from .. import train



fs=10

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

    trial_sample, run_step, rnn_network = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False,
                                                       noise_on=noise_on,
                                                       batch_size=batch_size,
                                                       c_color=c_colors,
                                                       c_motion=c_motions,
                                                       gamma_bar_color=gamma_bar_colors,
                                                       gamma_bar_motion=gamma_bar_motions,
                                                       cue=cues)

    return trial_sample, run_step, rnn_network

def my_fixed_point(model_i=0, init_index=0, rand_init = 0, start_time=65, end_time=105,cue=0, c_color=0, c_motion=0):
    num_points = 1; eps = 0.001; opt_iters = 400000; thresh = 0.0001; max_tries = 40; init_scale = 5; plot_loss = 1
    '''This function uses the trained parameters to find num_points fixed points. It does a gradient
    descent to minimize q(x), which is analagous to the energy of the system. To just plot the gradient descent loss
    and step size for finding a single fixed point,  set the plot_loss flag to 1.
    Inputs:
        rnn: Should be a JazNet class object.
        inp: A fixed value for the input(s). Can just be a list (e.g. [1,0])
        num_points: Number of points to find (if plot_loss=0)
        eps: Epsilon value that scales the step size
        opt_iters: How many iterations to run to try to converge on a fixed point
        thresh: Threshold for the norm of the network activity before calling it a fixed point
        rand_init: Randomly pick a starting point if 1 (default), otherwise go with the network's current activity.
        plot_loss: Will result in only finding one fixed point. Shows how loss function/step size changes. Default 0

    Outputs:
        all_points: Gives activity for all fixed points found in a num_points-by-N array
        fp_outputs: Network output at each fixed point. Note: Should change this depending on
            whether network uses tanh of activities for outpus, or if it has biases.
        trajectories: List with num_points elements, where each element is a TxN array, where T is the number of
        steps it took to find the fixed point and N is the number of neurons.
        '''
    print('optimization:','eps',eps,'iters',opt_iters)



    rnn_par = {}
    rule_trains = "contextInteg_decision_making"
    model_serial_idx = 'cue_20_delay_40/model_' + str(model_i) + '/finalResult'
    model_dir = './src/saved_model/' + rule_trains + '/' + str(model_serial_idx)

    _, _, rnn_network = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False, noise_on=True)
    rnn_par['rec_weights'] = rnn_network.weight_hh.detach().cpu().numpy()
    rnn_par['inp_weights'] = rnn_network.weight_ih.detach().cpu().numpy()
    rnn_par['out_weights'] = rnn_network.weight_out.detach().cpu().numpy()
    rnn_par['bias'] = rnn_network.bias_h.detach().cpu().numpy()


    trial_sample, run_step, _ = train.Runner(model_dir=model_dir, rule_trains=rule_trains, is_cuda=False,
                                                       noise_on=True,
                                                       batch_size=1,
                                                       c_color=c_color,
                                                       c_motion=c_motion,
                                                       gamma_bar_color=0.8,
                                                       gamma_bar_motion=0.8,
                                                       cue=cue)


    inp =  trial_sample['inputs'][start_time:end_time,:,:].numpy()
    initial_state = run_step.state_shaped[init_index, :, :].detach().cpu().numpy()

    def softplus(x):
        return np.log(1 + np.exp(x))
    def output(x):
        return np.dot(softplus(x), rnn_par['out_weights'])

    def F(x):
        return (-x + np.dot(softplus(x), rnn_par['rec_weights']) +
                np.dot(inp, rnn_par['inp_weights']) + rnn_par['bias'])

    def q(x):
        return 1 / 2 * np.linalg.norm(F(x)) ** 2
    def loss_grad(x):

        return grad(x)

    def find_point(opt_iters, eps):
        loss = []
        stepsize = []
        x_history = []
        x_list = []
        if rand_init:
            x = np.random.randn(initial_state.size)*init_scale  # The randomized initial activity needs to be big enough to relax to interesting points
        else:
            x = initial_state

        for i in range(opt_iters):
            loss.append(q(x))
            if loss[i] < thresh:
                break
            gradient = loss_grad(q)
            step = eps * gradient(x)
            stepsize.append(np.linalg.norm(step))
            x = x - step
            x_history.append(x)
            optimal_value = q(x)

        return x, loss, stepsize, x_history, x_list#

    start = time.time()

    if plot_loss:  # To see the optimization process to find one fixed point
        x, loss, stepsize, x_history, x_list = find_point(opt_iters, eps)
        print('min_val:', loss[-1])
        x_list = np.array(x_list)
        # with open('data.npy', 'wb') as f:
        #     np.save(f, x_list)

        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.plot(loss[-100:-1])
        # plt.title('Loss, last 100')
        # plt.subplot(2, 2, 2)
        # plt.plot(loss)
        # plt.xlabel('Iteration')
        # plt.title('Loss, all')
        # plt.subplot(2, 2, 3)
        # plt.plot(stepsize)
        # plt.xlabel('Iteration')
        # plt.title('Step size')
        # plt.subplot(2, 2, 4)
        # plt.plot(abs(x))
        # plt.xlabel('Iteration')
        # plt.title('Step size')
        # plt.show()
        # print(model_i, 'Last loss:',loss[-1])#
        return x
    else:  # For finding a bunch of fixed points
        all_points = np.zeros((num_points, np.size(initial_state)))
        fp_outputs = np.zeros((num_points, rnn_par['out_weights'].shape[1]))
        trajectories = []
        for p in range(num_points):
            endloss = 1000
            tries = 0
            while endloss > thresh:
                if tries < max_tries:
                    x, loss, stepsize, x_history = find_point(opt_iters, eps)
                    endloss = loss[-1]
                    tries += 1
                else:
                    print('Unsuccessful run; error=%g' % endloss)
                    raise TimeoutError('No fixed points found in tries')
            all_points[p, :] = x
            fp_outputs[p] = output(x)
            trajectories.append(np.array(x_history))
            print('.', "end=")
        finish = time.time()
        print('Done with fixed points in %d seconds' % (finish - start))
        return all_points, fp_outputs, trajectories




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

    plt.gca().spines['top'].set_colorible(False)
    plt.gca().spines['right'].set_colorible(False)

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

    trial_sample_0, run_step_0, _ = generate_test_trial(model_serial_idx=model_serial_idx,
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


    neural_activity_cue_0 = run_step_0.neural_activity_binder.detach().cpu().numpy()
    concate_neural_activity_0 = np.concatenate(list(neural_activity_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size)), axis=0)
    print(concate_neural_activity_0)

    trial_sample_1, run_step_1, _ = generate_test_trial(model_serial_idx=model_serial_idx,
                                                      batch_size=batch_size,
                                                      c_colors=0.2,
                                                      c_motions=c_motions,
                                                      gamma_bar_colors=gamma_bar_colors,
                                                      gamma_bar_motions=gamma_bar_motions,
                                                      cues=1,
                                                      noise_on=noise_on)
    start_time_1 = start_time_0
    end_time_1 = end_time_0

    neural_activity_cue_1 = run_step_1.neural_activity_binder.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)

    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1), axis=0)
    print(concate_neural_activity)

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

    plt.gca().spines['top'].set_colorible(False)
    plt.gca().spines['right'].set_colorible(False)

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


def return_choice_axis_color(model_serial_idx=0,start_projection=105,end_projection=120,compont_i=0,cue=0,noise_on=False):


    integrate = end_projection-start_projection
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



    neural_activity_cue_0 = run_step_0.neural_activity_binder.detach().cpu().numpy()
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

    neural_activity_cue_1 = run_step_1.neural_activity_binder.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)
    # plot state space
    pca = PCA(n_components=2)
    pca.fit(concate_neural_activity)
    concate_neural_activity_transform = pca.transform(concate_neural_activity)
    choice_axis = pca.components_[compont_i]

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

    neural_activity_cue_0 = run_step_0.neural_activity_binder.detach().cpu().numpy()
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

    neural_activity_cue_1 = run_step_1.neural_activity_binder.detach().cpu().numpy()
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
    print("start_time_0,end_time_0",start_time_0,end_time_0)



    neural_activity_cue_0 = run_step_0.neural_activity_binder.detach().cpu().numpy()
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

    neural_activity_cue_1 = run_step_1.neural_activity_binder.detach().cpu().numpy()
    concate_neural_activity_1 = np.concatenate(list(neural_activity_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size)), axis=0)
    concate_neural_activity = np.concatenate((concate_neural_activity_0, concate_neural_activity_1),axis=0)
    # plot state space
    pca_cue = PCA(n_components=2)
    pca_cue.fit(concate_neural_activity)
    cue_axis = pca_cue.components_[compont_i]

    # plot variancs
    #fig_choice = plot_Explained_Ratio(concate_neural_activity,name_axis = 'cue_motion'+ str(cue))

    return cue_axis



