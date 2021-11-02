import errno
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio


from src.analysis import neural_activity_analysis
from src.analysis import state_space_analysis
from src.analysis import find_fixed_point

def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


mkdir_p("../figure")
save_path = "../figure/"



#plot variance
for i in np.array([1]):
    fig_explained_variance = neural_activity_analysis.explained_variance('cue_20_delay_40/model_' + str(i) + '/finalResult',
                                                       c_colors=0.02,
                                                       c_motions=np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.06, 0.08]),
                                                       gamma_bar_motion=0.8,
                                                       gamma_bar_color=0.8,
                                                       noise_on=False)

    fig_explained_variance.savefig(save_path+"fig4_variance.pdf")


for i in np.array([1]):
    fig5_pca = neural_activity_analysis.activity_peak_order_complex('cue_20_delay_40/model_' + str(i) + '/finalResult', noise_on=False)
    fig5_pca.savefig(save_path+"activity_peak_order_plot_train.pdf")


for i in np.array([31]):
    fig_pca = neural_activity_analysis.activity_peak_order_plot_untrain('cue_20_delay_40/model_' + str(i) + '/finalResult', noise_on=False)
    fig_pca.savefig(save_path+"activity_peak_order_plot_untrain.pdf")

# connection_peak_order
fig_untrain = neural_activity_analysis.connection_peak_order_plot_batch_untrain(pattern='untrained',noise_on=False)
fig_untrain.savefig(save_path+"connection_peak_order_plot_untrain_8.pdf")

fig_train = neural_activity_analysis.connection_peak_order_plot_batch(pattern='trained',noise_on=False)
fig_train.savefig(save_path+"connection_peak_order_plot_train.pdf")














