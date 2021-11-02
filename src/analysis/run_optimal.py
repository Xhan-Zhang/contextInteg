import errno
import numpy as np
import os
import sys

import find_fixed_point

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def optimal(count_init=0,model_i=0, random_init=0):
    data_different_init = []
    i=0
    for init_value in np.array(count_init):

        print('===========',i+1,'=========================================================')
        print('========================================================================')
        x_8 = []
        for c in coherence:
            if cue==0:
                print('@@@@cue',cue,'c', c)
                c_color=c
                c_motion = np.array([0.02])
            if cue==1:
                print('@@@@cue', cue,'c', c)
                c_color = np.array([0.02])
                c_motion = c

            print('**condition:', 'model',model_i, 'init_x', init_value, 'cue', cue, 'c', c, 'interval',start_time, end_time)
            x = find_fixed_point.my_fixed_point(model_i=model_i,
                                                         init_index=init_value,
                                                         rand_init=random_init,
                                                         start_time=start_time,
                                                         end_time=end_time,
                                                         c_color=c_color,
                                                         c_motion=c_motion,
                                                         cue=cue)#
            x = np.squeeze(x)
            x_8.append(x)
        data_different_init.append(np.array(x_8))
    print('**',len(data_different_init))
    #data_all_init =  np.array(data_different_init).reshape(count_init_value*8, 256)

    return data_different_init

########number_case in each file is count_init_value*8
if __name__ == '__main__':
    model_i = 1
    random_init = 1

    if random_init:
        file_path = "./data_random_init_40w/"
    else:
        file_path = "./data_selected_init/"
    mkdir_p(file_path + "data_model" + str(model_i) + '/')
    coherence = np.array([-0.02, -0.04, -0.06, -0.08, 0.02, 0.04, 0.06, 0.08])
    start_time = 65
    end_time = 105
    # file name
    cue = int(sys.argv[1])
    file_index = int(sys.argv[2])
    print('file_path',file_path)

    if random_init:
        count_init_value = [1,1,1]
        data_different_init = optimal(count_init=count_init_value, model_i=model_i, random_init=random_init)
    else:
        init_value = [i+1 for i in range(27)]
        start_dots = 3*file_index-2
        count_init_value = init_value[start_dots-1:start_dots+2]
        data_different_init = optimal(count_init=count_init_value, model_i=model_i, random_init=random_init)



    file_name = str(model_i) + 'model_' + 'cue' + str(cue) + '_count_' + str(len(count_init_value)) + '_' + str(
        start_time) + '_' + str(end_time) + '_' + str(file_index) + '.npy'
    print('file_path, file_name',file_path, file_name)
    with open(file_path+'data_model'+str(model_i)+'/' + file_name, 'wb') as f:
        np.save(f, data_different_init)

    print(len(data_different_init))

