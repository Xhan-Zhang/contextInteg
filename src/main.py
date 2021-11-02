import argparse
import os
import sys
sys.path.append('..')
import train


parser = argparse.ArgumentParser(description='contextInteg decision making')
parser.add_argument('--rule_trains', type=str, default='contextInteg_decision_making', help='Which task should be trained')#default = 0
parser.add_argument('-r2','--l2_reg_rate', type=float, default=0, help='L2 regularization on activity')
parser.add_argument('-w2','--l2_reg_wight', type=float, default=0, help='L2 regularization on weight')
parser.add_argument('--index', type=int, default=0, help='the model index being trained')
parser.add_argument('-lr','--learning_rate', type=float, default=0.0005, help='Learning rate')

args = parser.parse_args()
model_folder_name = os.path.join('./saved_model/',args.rule_trains,'cue_20_delay_40', 'model_' + str(args.index))

while True:
    hp = train.get_default_hp(args.rule_trains)
    hp['l2_activity'] = args.l2_reg_rate
    hp['l2_weight'] = args.l2_reg_wight
    hp['learning_rate'] = args.learning_rate

    stat = train.Trainer(max_samples=1e7,
                   display_step=200,
                   model_dir=model_folder_name,
                   rule_trains=args.rule_trains,
                   hp=hp,
                   is_cuda=False)

    print("====================================================")
    print("state", stat)
    print("====================================================")

    if stat is 'OK':
        break
    else:
        run_cmd = 'rm -r ' + model_folder_name
        os.system(run_cmd)
