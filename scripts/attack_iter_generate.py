### This file is used to generate attack costs for analysis.
### Our repo provided models and adv examples, the data were generated by baselines.
### 

import sys
sys.path.append('../')
import numpy as np
import foolbox
import time
import os
import argparse
import warnings
from progress.bar import Bar

from sklearn.neighbors import KernelDensity
from keras.layers import Input
from keras.models import load_model
from detect.util import (get_data, get_noisy_samples, get_mc_predictions,
                         get_deep_representations, score_samples, normalize,
                         train_lr, compute_roc)

import attacks_method.saliency as saliency
from attacks_method.iterative_projected_gradient import BIM, L2BasicIterativeAttack
import attacks_method.boundary_attack as DBA
import gl_var

def attack_for_input(model, input_data, method):
    """Gen attack costs for input data by input attack method

    Args:
        model: keras model
        input_data [str]: input data name
        method [str]: attack method for defense

    """
    print('Attack %s samples by %s' %(input_data, method))
      
    # * attack parameters
    if args.dataset == 'mnist':
        img_rows, img_cols, img_rgb = 28, 28, 1
        epsilon = 0.3
        stepsize = 0.0006
        epsilon_l2 = 0.1
        stepsize_l2 = 0.0002
    elif args.dataset == 'cifar':
        img_rows, img_cols, img_rgb = 32, 32, 3
        epsilon = 0.03
        stepsize = 0.00006
        epsilon_l2 = 0.01
        stepsize_l2 = 0.00002
    else:
        raise Exception('Input dataset error:', args.dataset)
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))

    root_path = '../results/' 
    folder_name = args.dataset + '_attack_iter_stats/'
    folder_path = os.path.join(root_path, folder_name)
    file_name = method + '_attack_' + input_data

    if not os.path.exists(folder_path + file_name):
        os.makedirs(folder_path + file_name)
    f = open(folder_path + file_name, 'w')

    # load benign samples
    _, _, x_input, y_input = get_data(args.dataset)
    
    # except wrong label
    preds_test = model.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input.argmax(axis=1))[0]
    
    # load adv samples
    if input_data != 'benign':
        x_input = np.load('../data/Adv_'+ args.dataset +'_' + str(input_data) +'.npy')
    # cw adv file only have 1000 examples, which have been filtered 
    if input_data != 'cw':
        x_input = x_input[inds_correct]

    # only need 1000 samples
    auroc_calc_size = 1000
    x_input = x_input[0: auroc_calc_size]

    # check the input bound
    assert np.min(x_input) == 0 and np.max(x_input) == 1
    x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, img_rgb).astype('float')

    bar = Bar('Attack processing', max=auroc_calc_size)
    for index, data in enumerate(x_input):
        fmodel_predict = fmodel.predictions(data)
        label = np.argmax(fmodel_predict)

        # define criterion
        if method != 'JSMA':
            # untarged attack
            criterion = foolbox.criteria.Misclassification()
        else:
            # JSMA attack only support targeted attack, second target attack
            second_result = fmodel_predict.copy()
            second_result[np.argmax(second_result)] = np.min(second_result)
            target_label = np.argmax(second_result)
            criterion = foolbox.criteria.TargetClass(int(target_label))

        # define method
        if method == 'BIM':
            gl_var.return_iter_bim = -500
            attack = BIM(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)
            adversarial = attack(data, label, binary_search=False,
                                 epsilon=epsilon, stepsize=stepsize, iterations=500)
            attack_iter = gl_var.return_iter_bim
            max_iter = 500

        elif method == 'BIM2':
            gl_var.return_iter_bim = -500
            attack = L2BasicIterativeAttack(fmodel, criterion=criterion)  # distance=foolbox.distances.Linfinity
            adversarial = attack(data, label, binary_search=False,
                                 epsilon=epsilon_l2, stepsize=stepsize_l2, iterations=500)
            attack_iter = gl_var.return_iter_bim
            max_iter = 500

        elif method == 'JSMA':
            attack = saliency.SaliencyMapAttack(fmodel, criterion)
            gl_var.return_iter_jsma = -2000
            adversarial = attack(data, label)
            attack_iter = gl_var.return_iter_jsma
            max_iter = 2000

        elif method == 'DBA':
            gl_var.return_iter_dba = -5000
            attack = DBA.BoundaryAttack(fmodel, criterion=criterion)
            adversarial = attack(data, label)
            attack_iter = gl_var.return_iter_dba
            max_iter = 5000

        else:
            raise Exception('Invalid attack method:', method)

        # * adversarial confirm part
        # * as we use foolbox to attack, ignore this part
        # adv_predict = fmodel.predictions(adversarial)
        # adv_label = np.argmax(fmodel_predict)
        # assert adv_label != label

        if adversarial is not None:
            f.write(str(attack_iter) + "\n")
        # adversarial is none means the attack method cannot find an adv, so return the max iterations
        else:
            f.write(str(max_iter) + "\n")
        
        bar.suffix = '({index}/{size}) | Total: {total:} | ETA: {eta:}'.format(
                    index=index,
                    size=auroc_calc_size,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()
    f.close() # close the file

def main(args):
    # load the model
    kmodel = load_model('../data/model_%s.h5' % args.dataset)

    # get attack cost
    attack_for_input(kmodel, input_data=args.input, method=args.attack)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar'",
        required=False, type=str, default='mnist',
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; recommanded to use JSMA, BIM, BIM2 or DBA. ",
        required=True, type=str
    )
    parser.add_argument(
        '-i', '--input',
        help="The input data to be attacked; either 'fgsm', 'bim', 'jsma', 'cw'. ",
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)

