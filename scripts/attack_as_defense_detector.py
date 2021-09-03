import sys
sys.path.append('../')
import numpy as np
import foolbox
import os
import argparse

from keras.layers import Input
from keras.models import load_model

from detect.util import get_data
from utils import attack_for_input, get_single_detector_train_data


def generate_training_data(model, dataset):
    """ 
    Generate adversarial examples for training,
    as we use the test set to calc auroc, 
    now we use the training set to generate new adversarial examples.
    """  

    # use training set to generate adv
    x_input, y_input, _, _ = get_data('mnist') # the input data min-max -> 0.0 1.0
    
    # except wrong label data
    preds_test = model.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input.argmax(axis=1))[0]
    print('The model accuracy is', len(inds_correct) / len(x_input))
    x_input = x_input[inds_correct]
    y_input = y_input[inds_correct]
    
    # use foolbox to generate 4 kinds of adversarial examples
    # fgsm, bim-a, jsma and cw
    # as the baseline uses the 4 kinds of adv to test
    # we just use the default parameters to generate adv
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
    adversarial_result = []

    training_data_folder = '../results/detector/'
    if not os.path.exists(training_data_folder):
        os.makedirs(training_data_folder)

    # just generate adversarial examples by untargeted attacks
    criterion = foolbox.criteria.Misclassification()

    # we prefer to use fgsm，bim，jsma and c&w
    # but due to the high time overhead of c&w(as binary search
    # we use deepfool attack to replace it here
    # in order to reproduce our method efficiently
    adv_types_list = ['fgsm', 'bim', 'jsma', 'df']

    adversarial_result = []
    input_data = enumerate(zip(x_input, y_input))
    for adv_type in adv_types_list:
        counter = 1
        if adv_type == 'fgsm':
            attack = foolbox.attacks.FGSM(fmodel, criterion)
        elif adv_type == 'jsma':
            attack = foolbox.attacks.saliency.SaliencyMapAttack(fmodel, criterion)
        elif adv_type == 'cw':
            attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, criterion)
        elif adv_type == 'bim':
            attack = foolbox.attacks.BIM(fmodel, criterion, distance=foolbox.distances.Linfinity)
        elif adv_type == 'df':
            attack = foolbox.attacks.DeepFoolAttack(fmodel, criterion)
        elif adv_type == 'lsa':
            attack = foolbox.attacks.LocalSearchAttack(fmodel, criterion)
        elif adv_type == 'dba':
            attack = foolbox.attacks.BoundaryAttack(fmodel, criterion)
        else:
            raise Exception('Unkonwn attack method:', str(adv_type))
        
        # only need 1000 train adv samples
        # so 250 adv examples for each attack is enough
        # hard coding the bound
        while counter <= 250:
            # 使用training set
            _, (img, label) = next(input_data)
            label = np.argmax(label)

            if adv_type == 'fgsm':
                if dataset == 'mnist':
                    adversarial = attack(img, label, epsilons=1, max_epsilon=0.3)
                else:
                    adversarial = attack(img, label, epsilons=1, max_epsilon=0.03) 
                    # adversarial = attack(img, label, binary_search=True, epsilon=0.3, stepsize=0.05, iterations=10
            elif adv_type == 'bim':
                if dataset == 'mnist':
                    adversarial = attack(img, label, binary_search=False,
                                         epsilon=0.3, stepsize=0.03, iterations=10)
                else:
                    adversarial = attack(img, label, binary_search=False,
                                         epsilon=0.03, stepsize=0.003, iterations=10)
            else:
                adversarial = attack(img, label)

            if adversarial is not None:
                print('\r attack success:', counter, end="")
                adv_label = np.argmax(fmodel.predictions(adversarial))
                if adv_label != label:
                    adversarial_result.append(adversarial)
                    counter += 1

        print('\n%s attack finished.' % adv_type)
        file_name = '%s_%s_adv_examples.npy' % (dataset, adv_type)
        np.save(training_data_folder + '/' + file_name, np.array(adversarial_result))
        adversarial_result = []


def generate_attack_cost(kmodel, dataset, attack_method):
    """ 
    Generate attack costs for benign and adversarial examples.
    """  
    
    x_input, y_input, _, _ = get_data(dataset) # the input data min-max -> 0.0 1.0
    
    # except wrong label data
    preds_test = kmodel.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input.argmax(axis=1))[0]
    x_input = x_input[inds_correct]
    x_input = x_input[0: 1000]
    save_path = '../results/detector/' + dataset + '_benign'
    attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)

    for adv_type in ['fgsm', 'bim', 'jsma', 'df']:
        training_data_folder = '../results/detector/'
        file_name = '%s_%s_adv_examples.npy' % (dataset, adv_type)
        x_input = np.load(training_data_folder + file_name)
        save_path = '../results/detector/' + attack_method + '/' + dataset + '_' + adv_type
        attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)


def main(args):
    dataset = args.dataset
    detector_type = args.detector 
    attack_method = args.attack

    # * load model
    kmodel = load_model('../data/model_%s.h5' % dataset)
    
    if args.init:
        print('generate training data...')
        # ! Step 1. Generate training adv
        # * use this method to generate 1000 adversarial examples from training set as training set
        generate_training_data(kmodel, dataset)
        
        # ! Step 2. Get the attack costs of training adv
        # * use this method to obtain atatck costs on training data
        generate_attack_cost(kmodel, dataset, attack_method)
    
    # ! Step 3. Train and pred
    root_path = '../results/'
    train_benign_data_path = 'detector/' + args.attack + '/' + args.dataset + '_benign'
    adv_types_list = ['fgsm', 'bim', 'jsma', 'df']
    test_types_list = ['benign', 'fgsm', 'bim-a', 'cw', 'jsma']
    train_adv_data_paths = []
    for adv_type in adv_types_list:
        train_adv_data_paths.append('detector/' + args.attack + '/' + args.dataset + '_' + adv_type)

    if detector_type == 'knn':
        # * train k-nearest neighbors detector
        from sklearn import neighbors
        N = 100
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=N)
        print('knn based detector, k value is:', N)

        data_train, target_train = get_single_detector_train_data(root_path, train_benign_data_path, train_adv_data_paths)
        knn_model.fit(data_train, target_train)
        print('training acc:', knn_model.score(data_train, target_train))

        # * test k-nearest neighbors detector
        for test_type in test_types_list:
            test_path = args.dataset + '_attack_iter_stats/' + args.attack + '_attack_' + test_type
            test_path = os.path.join(root_path, test_path)
            with open(test_path) as f:
                lines = f.read().splitlines()
            test_lines_list = []
            for line in lines:
                try:
                    test_lines_list.append(int(line))
                except:
                    raise('Invalid data type in test data:', line)
            assert len(test_lines_list) == 1000

            test_lines_list = np.expand_dims(test_lines_list, axis=1)
            result = knn_model.predict(test_lines_list)
            acc = sum(result) / len(result)
            if test_type is 'benign':
                print('For knn based detector, detect acc on %s samples is %.4f'%(test_type, 1-acc))
            else:
                print('For knn based detector, detect acc on %s samples is %.4f'%(test_type, acc))
    else:
        # * z-score based detector
        from scipy import stats
        from scipy.special import inv_boxcox
        from scipy.stats import normaltest

        data_train, _ = get_single_detector_train_data(root_path, train_benign_data_path, train_adv_data_paths)
        data_train = data_train[:1000].reshape(1, len(data_train[:1000]))[0] # only need benign data
        data_train[data_train == 0] = 0.01
        _, p_value = normaltest(data_train)
        z_score_thrs = 1.281552
        arr_mean = np.mean(data_train)
        arr_std = np.std(data_train)

        if p_value < 0.05:
            print('raw attack cost dist not pass the normal test, need boxcox')
            _, lmd = stats.boxcox(data_train)
            thrs_min = arr_mean - z_score_thrs * arr_std
            thrs_min = inv_boxcox(thrs_min, lmd)
        else:
            print('raw attack cost dist pass the normal test')
            thrs_min = arr_mean - z_score_thrs * arr_std

        # * test z-score detector
        for test_type in test_types_list:
            test_path = args.dataset + '_attack_iter_stats/' + args.attack + '_attack_' + test_type
            test_path = os.path.join(root_path, test_path)
            with open(test_path) as f:
                lines = f.read().splitlines()
            test_lines_list = []
            for line in lines:
                try:
                    test_lines_list.append(int(line))
                except:
                    raise('Invalid data type in test data:', line)
            assert len(test_lines_list) == 1000

            result = [1 if _ < thrs_min else 0 for _ in test_lines_list] # for ensemble detector, need np.intersect1d and np.union1d
            acc = sum(result) / len(result)
            if test_type is 'benign':
                print('For z-score based detector, detect acc on %s samples is %.4f'%(test_type, 1-acc))
            else:
                print('For z-score based detector, detect acc on %s samples is %.4f'%(test_type, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help="Dataset to use; either 'mnist', 'cifar'",
        required=False, type=str, default='mnist',
    )
    parser.add_argument(
        '--init',
        help="If this is the first time to run this script, \
            need to generate the training data and attack costs.",
        action='store_true',
    )
    parser.add_argument(
        '-d', '--detector',
        help="Detetor type; either 'knn' or 'zscore'.",
        required=True, type=str,
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; recommanded to use JSMA, BIM or BIM2.",
        required=True, type=str,
    )
    args = parser.parse_args()
    assert args.detector in ['knn', 'zscore'], "Detector parameter error"
    assert args.attack in ['JSMA', 'BIM', 'BIM2'], "Attack parameter error"
    
    args = parser.parse_args()
    main(args)