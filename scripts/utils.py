import os
import numpy as np
import foolbox
from progress.bar import Bar

import attacks_method.saliency as saliency
from attacks_method.iterative_projected_gradient import BIM, L2BasicIterativeAttack
import attacks_method.boundary_attack as DBA
import gl_var

def attack_for_input(model, x_input, y_input, method, dataset, save_path):
    """Gen attack costs for input data by input attack method

    Args:
        model: keras model
        input_data [str]: input data name
        method [str]: attack method for defense

    """  
    # * attack parameters
    if dataset == 'mnist':
        img_rows, img_cols, img_rgb = 28, 28, 1
        epsilon = 0.3
        stepsize = 0.0006
        epsilon_l2 = 0.1
        stepsize_l2 = 0.0002
    elif dataset == 'cifar':
        img_rows, img_cols, img_rgb = 32, 32, 3
        epsilon = 0.03
        stepsize = 0.00006
        epsilon_l2 = 0.01
        stepsize_l2 = 0.00002
    else:
        raise Exception('Input dataset error:', dataset)
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))

    f = open(save_path, 'w')

    # check the input bound
    assert np.min(x_input) == 0 and np.max(x_input) == 1
    x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, img_rgb).astype('float')

    bar = Bar('Attack processing', max=x_input.shape[0])
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
                    index=index+1,
                    size=x_input.shape[0],
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()
    f.close() # close the file


def get_single_detector_train_data(root_path, train_benign_path, train_adv_paths):
    # load benign training txt file
    train_benign_path = os.path.join(root_path, train_benign_path)
    with open(train_benign_path) as f:
        lines = f.read().splitlines()

    train_benign_lines_list = []
    for line in lines:
        try:
            train_benign_lines_list.append(int(line))
        except:
            raise('Invalid data type in training data:', str(line))
    assert len(train_benign_lines_list) == 1000

    # load adv training txt file
    train_adv_lines_list = []
    for train_adv_path in train_adv_paths:
        train_adv_path = os.path.join(root_path, train_adv_path)
        with open(train_adv_path) as f:
            lines = f.read().splitlines()

        counter = 0
        for line in lines:
            if counter < int(1000 / len(train_adv_paths)): # could change dataset ratio here
                try:
                    train_adv_lines_list.append(int(line))
                    counter += 1
                except:
                    raise('Invalid data type in training data:', line)
    assert len(train_adv_lines_list) == 1000

    data_train = train_benign_lines_list.copy()
    data_train.extend(train_adv_lines_list)
    data_train = np.array(data_train)
    data_train = np.expand_dims(data_train, axis=1)

    target_train = [0] * len(train_benign_lines_list)
    target_train.extend([1] * len(train_adv_lines_list))
    target_train = np.array(target_train)
    target_train = np.expand_dims(target_train, axis=1)
    print('training data generated, dimension are:', data_train.shape, target_train.shape)

    return data_train, target_train