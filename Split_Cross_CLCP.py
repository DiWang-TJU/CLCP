import numpy as np
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

def make_prob_matrix_for_br(pred_proba):
    re = None
    for tem in pred_proba:
        if re is None:
            re = tem[:,1].reshape((tem.shape[0], -1))
        else:
            re = np.hstack((re, tem[:,1].reshape((tem.shape[0], -1))))
    return re

# normal classification loss
def calc_clf_err(pred_ind_mat, label_mat):
    tem_label_mat = label_binarize(label_mat, classes = np.arange(len(np.unique(label_mat))).tolist())
    if tem_label_mat.shape[1] == 1:
        tem_label_mat = np.hstack((1 - tem_label_mat, tem_label_mat))
    return 1 - (pred_ind_mat * tem_label_mat).sum(1)

def calc_clf_weighted_err(pred_ind_mat, label_mat):
    np.random.seed(88)
    tem_label_mat = label_binarize(label_mat, classes = np.arange(len(np.unique(label_mat))).tolist())
    if tem_label_mat.shape[1] == 1:
        tem_label_mat = np.hstack((1 - tem_label_mat, tem_label_mat))
    weight_mat = np.repeat(np.random.uniform(size = tem_label_mat.shape[1]).reshape((1,-1)), tem_label_mat.shape[0], axis = 0)
    return (1 - (pred_ind_mat * tem_label_mat).sum(1)) * weight_mat[tem_label_mat == 1]

# normal regression loss
#def calc_reg_weighted_err(pred_interval_mat, label_mat):
#    np.random.seed(88)

# 分母全都改成除以输出个数，用于和降水实验统一
# multilabel classification loss
def calc_fp_ratio(pred_ind_mat, label_mat):
    return 1 - (pred_ind_mat * label_mat).sum(1)/label_mat.shape[1]

def calc_set_mean_hamming_loss(pred_ind_mat, label_mat):
    # hamming_loss treats 1 in pred_ind_mat as 1 or 0.
    return (((pred_ind_mat == 0) & (label_mat == 1)) * 1.0).sum(1)/label_mat.shape[1]

# 损失使用image regression论文中的每张图的损失
def calc_mean_cover(img, pred_upper, pred_lower):
    tem = (((img <= pred_upper) & (img >= pred_lower)) * 1.0)
    return 1 - tem.sum(1)/img.shape[1]

def get_lhat_precise(quantile_array, lambdas, target_loss):
    lhat_idx = max(np.argmax(quantile_array >= target_loss) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]

def get_lhat_precise_nonmono(quantile_array, lambdas, target_loss):
    try:
        return lambdas[np.where(quantile_array <= target_loss)[0][0]]
    except:
        print('meaningless results.......')
        return lambdas[0]
# 第二篇论文多标签损失，非单调的
def calc_hammingloss(pred_ind_mat, label_mat):
    return ((pred_ind_mat != label_mat) * 1.0).mean(1)

def calc_one_minus_precision(pred_ind_mat, label_mat):
    return 1 - (pred_ind_mat * label_mat).sum(1)/(pred_ind_mat.sum(1) + 1e-6)

def calc_one_minus_f1_score(pred_ind_mat, label_mat):
    precision = (pred_ind_mat * label_mat).sum(1)/(pred_ind_mat.sum(1) + 1e-6)
    recall = (pred_ind_mat * label_mat).sum(1)/label_mat.sum(1)
    f1_score = 2*precision*recall/(precision + recall)

    return 1 - f1_score

def Split_CLCP_clf(clf_class, para_dic, X_train, y_train, X_test, y_test, cal_rate = 0.2,
                   alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01, 0.05, 0.1],
                   multi_label = True,
                   use_make_prob_fun = True,
                   loss_fun = calc_fp_ratio):
    
    np.random.seed(66)
    if multi_label:
        X_train_proper, y_train_proper, X_cal, y_cal = iterative_train_test_split(X_train, y_train, test_size=cal_rate)
    else:
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=cal_rate, random_state=0, stratify=y_train)

    if para_dic is None:
        clf = eval(clf_class)
    else:
        clf = clf_class(**para_dic)
    clf.fit(X_train_proper, y_train_proper)

    pred_proba_cal = clf.predict_proba(X_cal)

    if use_make_prob_fun:
        pred_proba_cal = make_prob_matrix_for_br(pred_proba_cal)

    #lmda_ls = np.linspace(0,1.001,1000)
    lmda_ls = np.arange(0,1.002,0.001)

    calib_loss_table = np.ones((X_cal.shape[0]+1, len(lmda_ls)))

    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = y_cal
        pred_ind = ((pred_proba_cal)>=lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)

    pred_proba_test = clf.predict_proba(X_test)
    if use_make_prob_fun:
        pred_proba_test = make_prob_matrix_for_br(pred_proba_test)

    test_num = X_test.shape[0]
    re_loss_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    re_cls_num_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    for target_loss_ind, target_loss in enumerate(alpha_ls):
        for delta_ind, delta in enumerate(delta_ls):
            #print(delta)
            quantile_array = np.quantile(calib_loss_table, 1 - delta, axis = 0)
            lmda_final = get_lhat_precise_nonmono(quantile_array, lmda_ls, target_loss = target_loss)
            #print(delta, lmda_final)
            img_ind = y_test
            pred_ind = ((pred_proba_test) >= lmda_final) * 1.0
            pred_loss = loss_fun(pred_ind, img_ind)
            re_loss_mat[target_loss_ind, delta_ind] = pred_loss
            re_cls_num_mat[target_loss_ind, delta_ind] = pred_ind.sum(1)

    return {'re_loss_mat':re_loss_mat, 're_cls_num_mat':re_cls_num_mat}

def Cross_CLCP_clf(clf_class, para_dic, X_train, y_train, X_test, y_test, cv = 5,
                   alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01, 0.05, 0.1],
                   multi_label = True,
                   use_make_prob_fun = True,
                   loss_fun = calc_fp_ratio):
    
    np.random.seed(66)
    train_num = X_train.shape[0]
    random_ind = np.random.choice(train_num, train_num, replace = False)
    X_train_random = X_train[random_ind]
    y_train_random = y_train[random_ind]
    
    pred_proba_mat_cal = None
    label_cal = None

    if multi_label:
        k_fold = IterativeStratification(n_splits=cv, order=1)
    else:
        k_fold = StratifiedKFold(n_splits=cv)

    for train, cal in k_fold.split(X_train_random, y_train_random):
        if para_dic is None:
            clf = eval(clf_class)
        else:
            clf = clf_class(**para_dic)
        clf.fit(X_train_random[train], y_train_random[train])
        pred_proba_cal = clf.predict_proba(X_train_random[cal])

        if pred_proba_mat_cal is None:
            if use_make_prob_fun:
                pred_proba_mat_cal = make_prob_matrix_for_br(pred_proba_cal)
            else:
                pred_proba_mat_cal = pred_proba_cal
            label_cal = y_train_random[cal]
        else:
            if use_make_prob_fun:
                pred_proba_mat_cal = np.vstack((pred_proba_mat_cal, make_prob_matrix_for_br(pred_proba_cal)))
            else:
                pred_proba_mat_cal = np.vstack((pred_proba_mat_cal, pred_proba_cal))

            if multi_label:
                label_cal = np.vstack((label_cal, y_train_random[cal]))
            else:
                label_cal = np.concatenate((label_cal, y_train_random[cal]))

    #lmda_ls = np.linspace(0,1,1000)
    lmda_ls = np.arange(0,1.002,0.001)

    calib_loss_table = np.ones((X_train_random.shape[0]+1, len(lmda_ls)))

    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = label_cal
        pred_ind = ((pred_proba_mat_cal)>=lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)

    if para_dic is None:
        clf = eval(clf_class)
    else:
        clf = clf_class(**para_dic)
    clf.fit(X_train_random, y_train_random)
    pred_proba_test = clf.predict_proba(X_test)
    if use_make_prob_fun:
        pred_proba_test = make_prob_matrix_for_br(pred_proba_test)

    test_num = X_test.shape[0]
    re_loss_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    re_cls_num_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    for target_loss_ind, target_loss in enumerate(alpha_ls):
        for delta_ind, delta in enumerate(delta_ls):
            quantile_array = np.quantile(calib_loss_table, 1 - delta, axis = 0)
            lmda_final = get_lhat_precise(quantile_array, lmda_ls, target_loss = target_loss)
            img_ind = y_test
            pred_ind = ((pred_proba_test) >= lmda_final) * 1.0
            pred_loss = loss_fun(pred_ind, img_ind)
            re_loss_mat[target_loss_ind, delta_ind] = pred_loss
            re_cls_num_mat[target_loss_ind, delta_ind] = pred_ind.sum(1)

    return {'re_loss_mat':re_loss_mat, 're_cls_num_mat':re_cls_num_mat}

def Split_CLCP_clf_with_para_choosing(clf_class, para_dic, X_train, y_train, X_test, y_test, cal_rate = 0.2, para_cv = 3,
                                      alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01, 0.05, 0.1],
                                      multi_label = True,
                                      use_multi_output_class = False,
                                      use_make_prob_fun = True,
                                      loss_fun = calc_fp_ratio):
    
    np.random.seed(66)
    if multi_label:
        X_train_proper, y_train_proper, X_cal, y_cal = iterative_train_test_split(X_train, y_train, test_size=cal_rate)
    else:
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=cal_rate, random_state=0, stratify=y_train)

    if use_multi_output_class:
        clf_para = GridSearchCV(MultiOutputClassifier(clf_class()), para_dic, cv = para_cv)
        clf_para.fit(X_train_proper, y_train_proper)
        clf = MultiOutputClassifier(clf_class(probability = True))
        clf.estimator__C = clf_para.best_params_['estimator__C']
    else:
        clf_para = GridSearchCV(clf_class(), para_dic, cv = para_cv)
        clf_para.fit(X_train_proper, y_train_proper)
        clf = clf_class(**clf_para.best_params_)

    clf.fit(X_train_proper, y_train_proper)

    pred_proba_cal = clf.predict_proba(X_cal)

    if use_make_prob_fun:
        pred_proba_cal = make_prob_matrix_for_br(pred_proba_cal)

    #lmda_ls = np.linspace(0,1.001,1000)
    lmda_ls = np.arange(0,1.001,0.001)

    calib_loss_table = np.ones((X_cal.shape[0]+1, len(lmda_ls)))

    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = y_cal
        pred_ind = ((pred_proba_cal)>=lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)

    pred_proba_test = clf.predict_proba(X_test)
    if use_make_prob_fun:
        pred_proba_test = make_prob_matrix_for_br(pred_proba_test)

    test_num = X_test.shape[0]
    re_loss_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    re_cls_num_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    for target_loss_ind, target_loss in enumerate(alpha_ls):
        for delta_ind, delta in enumerate(delta_ls):
            #print(delta)
            quantile_array = np.quantile(calib_loss_table, 1 - delta, axis = 0)
            lmda_final = get_lhat_precise(quantile_array, lmda_ls, target_loss = target_loss)
            #print(delta, lmda_final)
            img_ind = y_test
            pred_ind = ((pred_proba_test) >= lmda_final) * 1.0
            pred_loss = loss_fun(pred_ind, img_ind)
            re_loss_mat[target_loss_ind, delta_ind] = pred_loss
            re_cls_num_mat[target_loss_ind, delta_ind] = pred_ind.sum(1)

    return {'re_loss_mat':re_loss_mat, 're_cls_num_mat':re_cls_num_mat}

################################# prediction_set_from_null_to_full ####################################
def calc_real_fpr(pred_ind, img_ind):
    return (pred_ind.sum(1) - (pred_ind * img_ind).sum(1))/img_ind.shape[1]

def Split_CLCP_clf_with_para_choosing_for_real_fpr(clf_class, para_dic, X_train, y_train, X_test, y_test, cal_rate = 0.2, para_cv = 3,
                                                   alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01, 0.05, 0.1],
                                                   multi_label = True,
                                                   use_multi_output_class = False,
                                                   use_make_prob_fun = True,
                                                   loss_fun = calc_real_fpr):
    
    np.random.seed(66)
    if multi_label:
        X_train_proper, y_train_proper, X_cal, y_cal = iterative_train_test_split(X_train, y_train, test_size=cal_rate)
    else:
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=cal_rate, random_state=0, stratify=y_train)

    if use_multi_output_class:
        clf_para = GridSearchCV(MultiOutputClassifier(clf_class()), para_dic, cv = para_cv)
        clf_para.fit(X_train_proper, y_train_proper)
        clf = MultiOutputClassifier(clf_class(probability = True))
        clf.estimator__C = clf_para.best_params_['estimator__C']
    else:
        clf_para = GridSearchCV(clf_class(), para_dic, cv = para_cv)
        clf_para.fit(X_train_proper, y_train_proper)
        clf = clf_class(**clf_para.best_params_)

    clf.fit(X_train_proper, y_train_proper)

    pred_proba_cal = clf.predict_proba(X_cal)

    if use_make_prob_fun:
        pred_proba_cal = make_prob_matrix_for_br(pred_proba_cal)

    #lmda_ls = np.linspace(0,1.001,1000)
    lmda_ls = np.arange(0,1.001,0.001)

    calib_loss_table = np.ones((X_cal.shape[0]+1, len(lmda_ls)))

    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = y_cal
        pred_ind = ((pred_proba_cal)>=1 - lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)

    pred_proba_test = clf.predict_proba(X_test)
    if use_make_prob_fun:
        pred_proba_test = make_prob_matrix_for_br(pred_proba_test)

    test_num = X_test.shape[0]
    re_loss_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    re_cls_num_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    for target_loss_ind, target_loss in enumerate(alpha_ls):
        for delta_ind, delta in enumerate(delta_ls):
            #print(delta)
            quantile_array = np.quantile(calib_loss_table, 1 - delta, axis = 0)
            lmda_final = get_lhat_precise(quantile_array, lmda_ls, target_loss = target_loss)
            #print(delta, lmda_final)
            img_ind = y_test
            pred_ind = ((pred_proba_test) >= 1 - lmda_final) * 1.0
            pred_loss = loss_fun(pred_ind, img_ind)
            re_loss_mat[target_loss_ind, delta_ind] = pred_loss
            re_cls_num_mat[target_loss_ind, delta_ind] = pred_ind.sum(1)

    return {'re_loss_mat':re_loss_mat, 're_cls_num_mat':re_cls_num_mat}

def Split_CLCP_clf_for_real_fpr(clf, X_train, y_train, X_test, y_test, cal_rate = 0.2, para_cv = 3,
                                alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01, 0.05, 0.1],
                                multi_label = True,
                                use_multi_output_class = False,
                                use_make_prob_fun = True,
                                loss_fun = calc_real_fpr):
    
    np.random.seed(66)
    if multi_label:
        X_train_proper, y_train_proper, X_cal, y_cal = iterative_train_test_split(X_train, y_train, test_size=cal_rate)
    else:
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=cal_rate, random_state=0, stratify=y_train)

    clf.fit(X_train_proper, y_train_proper)

    pred_proba_cal = clf.predict_proba(X_cal)

    if use_make_prob_fun:
        pred_proba_cal = make_prob_matrix_for_br(pred_proba_cal)

    #lmda_ls = np.linspace(0,1.001,1000)
    lmda_ls = np.arange(0,1.001,0.001)

    calib_loss_table = np.ones((X_cal.shape[0]+1, len(lmda_ls)))

    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = y_cal
        pred_ind = ((pred_proba_cal)>=1 - lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)

    pred_proba_test = clf.predict_proba(X_test)
    if use_make_prob_fun:
        pred_proba_test = make_prob_matrix_for_br(pred_proba_test)

    test_num = X_test.shape[0]
    re_loss_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    re_cls_num_mat = np.zeros((len(alpha_ls), len(delta_ls), test_num))
    for target_loss_ind, target_loss in enumerate(alpha_ls):
        for delta_ind, delta in enumerate(delta_ls):
            #print(delta)
            quantile_array = np.quantile(calib_loss_table, 1 - delta, axis = 0)
            lmda_final = get_lhat_precise(quantile_array, lmda_ls, target_loss = target_loss)
            #print(delta, lmda_final)
            img_ind = y_test
            pred_ind = ((pred_proba_test) >= 1 - lmda_final) * 1.0
            pred_loss = loss_fun(pred_ind, img_ind)
            re_loss_mat[target_loss_ind, delta_ind] = pred_loss
            re_cls_num_mat[target_loss_ind, delta_ind] = pred_ind.sum(1)

    return {'re_loss_mat':re_loss_mat, 're_cls_num_mat':re_cls_num_mat}