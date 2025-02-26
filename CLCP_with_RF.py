import numpy as np 
from Split_Cross_CLCP import calc_clf_err, calc_clf_weighted_err, calc_fp_ratio, calc_set_mean_hamming_loss
from Split_Cross_CLCP import get_lhat_precise, make_prob_matrix_for_br
from sklearn.ensemble import RandomForestClassifier

def CLCP_with_RF_clf(X_train, y_train, X_test, y_test, 
                     alpha_ls = [0.01,0.05,0.1], delta_ls = [0.01,0.05,0.1],
                     multi_label = True,
                     loss_fun = calc_fp_ratio):

    np.random.seed(66)
    train_num = X_train.shape[0]
    random_ind = np.random.choice(train_num, train_num, replace = False)
    X_train_random = X_train[random_ind]
    y_train_random = y_train[random_ind]
    clf = RandomForestClassifier(oob_score = True)
    clf.fit(X_train_random, y_train_random)

    if multi_label:
        pred_proba_mat_cal = clf.oob_pred[:,1]
    else:
        pred_proba_mat_cal = clf.oob_pred[:,:,0]
        
    #lmda_ls = np.linspace(0, 1, 1000)
    lmda_ls = np.arange(0,1.002,0.001)

    calib_loss_table = np.ones((X_train_random.shape[0] + 1, len(lmda_ls)))
    for lmda_ind in range(len(lmda_ls)):
        lmda = lmda_ls[lmda_ind]
        img_ind = y_train_random
        pred_ind = ((pred_proba_mat_cal) >= lmda) * 1.0
        calib_loss_table[:-1,lmda_ind] = loss_fun(pred_ind, img_ind)
    
    pred_proba_test = clf.predict_proba(X_test)
    if multi_label:
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