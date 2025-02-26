import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from Split_Cross_CLCP import Split_CLCP_clf, Cross_CLCP_clf, Split_CLCP_clf_with_para_choosing
from Split_Cross_CLCP import calc_clf_err, calc_clf_weighted_err, calc_fp_ratio, calc_set_mean_hamming_loss
from CLCP_with_RF import CLCP_with_RF_clf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# whether to collect results from scratch
collect_results_from_scratch = False

if collect_results_from_scratch:
    np.random.seed(66)
    warnings.filterwarnings('ignore')
    from_dir = r'data\classification'
    
    experiment_name = 'weighted_classification'
    dataset_name_ls = os.listdir(from_dir)
    
    test_rate = 0.2
    exp_times = 10
    
    alpha_ls = [0.1,0.2]
    delta_ls = [0.05, 0.1, 0.15, 0.2]
    
    def collect_clf_results(clf_str, alpha_ls, delta_ls):
        re1 = Split_CLCP_clf(clf_str, X_train, y_train, X_test, y_test, alpha_ls = alpha_ls, delta_ls = delta_ls, multi_label = False, use_make_prob_fun = False,
                             loss_fun = calc_clf_weighted_err)
    
        re2 = Cross_CLCP_clf(clf_str, X_train, y_train, X_test, y_test, cv = 10, alpha_ls = alpha_ls, delta_ls = delta_ls, multi_label = False, use_make_prob_fun = False,
                             loss_fun = calc_clf_weighted_err)
        
        return re1, re2
    
    def collect_clf_results_split(clf_str, para_dic, alpha_ls, delta_ls, use_make_prob_fun = False):
        re1 = Split_CLCP_clf_with_para_choosing(clf_str, para_dic, X_train, y_train, X_test, y_test, alpha_ls = alpha_ls, delta_ls = delta_ls, multi_label = False, use_make_prob_fun = use_make_prob_fun,
                                                loss_fun = calc_clf_weighted_err)
        
        return re1
    
    class obtain_summary_results():
        def __init__(self, alpha_ls, delta_ls):
            self.re_loss_mat = None
            self.re_cls_num_mat = None
            self.alpha_ls = alpha_ls
            self.delta_ls = delta_ls
            
        def update(self, re_dic):
            if self.re_loss_mat is None:
                self.re_loss_mat = re_dic['re_loss_mat']
                self.re_cls_num_mat = re_dic['re_cls_num_mat']
            else:
                self.re_loss_mat = np.concatenate((self.re_loss_mat, re_dic['re_loss_mat']), axis = -1)
                self.re_cls_num_mat = np.concatenate((self.re_cls_num_mat, re_dic['re_cls_num_mat']), axis = -1)
                
        def calc_final_re_mat(self):
            final_loss_mat = np.zeros((len(self.alpha_ls), len(self.delta_ls)))
            final_cls_num_mat = np.zeros((len(self.alpha_ls), len(self.delta_ls)))
            
            for alpha_ind, alpha in enumerate(self.alpha_ls):
                for delta_ind, delta in enumerate(self.delta_ls):
                    final_loss_mat[alpha_ind, delta_ind] = ((self.re_loss_mat[alpha_ind, delta_ind] > alpha) * 1.0).mean()
                    final_cls_num_mat[alpha_ind, delta_ind] = self.re_cls_num_mat[alpha_ind, delta_ind].mean()
                    
            return {'final_loss_mat':final_loss_mat, 'final_cls_num_mat':final_cls_num_mat}
    
    SVM_para_dic = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'probability': [True]}
    MLP_para_dic = {'learning_rate_init': [0.001, 0.0001], 'max_iter': [200, 500, 1000]}
    RF_para_dic = {'n_estimators': [100, 300, 500], 'criterion':['gini', 'entropy']}
    for data_name_ind, dataset_name in enumerate(dataset_name_ls):
    
        SVM_split_re = obtain_summary_results(alpha_ls, delta_ls)
        MLP_split_re = obtain_summary_results(alpha_ls, delta_ls)
        RF_split_re = obtain_summary_results(alpha_ls, delta_ls)
        
        data = pd.read_csv(os.path.join(from_dir, dataset_name), header = None)
    
        train_feature = data.values[:,:-1]
        train_label = data.values[:,-1]
    
        enc = preprocessing.OrdinalEncoder()
        label_transformed = enc.fit_transform(train_label.reshape((-1,1)))
    
        X_total = preprocessing.MinMaxScaler().fit_transform(train_feature)
        y_total = label_transformed.reshape(-1)
    
        for counts in range(exp_times):
            print('%s: %s ......'%(dataset_name[:-4], str(counts)))
    
            X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=test_rate, stratify=y_total)
    
            SVM_split= collect_clf_results_split(SVC, SVM_para_dic, alpha_ls = alpha_ls, delta_ls = delta_ls)
            SVM_split_re.update(SVM_split)
    
            MLP_split= collect_clf_results_split(MLPClassifier, MLP_para_dic, alpha_ls = alpha_ls, delta_ls = delta_ls, use_make_prob_fun = False)
            MLP_split_re.update(MLP_split)
    
            RF_split = collect_clf_results_split(RandomForestClassifier, RF_para_dic, alpha_ls = alpha_ls, delta_ls = delta_ls)
            RF_split_re.update(RF_split)
    
        SVM_split_re_dic = SVM_split_re.calc_final_re_mat()
        MLP_split_re_dic = MLP_split_re.calc_final_re_mat()
        RF_split_re_dic = RF_split_re.calc_final_re_mat()
    
        
        re_dic_ls = [SVM_split_re_dic, MLP_split_re_dic, RF_split_re_dic]
        
        clf_name_ls = ['SVM_split', 'MLP_split', 'RF_split']
        clf_name_ls = np.array(clf_name_ls).reshape((-1,1))
        re_alpha_ls = []
        re_delta_ls = []
        
        for alpha_ind, alpha in enumerate(alpha_ls):
            tem_loss_mat = None
            tem_len_mat = None
            for tem_re_dic in re_dic_ls:
                if tem_loss_mat is None:
                    tem_loss_mat = tem_re_dic['final_loss_mat'][alpha_ind].reshape((1,-1))
                    tem_len_mat = tem_re_dic['final_cls_num_mat'][alpha_ind].reshape((1,-1))
                else:
                    tem_loss_mat = np.vstack((tem_loss_mat, tem_re_dic['final_loss_mat'][alpha_ind].reshape((1,-1))))
                    tem_len_mat = np.vstack((tem_len_mat, tem_re_dic['final_cls_num_mat'][alpha_ind].reshape((1,-1))))
                    
            to_save_loss_mat = np.hstack((clf_name_ls, tem_loss_mat))
            to_save_len_mat = np.hstack((clf_name_ls, tem_len_mat))
            
            if not os.path.exists(os.path.join('save_results_final_design', 'csv_results', experiment_name, dataset_name[:-4])):
                os.makedirs(os.path.join('save_results_final_design', 'csv_results', experiment_name, dataset_name[:-4]))
            
            pd.DataFrame(to_save_loss_mat).to_csv(os.path.join('save_results_final_design', 'csv_results', experiment_name, dataset_name[:-4], dataset_name[:-4] + '_%s_err_rate.csv'%(str(alpha))), header = None, index = None)
            pd.DataFrame(to_save_len_mat).to_csv(os.path.join('save_results_final_design', 'csv_results',experiment_name, dataset_name[:-4], dataset_name[:-4] + '_%s_len.csv'%(str(alpha))), header = None, index = None)

# draw images in the paper
from_dir = r'save_results_final_design\csv_results\weighted_classification'
dataset_name_ls = os.listdir(from_dir)

# images for validity
alpha_all = [0.1, 0.2]
total_tensor_ls = []
for alpha_ind, alpha in enumerate(alpha_all):
    total_tensor = None
    for dataset_name in dataset_name_ls:
        tem = pd.read_csv(os.path.join(from_dir, dataset_name, '%s_%s_err_rate.csv'%(dataset_name, str(alpha))), header = None).values
        if total_tensor is None:
            total_tensor = tem.reshape((1, tem.shape[0], tem.shape[1]))
        else:
            total_tensor = np.concatenate((total_tensor, tem.reshape((1, tem.shape[0], tem.shape[1]))), axis = 0)
    total_tensor_ls.append(total_tensor.reshape(([1] + list(total_tensor.shape))))

total_tensor_ls = np.concatenate(total_tensor_ls, axis = 0)

data_name_ls = []
delta_ls = []
alpha_ls = []
clf_ls = []
err_rate_ls = []

delta_all = [0.05, 0.1, 0.15, 0.2]
clf_all = total_tensor_ls[0,0,:,0]
clf_all = ['SVM', 'NN', 'RF']
dataset_name_ls = ['bc-wisc-diag','car','chess-kr-kp','contrac','credit-a','credit-g','ctg-10classes','ctg-3classes',
                  'haberman','optical','phishing-web','st-image','st-landsat','tic-tac-toe','wall-following','waveform','waveform-noise',
                  'wilt','wine-quality-red','wine-quality-white']

for alpha_ind, alpha in enumerate(alpha_all):
    for data_name_ind in range(len(dataset_name_ls)):
        for p in range(len(delta_all)):
            for q in range(len(clf_all)):
                data_name_ls.append(dataset_name_ls[data_name_ind])
                delta_ls.append(delta_all[p])
                clf_ls.append(clf_all[q])
                err_rate_ls.append(total_tensor_ls[alpha_ind, data_name_ind,q,p+1])
                alpha_ls.append(alpha)


re_dic = {}
re_dic['$\delta$'] = delta_ls
re_dic['Classifier'] = clf_ls
re_dic['Frequency of Loss > $\\alpha$'] = err_rate_ls
re_dic['Dataset'] = data_name_ls
re_dic['$\\alpha$'] = alpha_ls
re_df = pd.DataFrame(re_dic)

sns.set_style('darkgrid')
sns.set(font_scale = 1.2)
tem_re_df = re_df
plt.figure(figsize = (20,10))
sns.catplot(
    data=tem_re_df, kind="bar", col = 'Classifier',row = '$\\alpha$',
    x="$\delta$", y="Frequency of Loss > $\\alpha$", hue="Dataset", palette="dark"
)
plt.yticks(np.arange(0,0.35, 0.05))
plt.ylim((0, 0.3))
plt.savefig(r'save_images\classification_validity.jpg', dpi = 300)
plt.show()

# images for efficiency
dataset_name_ls = os.listdir(from_dir)
total_tensor_ls = []
for alpha_ind, alpha in enumerate(alpha_all):
    total_tensor = None
    for dataset_name in dataset_name_ls:
        tem = pd.read_csv(os.path.join(from_dir, dataset_name, '%s_%s_len.csv'%(dataset_name, str(alpha))), header = None).values
        if total_tensor is None:
            total_tensor = tem.reshape((1, tem.shape[0], tem.shape[1]))
        else:
            total_tensor = np.concatenate((total_tensor, tem.reshape((1, tem.shape[0], tem.shape[1]))), axis = 0)
    total_tensor_ls.append(total_tensor.reshape(([1] + list(total_tensor.shape))))

total_tensor_ls = np.concatenate(total_tensor_ls, axis = 0)

data_name_ls = []
delta_ls = []
alpha_ls = []
clf_ls = []
err_rate_ls = []

delta_all = [0.05, 0.1, 0.15, 0.2]
clf_all = total_tensor_ls[0,0,:,0]
clf_all = ['SVM', 'NN', 'RF']
dataset_name_ls = ['bc-wisc-diag','car','chess-kr-kp','contrac','credit-a','credit-g','ctg-10classes','ctg-3classes',
                  'haberman','optical','phishing-web','st-image','st-landsat','tic-tac-toe','wall-following','waveform','waveform-noise',
                  'wilt','wine-quality-red','wine-quality-white']

for alpha_ind, alpha in enumerate(alpha_all):
    for data_name_ind in range(len(dataset_name_ls)):
        for p in range(len(delta_all)):
            for q in range(len(clf_all)):
                data_name_ls.append(dataset_name_ls[data_name_ind])
                delta_ls.append(delta_all[p])
                clf_ls.append(clf_all[q])
                err_rate_ls.append(total_tensor_ls[alpha_ind, data_name_ind,q,p+1])
                alpha_ls.append(alpha)

re_dic = {}
re_dic['$\delta$'] = delta_ls
re_dic['Classifier'] = clf_ls
re_dic['Average Size'] = err_rate_ls
re_dic['Dataset'] = data_name_ls
re_dic['$\\alpha$'] = alpha_ls
re_df = pd.DataFrame(re_dic)
tem_re_df = re_df
sns.catplot(
    data=tem_re_df, kind="bar", col = 'Classifier',row = '$\\alpha$',
    x="$\delta$", y="Average Size", hue="Dataset", palette="dark"
)
plt.savefig(r'save_images\classification_efficiency.jpg', dpi = 300)
plt.show()