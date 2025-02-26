import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# draw images in the paper. re_dic_plclf.pth and re_dic_plreg.pth are calculated based on the prediction fields of neural nets
# high or low temperature
re_dic = torch.load(r'save_results_final_design\save_re_dic_for_weather_forecasting\re_dic_plclf.pth')

dataset_name_ls = ['temperature_max_24h', 'temperature_min_24h']
model_name_ls = ['deepnet', 'unet']

dataset_name_map = {'temperature_max_24h': 'HighTemp', 'temperature_min_24h': 'LowTemp'}
model_name_map = {'deepnet':'nDNN', 'unet':'U-Net'}
alpha_ls = [0.05, 0.1, 0.15, 0.2]
delta_ls = [0.05, 0.1, 0.15, 0.2]

df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []
df_err_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                df_dataset_ls.append(dataset_name_map[dataset_name])
                df_model_ls.append(model_name_map[model_name])
                df_alpha_ls.append(alpha)
                df_delta_ls.append(delta)
                df_loss_ls.append((re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind].reshape(-1) > alpha).mean())
                df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind].mean())
                
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Frequency of Loss > $\\alpha$'] = df_loss_ls
re_df['Average Size'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

plt.figure(figsize = (20,10))
sns.set_style('darkgrid')
sns.set(font_scale = 1.6)

sns.catplot(x="$\delta$", y="Frequency of Loss > $\\alpha$",kind="bar", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)
plt.yticks(np.arange(0,0.35, 0.05))
plt.ylim((0, 0.3))

plt.savefig(r'save_images\pixel_classification_err_rate.jpg', dpi = 300)


df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                for p in range(10):
                    for q in range(re_dic[dataset_name][model_name]['re_mat'].shape[-1]):
                        df_dataset_ls.append(dataset_name_map[dataset_name])
                        df_model_ls.append(model_name_map[model_name])
                        df_alpha_ls.append(alpha)
                        df_delta_ls.append(delta)
                        df_loss_ls.append(re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind, p, q])
                        df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind, p, q])
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Loss'] = df_loss_ls
re_df['Average Size'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

sns.catplot(x="$\delta$", y="Loss",kind="boxen", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)

plt.savefig(r'save_images\pixel_classification_loss.jpg', dpi = 300)

df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                for p in range(10):
                    for q in range(re_dic[dataset_name][model_name]['re_mat'].shape[-1]):
                        df_dataset_ls.append(dataset_name_map[dataset_name])
                        df_model_ls.append(model_name_map[model_name])
                        df_alpha_ls.append(alpha)
                        df_delta_ls.append(delta)
                        df_loss_ls.append(re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind, p, q])
                        df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind, p, q])
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Loss'] = df_loss_ls
re_df['Normalized Size'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

sns.catplot(x="$\delta$", y="Normalized Size",kind="boxen", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)

plt.savefig(r'save_images\pixel_classification_efficiency.jpg', dpi = 300)


# maximum or minimum temperature
re_dic = torch.load(r'save_results_final_design\save_re_dic_for_weather_forecasting\re_dic_plreg.pth')

df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []
df_err_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                df_dataset_ls.append(dataset_name_map[dataset_name])
                df_model_ls.append(model_name_map[model_name])
                df_alpha_ls.append(alpha)
                df_delta_ls.append(delta)
                df_loss_ls.append((re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind].reshape(-1) > alpha).mean())
                df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind].mean())
                
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Frequency of Loss > $\\alpha$'] = df_loss_ls
re_df['Average Size'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

sns.catplot(x="$\delta$", y="Frequency of Loss > $\\alpha$",kind="bar", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)
plt.yticks(np.arange(0,0.35, 0.05))
plt.ylim((0, 0.3))

plt.savefig(r'save_images\pixel_regression_err_rate.jpg', dpi = 300)

df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                for p in range(10):
                    for q in range(re_dic[dataset_name][model_name]['re_mat'].shape[-1]):
                        df_dataset_ls.append(dataset_name_map[dataset_name])
                        df_model_ls.append(model_name_map[model_name])
                        df_alpha_ls.append(alpha)
                        df_delta_ls.append(delta)
                        df_loss_ls.append(re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind, p, q])
                        df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind, p, q])
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Loss'] = df_loss_ls
re_df['Average Size'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

sns.catplot(x="$\delta$", y="Loss",kind="boxen", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)

plt.savefig(r'save_images\pixel_regression_loss.jpg', dpi = 300)

df_dataset_ls = []
df_model_ls = []
df_alpha_ls = []
df_delta_ls = []
df_loss_ls = []
df_len_ls = []

for dataset_name in dataset_name_ls:
    for model_name in model_name_ls:
        for alpha_ind, alpha in enumerate(alpha_ls):
            for delta_ind, delta in enumerate(delta_ls):
                for p in range(10):
                    for q in range(re_dic[dataset_name][model_name]['re_mat'].shape[-1]):
                        df_dataset_ls.append(dataset_name_map[dataset_name])
                        df_model_ls.append(model_name_map[model_name])
                        df_alpha_ls.append(alpha)
                        df_delta_ls.append(delta)
                        df_loss_ls.append(re_dic[dataset_name][model_name]['re_mat'][alpha_ind, delta_ind, p, q])
                        df_len_ls.append(re_dic[dataset_name][model_name]['len_mat'][alpha_ind, delta_ind, p, q])
                        
re_df = {}
re_df['$\delta$'] = df_delta_ls
re_df['$\\alpha$'] = df_alpha_ls
re_df['Loss'] = df_loss_ls
re_df['Average Interval Length'] = df_len_ls
re_df['Dataset'] = df_dataset_ls
re_df['Model'] = df_model_ls
re_df = pd.DataFrame(re_df)

sns.catplot(x="$\delta$", y="Average Interval Length",kind="boxen", col = '$\\alpha$',row = 'Dataset',
            hue="Model",
            data=re_df)

plt.savefig(r'save_images\pixel_regression_efficiency.jpg', dpi = 300)




