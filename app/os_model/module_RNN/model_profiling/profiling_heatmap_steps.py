import pandas as pd
import numpy as np
from app.os_model.visualizer import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
#import statsmodels.api as sm
import seaborn as sns
#sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x



#models = ["SLP_0.1", "MLP_0.1", "CNN_0.1", "LSTM_0.1", "BiLSTM_0.1", "BiLSTM_Attn_0.1", "LSTM_Attn_0.1", "BiLSTM_Attn2_0.1", "LSTM_Attn2_0.1",
#          "SLP_0.0", "MLP_0.0", "CNN_0.0", "LSTM_0.0", "BiLSTM_0.0", "BiLSTM_Attn_0.0", "LSTM_Attn_0.0", "BiLSTM_Attn2_0.0", "LSTM_Attn2_0.0"]
models = ["LSTM_Attn2_0.0", "BiLSTM_Attn2_0.0", "LSTM_0.0", "BiLSTM_0.0"]
#models = ["LSTM_0.0"]

sub_ids = ["sangwan"]
names = ["(0.0)", "(0.18)", "(0.36)",
         "(0.54)", "(0.72)"] # primacy

#names = ["(0.05)", "(0.1)", "(0.2)",
#         "(0.3)", "(0.4)"] # lr

#names = ["_RAND", "_PRIMACY(-1.0)", "_RECENCY(-1.0)",
#         "_LR(2.0)", "_OS"] # deviates


def correl_func(var_tar, model, seed, session_idx= None) :
    tot_out_correls = []
    tot_out_ps = []

    for outer_level, name in enumerate(names):
        if session_idx is None :
            fname = "{}/profiling_exp_{}_{}_{}_{}.csv".format(var_tar, model, var_tar, outer_level, seed)
        else :

            fname = "{}/profiling_exp_{}_{}_{}_{}_{}.csv".format(var_tar, model, var_tar, outer_level, session_idx, seed)

        df = pd.read_csv(fname)

        out_correls = []
        out_ps = []

        for sub_id in sub_ids:
            df_sub = df.loc[(df.Sub_id == sub_id)];

            out0 = df_sub["out_0"]
            out1 = df_sub["out_1"]

            for level in range(len(names)):
                br0 = df_sub["br_rating_{}_0".format(level)]
                br1 = df_sub["br_rating_{}_1".format(level)]
                br2 = df_sub["br_rating_{}_2".format(level)]

                br0 = br0 / (br0 + br1 + br2)
                br1 = br1 / (br0 + br1 + br2)
                br2 = br2 / (br0 + br1 + br2)
                normalized_br = br2 / (br0 + br1 + br2)

                nbr = normalized_br;
                df_sub["nbr_{}".format(level)] = nbr

                nout = df_sub["out_1"]

                # corr, pv = spearmanr(nbr, nout);
                corr, pv = pearsonr(nbr, nout);
                out_ps.append(pv)
                out_correls.append(corr)

            tot_out_correls.append(np.stack(out_correls))
            tot_out_ps.append(np.stack(out_ps))

    return tot_out_correls, tot_out_ps

seeds = [0, 1, 2, 3, 4]

all_attn_correls = []
all_attn_ps = []

all_out_correls = []
all_out_ps = []

var_tar = "recency"

plt_save = False

summed_accs = dict();
summed_spec = dict();

for model in models :
    summed_accs[model] = {}
    summed_spec[model] = {}

    summed_accs[model][0] = []
    summed_accs[model][1] = []
    summed_accs[model][2] = []
    summed_accs[model][3] = []
    summed_accs[model][4] = []

    summed_spec[model][0] = []
    summed_spec[model][1] = []
    summed_spec[model][2] = []
    summed_spec[model][3] = []
    summed_spec[model][4] = []

    for seed in seeds :

        sw_tot_out_cs = []
        sw_normed_out_cs = []

        for session_idx in range(5) :

            tot_out_correls, tot_out_ps = correl_func(var_tar=var_tar, model=model, session_idx=session_idx, seed=seed)
            tot_ans_correls, _ = correl_func(var_tar=var_tar, model="ans", seed=seed)

            all_out_correls.append(tot_out_correls)
            all_out_ps.append(tot_out_ps)

            tot_out_correls = np.stack(tot_out_correls)
            #tot_out_ps = np.stack(all_attn_ps).mean(0);

            tot_out_cs = np.stack(tot_out_correls);
            tot_ans_cs = np.stack(tot_ans_correls);


            normed_out_cs = (tot_out_cs - tot_out_cs.min(1,keepdims=True))/(tot_out_cs.max(1,keepdims=True) - tot_out_cs.min(1,keepdims=True))
            normed_ans_cs = (tot_ans_cs - tot_ans_cs.min(1,keepdims=True))/(tot_ans_cs.max(1,keepdims=True) - tot_ans_cs.min(1,keepdims=True))

            acc = tot_out_cs.diagonal().mean();
            rob = abs(normed_ans_cs - normed_out_cs).mean()

            summed_accs[model][session_idx].append(acc)
            summed_spec[model][session_idx].append(1-rob)

            tot_out_ps = np.stack(tot_out_ps);

            min_cs = np.min(tot_out_cs,axis=1,keepdims=True)
            max_cs = np.max(tot_out_cs,axis=1,keepdims=True)

            sw_tot_out_cs.append(tot_out_cs)
            sw_normed_out_cs.append(normed_out_cs)
            #tot_attn_cs = np.stack(tot_attn_correls);
            #tot_attn_ps = np.stack(tot_attn_ps);

            #min_cs = np.min(tot_attn_cs,axis=1,keepdims=True)
            #max_cs = np.max(tot_attn_cs,axis=1,keepdims=True)


            df_temp = pd.DataFrame();

            for m_idx, mod in enumerate(["Model{}".format(names[i]) for i in range(len(names))]) :
                df_temp[mod] = tot_out_cs[:,m_idx]
                #df_temp[mod] = normed_out_cs[:, m_idx]

            #df_temp.to_csv("./{}/{}_{}_normed_hm_수진전달.csv".format(var_tar, model, var_tar))

        if plt_save:
            fig, axes = plt.subplots(nrows=5, ncols=2)

            for session_idx in range(5) :

                im, cbar = heatmap(sw_tot_out_cs[session_idx], ["Behav{}".format(names[i]) for i in range(len(names))],
                        ["Model{}".format(names[i]) for i in range(len(names))], ax=axes[session_idx][0],
                        cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

                im2, cbar2 = heatmap(sw_normed_out_cs[session_idx], ["Behav{}".format(names[i]) for i in range(len(names))],
                        ["Model{}".format(names[i]) for i in range(len(names))], ax=axes[session_idx][1],
                        cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

                texts = annotate_heatmap(im, valfmt="{x:.3f}")
                texts2 = annotate_heatmap(im2, valfmt="{x:.3f}")


            fig.tight_layout();
            plt.show()

accs_means = dict()
specs_means = dict()
accs_stds = dict()
specs_stds = dict()

for model in models :
    accs_means[model] = []
    specs_means[model] = []
    accs_stds[model] = []
    specs_stds[model] = []

    for session_idx in range(5) :
        accs_means[model].append(np.mean(summed_accs[model][session_idx]))
        accs_stds[model].append(np.std(summed_accs[model][session_idx]))
        specs_means[model].append(np.mean(summed_spec[model][session_idx]))
        specs_stds[model].append(np.std(summed_spec[model][session_idx]))

#specs_means['LSTM_Attn2_0.0'][2]=0.890349673530045

accs_means_df = pd.DataFrame.from_dict(accs_means);
accs_stds_df = pd.DataFrame.from_dict(accs_stds)
spec_means_df = pd.DataFrame.from_dict(specs_means);
spec_stds_df = pd.DataFrame.from_dict(specs_stds);

accs_means_df.to_csv("Steps_Acc_Mean_{}.csv".format(var_tar))
accs_stds_df.to_csv("Steps_Acc_Std_{}.csv".format(var_tar))
spec_means_df.to_csv("Steps_Rob_Mean_{}.csv".format(var_tar))
spec_stds_df.to_csv("Steps_Rob_Std_{}.csv".format(var_tar))



