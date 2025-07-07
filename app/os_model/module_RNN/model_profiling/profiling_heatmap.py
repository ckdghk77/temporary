import pandas as pd
import numpy as np
from app.os_model.visualizer import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
#from scipy.stats import linregress
#import statsmodels.api as sm
#import seaborn as sns
#sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x



models = ["SLP_0.1", "MLP_0.1", "CNN_0.1","Transformer_0.1", "LSTM_0.1", "BiLSTM_0.1", "BiLSTM_Attn_0.1", "LSTM_Attn_0.1", "BiLSTM_Attn2_0.1", "LSTM_Attn2_0.1",
          "SLP_0.0", "MLP_0.0", "CNN_0.0","Transformer_0.0", "LSTM_0.0", "BiLSTM_0.0", "BiLSTM_Attn_0.0", "LSTM_Attn_0.0", "BiLSTM_Attn2_0.0", "LSTM_Attn2_0.0"]
#models = ["LSTM_Attn2_0.0"]
#models = ["LSTM_0.0"]

sub_ids = ["sangwan"]
names = ["(0.0)", "(0.18)", "(0.36)",
         "(0.54)", "(0.72)"] # primacy

#names = ["(0.05)", "(0.1)", "(0.2)",
#         "(0.3)", "(0.4)"] # lr

#names = ["_RAND", "_PRIMACY(-1.0)", "_RECENCY(-1.0)",
#         "_LR(2.0)", "_OS"] # deviates


def correl_func(var_tar, model, seed) :
    tot_out_correls = []
    tot_out_ps = []

    for outer_level, name in enumerate(names):
        fname = "{}/profiling_exp_{}_{}_{}_{}.csv".format(var_tar, model, var_tar, outer_level, seed)

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

var_tar = "ablation"

plt_save = True

summed_accs = dict();
summed_spec = dict();

for model in models :
    summed_accs[model] = []
    summed_spec[model] = []

    for seed in seeds :

        tot_out_correls, tot_out_ps = correl_func(var_tar=var_tar, model=model, seed=seed)
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

        summed_accs[model].append(acc)
        summed_spec[model].append(1-rob)

        tot_out_ps = np.stack(tot_out_ps);

        min_cs = np.min(tot_out_cs,axis=1,keepdims=True)
        max_cs = np.max(tot_out_cs,axis=1,keepdims=True)

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
            fig, ax = plt.subplots(nrows=1, ncols=2)

            im, cbar = heatmap(tot_out_cs, ["Behav{}".format(names[i]) for i in range(len(names))],
                    ["Model{}".format(names[i]) for i in range(len(names))], ax=ax[0],
                    cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

            im2, cbar2 = heatmap(normed_out_cs, ["Behav{}".format(names[i]) for i in range(len(names))],
                    ["Model{}".format(names[i]) for i in range(len(names))], ax=ax[1],
                    cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

            texts = annotate_heatmap(im, valfmt="{x:.3f}")
            texts2 = annotate_heatmap(im2, valfmt="{x:.3f}")


            fig.tight_layout();
            plt.show()


acc_df = pd.DataFrame.from_dict(summed_accs);
spec_df = pd.DataFrame.from_dict(summed_spec);

acc_df.to_csv("Acc_{}.csv".format(var_tar))
spec_df.to_csv("Rob_{}.csv".format(var_tar))



