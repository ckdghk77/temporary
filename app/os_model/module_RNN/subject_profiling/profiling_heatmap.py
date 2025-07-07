import pandas as pd
import numpy as np
from app.os_model.visualizer import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import os
from scipy.stats import linregress
import statsmodels.api as sm
import seaborn as sns
#sns.set(style="whitegrid")
from scipy.io import savemat

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


#names = ["(0.0|0.0)", "(0.36|0.0)", "(0.72|0.0)",
#         "(0.0|0.36)", "(0.36|0.36)", "(0.72|0.36)",
#         "(0.0|0.72)", "(0.36|0.72)", "(0.72|0.72)"] # contrast

names = ["(0.0|0.36)", "(0.72|0.36)"] # contrast

#names = ["(0.05)", "(0.1)", "(0.2)",
#         "(0.3)", "(0.4)"] # lr

#names = ["_RAND", "_PRIMACY(-1.0)", "_RECENCY(-1.0)",
#         "_LR(2.0)", "_OS"] # deviates


def correl_func(subj_tar, var_tar, models, seed) :
    tot_out_correls = []
    tot_out_ps = []

    for model in models: # iterate through subjects
        fname = "{}/profiling_exp_{}_{}_{}.csv".format(subj_tar, var_tar, model, seed)

        df = pd.read_csv(fname)

        in_stims = []
        out_correls = []
        out_ps = []

        df_sub = df

        out0 = df_sub["out_0"]
        out1 = df_sub["out_1"]

        for level in range(len(names)):
            br0 = df_sub["br_rating_{}_0".format(level)]
            br1 = df_sub["br_rating_{}_1".format(level)]
            br2 = df_sub["br_rating_{}_2".format(level)]

            #br_std = np.concatenate([br0.values, br1.values, br2.values]).std();

            #br0 /= br_std
            #br1 /= br_std
            #br2 /= br_std

            br0 = br0 / (br0 + br1 + br2)
            br1 = br1 / (br0 + br1 + br2)
            br2 = br2 / (br0 + br1 + br2)
            normalized_br = br2 / (br0 + br1 + br2 + 1e-8)

            nbr = normalized_br;
            df_sub["nbr_{}".format(level)] = nbr

            nout = df_sub["out_1"]
            corr, pv = spearmanr(nbr, nout);
            #corr, pv = pearsonr(nbr, nout);

            out_ps.append(pv)
            if pv < 5e-2:
                out_correls.append(corr)
            else :
                out_correls.append(0.0)

        tot_out_correls.append(np.stack(out_correls))
        tot_out_ps.append(np.stack(out_ps))

    return tot_out_correls, tot_out_ps

seeds = [0]
sbj_idxes = np.arange(25)
all_attn_correls = []
all_attn_ps = []

all_out_correls = []
all_out_ps = []

sbj_tar = "shuffle_backup_driving_LSTM_Attn2_Temp"
var_tar = "lr"

plt_save = True

summed_accs = dict();
summed_spec = dict();

sbjs = os.listdir(sbj_tar)


for seed in seeds :

    tot_out_correls, tot_out_ps = correl_func(subj_tar=sbj_tar, var_tar=var_tar, models=sbj_idxes, seed=seed)

    all_out_correls.append(tot_out_correls)
    all_out_ps.append(tot_out_ps)

    tot_out_correls = np.stack(tot_out_correls)
    #tot_out_ps = np.stack(all_attn_ps).mean(0);

    tot_out_cs = np.stack(tot_out_correls);

    normed_out_cs = (tot_out_cs - tot_out_cs.min(1,keepdims=True))/(tot_out_cs.max(1,keepdims=True) - tot_out_cs.min(1,keepdims=True)+1e-6)

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

    df_temp.to_csv("./{}/{}_normed_hm_수진전달.csv".format(sbj_tar, var_tar))

    if plt_save:

        fig,axes = plt.subplots(nrows=1, ncols=2)

        normed_sbj_vals = []
        selected_sbj_idxes = []

        for sbj_idx, subj in enumerate(tot_out_cs) :
            if subj.sum() == 0 :
                continue
            normalized_subj = subj/subj.sum()

            normed_sbj_vals.append(normalized_subj)
            selected_sbj_idxes.append(sbj_idx)

            for v_idx, val in enumerate(normalized_subj) :
                axes[v_idx].scatter(0,val, c="C{}".format(sbj_idx))

        plt.show()

        mean_sbj_vals = np.stack(normed_sbj_vals).mean(0,keepdims=True)

        selected_idxes = np.argmax(np.stack(normed_sbj_vals) > mean_sbj_vals, -1);

        sbj_uids = []
        sbj_groups = []
        sbj_values = []

        for usi in np.unique(selected_idxes) :

            for _ in np.where(selected_idxes==usi)[0] :
                sbj_idx = selected_sbj_idxes[_];
                sbj_groups.append(usi)
                sbj_values.append(normed_sbj_vals[_][0])

            print("usi({}) : {}".format(usi,[selected_sbj_idxes[_]
                                             for _ in np.where(selected_idxes==usi)[0]]))

        pd_sbj_infos = pd.DataFrame().from_dict({
            "sbj_group" : sbj_groups, "sbj_value" : sbj_values
        });

        savemat(file_name="./{}.mat".format(var_tar), mdict={
            "sbj_group" : sbj_groups, "sbj_value" : sbj_values
        })
        fig, ax = plt.subplots(nrows=1, ncols=1)

        im, cbar = heatmap(tot_out_cs, ["Sbj{}".format(sbj_idxes[i]) for i in range(len(sbj_idxes))],
                ["Model{}".format(names[i]) for i in range(len(names))], ax=ax,
                cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

        #im2, cbar2 = heatmap(normed_out_cs, ["Sbj{}".format(sbj_idxes[i]) for i in range(len(sbj_idxes))],
        #        ["Model{}".format(names[i]) for i in range(len(names))], ax=ax[1],
        #        cmap="YlGn", cbarlabel="Correlation of p-value < 0.05")

        texts = annotate_heatmap(im, valfmt="{x:.3f}")
        #texts2 = annotate_heatmap(im2, valfmt="{x:.3f}")


        fig.tight_layout();
        plt.show()





