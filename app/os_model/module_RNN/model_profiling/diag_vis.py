import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from scipy.stats import linregress

import seaborn as sns
sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

fname = "data_model_profiling_1_0.csv"
sub_ids = ["sangwan"]
levels = [0,1,2,3,4]

df = pd.read_csv(fname)

for sub_id in sub_ids :
    #df_sub = df.loc[(df.Sub_id==sub_id) & (df.OS_idx==1)];
    df_sub = df.loc[(df.Sub_id==sub_id)];

    fig, axes = plt.subplots(nrows=2, ncols=len(levels));

    attn0 = df_sub["attn_0"]
    attn1 = df_sub["attn_1"]
    attn2 = df_sub["attn_2"]

    normalized_attn = attn2/(attn0 + attn1 + attn2)
    df_sub["normalized_attn"] = normalized_attn

    out0 = df_sub["out_0"]
    out1 = df_sub["out_1"]

    for l_idx, level in enumerate(levels) :
        br0 = df_sub["br_rating_{}_0".format(level)]
        br1 = df_sub["br_rating_{}_1".format(level)]
        br2 = df_sub["br_rating_{}_2".format(level)]

        br0 = br0/(br0 + br1 + br2)
        br1 = br1/(br0 + br1 + br2)
        br2 = br2/(br0 + br1 + br2)
        #br2 = np.stack([np_softmax(dat)[-1] for dat in np.array([br0, br1, br2]).transpose()])
        normalized_br = br2/(br0 + br1 + br2)
        df_sub["normalized_br"] = normalized_br

        sns.scatterplot(x="SR_idx", y="normalized_br", color="C0", marker="*",s=100,
                        data=df_sub, ax=axes[0,l_idx])
        sns.lineplot(x="SR_idx", y="normalized_br", color="C0", linestyle="--",
                        data=df_sub, ax=axes[0,l_idx])
        sns.scatterplot(x="SR_idx", y="normalized_attn", color="C0",
                        data=df_sub, ax=axes[0,l_idx])
        sns.lineplot(x="SR_idx", y="normalized_attn", color="C0", linestyle="-",
                        data=df_sub, ax=axes[0,l_idx])
        #axes[0,l_idx].set_ylim([0.1,0.5])
        nbr = df_sub.normalized_br.values;
        nattn = df_sub.normalized_attn.values;
        corr, pv = pearsonr(nbr, nattn);
        #corr, pv = spearmanr(nbr, nattn);
        #_,_, corr, pv, _ = linregress(nbr, nattn);

        axes[0, l_idx].set_title("{:.3f}/{:.4f}".format(corr, pv))

        '''
        sns.scatterplot(x="SR_idx", y="normalized_br", color="C0", marker="*", s=100,
                        data=df_sub, ax=axes[1, l_idx])
        sns.lineplot(x="SR_idx", y="normalized_br", color="C0", linestyle="--",
                     data=df_sub, ax=axes[1, l_idx])
        sns.scatterplot(x="SR_idx", y="out_1", color="C0",
                        data=df_sub, ax=axes[1, l_idx])
        sns.lineplot(x="SR_idx", y="out_1", color="C0", linestyle="-",
                     data=df_sub, ax=axes[1, l_idx])
        '''

        sns.scatterplot(x="normalized_br", y="out_1", color="C0", linestyle="-",
                     data=df_sub, ax=axes[1, l_idx])

        nbr = df_sub.normalized_br.values;
        nout = df_sub.out_1.values;
        corr, pv = pearsonr(nbr, nout);
        #corr, pv = spearmanr(nbr, nout);
        ##_,_, corr, pv, _ = linregress(nbr, nout);
        axes[1, l_idx].set_title("{:.3f}/{:.4f}".format(corr, pv))

    fig.suptitle(sub_id)
    plt.show()

print("ss")




