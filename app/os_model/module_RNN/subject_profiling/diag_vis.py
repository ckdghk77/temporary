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

fname = "data_0901_me.csv"
sub_ids = ["37mm2uvd9he21lvlspolygpa", "6ks7q1bhv83hj7eqmobtcmxz", "9rsvd06013l607yf6gungpfb"]

df = pd.read_csv(fname)

for sub_id in sub_ids :
    df_sub = df.loc[(df.Sub_id==sub_id) & (df.OS_idx==1)];
    #df_sub = df.loc[(df.Sub_id==sub_id)];

    fig, axes = plt.subplots(nrows=2, ncols=1);

    attn0 = df_sub["attn_0"]
    attn1 = df_sub["attn_1"]
    attn2 = df_sub["attn_2"]

    normalized_attn = attn2/(attn0 + attn1 + attn2)
    df_sub["normalized_attn"] = normalized_attn

    out0 = df_sub["out_0"]
    out1 = df_sub["out_1"]

    br0 = df_sub["br_rating_0"]
    br1 = df_sub["br_rating_1"]
    br2 = df_sub["br_rating_2"]

    br0 = br0/(br0 + br1 + br2)
    br1 = br1/(br0 + br1 + br2)
    br2 = br2/(br0 + br1 + br2)
    #br2 = np.stack([np_softmax(dat)[-1] for dat in np.array([br0, br1, br2]).transpose()])
    normalized_br = br2/(br0 + br1 + br2)
    df_sub["normalized_br"] = normalized_br

    sns.scatterplot(x="SR_idx", y="normalized_br", color="C0", marker="*",s=100,
                    data=df_sub, ax=axes[0])
    sns.lineplot(x="SR_idx", y="normalized_br", color="C0", linestyle="--",
                    data=df_sub, ax=axes[0])
    sns.scatterplot(x="SR_idx", y="normalized_attn", color="C0",
                    data=df_sub, ax=axes[0])
    sns.lineplot(x="SR_idx", y="normalized_attn", color="C0", linestyle="-",
                    data=df_sub, ax=axes[0])
    #axes[0,l_idx].set_ylim([0.1,0.5])
    nbr = df_sub.normalized_br.values;
    nattn = df_sub.normalized_attn.values;
    #corr, pv = pearsonr(nbr, nattn);
    _,_, corr, pv, _ = linregress(nbr, nattn);

    axes[0].set_title("{:.3f}/{:.4f}".format(corr, pv))

    sns.scatterplot(x="SR_idx", y="normalized_br", color="C0", marker="*", s=100,
                    data=df_sub, ax=axes[1])
    sns.lineplot(x="SR_idx", y="normalized_br", color="C0", linestyle="--",
                 data=df_sub, ax=axes[1])
    sns.scatterplot(x="SR_idx", y="out_1", color="C0",
                    data=df_sub, ax=axes[1])
    sns.lineplot(x="SR_idx", y="out_1", color="C0", linestyle="-",
                 data=df_sub, ax=axes[1])

    nbr = df_sub.normalized_br.values;
    nout = df_sub.out_1.values;
    corr, pv = pearsonr(nbr, nout);
    #_,_, corr, pv, _ = linregress(nbr, nout);
    axes[1].set_title("{:.3f}/{:.4f}".format(corr, pv))

    fig.suptitle(sub_id)
    plt.show()

print("ss")




