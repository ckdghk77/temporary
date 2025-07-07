import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr

import seaborn as sns
sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


fname = "data_model.csv"
sub_ids = ["sangwan"]

#fname = "data_0901_me.csv"
#sub_ids = ["37mm2uvd9he21lvlspolygpa", "6ks7q1bhv83hj7eqmobtcmxz", "9rsvd06013l607yf6gungpfb"]
df = pd.read_csv(fname)

for sub_id in sub_ids :
    df_sub = df.loc[df.Sub_id==sub_id];

    fig, axes = plt.subplots(nrows=2, ncols=2);

    br0 = df_sub["br_rating_0"]
    br1 = df_sub["br_rating_1"]
    br2 = df_sub["br_rating_2"]

    br0 = br0/(br0 + br1 + br2)
    br1 = br1/(br0 + br1 + br2)
    br2 = br2/(br0 + br1 + br2)
    #br2 = np.stack([np_softmax(dat)[-1] for dat in np.array([br0, br1, br2]).transpose()])
    normalized_br = br2/(br0 + br1 + br2)
    df_sub["normalized_br"] = normalized_br

    attn0 = df_sub["attn_0"]
    attn1 = df_sub["attn_1"]
    attn2 = df_sub["attn_2"]

    #attn2 = np.stack([np_softmax(dat)[-1] for dat in np.array([attn0, attn1, attn2]).transpose()])
    normalized_attn = attn2/(attn0 + attn1 + attn2)
    #normalized_attn = attn2

    df_sub["normalized_attn"] = normalized_attn

    out0 = df_sub["out_0"]
    out1 = df_sub["out_1"]
    out2 = df_sub["out_2"]

    for os_idx in range(2) :
        df_os = df_sub.loc[(df_sub.OS_idx==os_idx) & (df_sub.SR_idx > 7)]
        sns.scatterplot(x="SR_idx", y="normalized_br", color="C{}".format(os_idx+1), marker="*",s=100,
                        data=df_os, ax=axes[0,os_idx])
        sns.lineplot(x="SR_idx", y="normalized_br", color="C{}".format(os_idx+1), linestyle="--",
                        data=df_os, ax=axes[0,os_idx])
        sns.scatterplot(x="SR_idx", y="normalized_attn", color="C{}".format(os_idx+1),
                        data=df_os, ax=axes[0,os_idx])
        sns.lineplot(x="SR_idx", y="normalized_attn", color="C{}".format(os_idx+1), linestyle="-",
                        data=df_os, ax=axes[0,os_idx])

        nbr = df_os.normalized_br.values;
        nattn = df_os.normalized_attn.values;
        corr, pv = spearmanr(nbr, nattn);
        axes[0, os_idx].set_title("{:.3f}/{:.4f}".format(corr, pv))

    for os_idx in range(2):
        df_os = df_sub.loc[(df_sub.OS_idx==os_idx) & (df_sub.SR_idx > 7)];
        sns.scatterplot(x="SR_idx", y="normalized_br", color="C{}".format(os_idx + 1), marker="*", s=100,
                        data=df_os, ax=axes[1, os_idx])
        sns.lineplot(x="SR_idx", y="normalized_br", color="C{}".format(os_idx + 1), linestyle="--",
                     data=df_os, ax=axes[1, os_idx])
        sns.scatterplot(x="SR_idx", y="out_2", color="C{}".format(os_idx + 1),
                        data=df_os, ax=axes[1, os_idx])
        sns.lineplot(x="SR_idx", y="out_2", color="C{}".format(os_idx + 1), linestyle="-",
                     data=df_os, ax=axes[1, os_idx])

        nbr = df_os.normalized_br.values;
        nout = df_os.out_2.values;
        corr, pv = spearmanr(nbr, nout);
        axes[1, os_idx].set_title("{:.3f}/{:.4f}".format(corr, pv))

    fig.suptitle(sub_id)
    plt.show()

print("ss")




