import pandas as pd
import numpy as np
import cv2
import pickle, os
import numpy as np
from scipy.io import savemat
from app.os_model.data_loader import json_to_nps
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from scipy import stats
from app.os_model.exp.Oneshot_sangwan import os_sangwan
# sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def sbj_meta_extract(data_root, meta_root, meta_dir) :
    sub_ids = []
    m_ids = []

    for sub_files in os.listdir(data_root):

        if sub_files.endswith(".json"):
            sub_id = sub_files[:-5]

            sub_ids.append(sub_id)
            for m in meta_dir:

                with open(os.path.join(meta_root, m), "r") as f:

                    lines = f.readlines();
                    if lines:
                        try:
                            sub_id_meta = lines[3].split("subId\t")[1];
                        except:
                            continue
                        if sub_id == sub_id_meta[:-1]:
                            m_name = m
                            m_ids.append(sub_id_meta[:-1])
                            break;

    return sub_ids, m_ids

def correl_anal(sub_ids, m_ids, model_root) :

    tot_correls = []
    tot_pvals = []

    for sbj, m_id in zip(sub_ids, m_ids):
        try :
            data_sbj = json_to_nps(os.path.join(model_root, sbj, sbj + "_4.json"))
        except :
            data_sbj = json_to_nps(os.path.join(model_root, sbj + ".json"))

        correl_vals = [[] for _ in range(len(os_models))]
        p_vals = [[] for _ in range(len(os_models))]
        for session in range(5):
            d_s = data_sbj["S{}".format(session)]

            h_outs = []
            m_outs = [[] for _ in range(len(os_models))]
            for round in range(8):
                d_s_r = d_s["R{}".format(round)]
                stim_idx = d_s_r['stim'].argmax(-1);
                if np.where(stim_idx==2)[0][0] !=  np.where(stim_idx==4)[0][0] :
                    continue;
                h_outs.append(d_s_r['br_rating'][-1]/(sum(d_s_r['br_rating'])+1e-6))
                #h_outs.append((d_s_r['br_rating'][0] + d_s_r['br_rating'][1])/(sum(d_s_r['br_rating'])+1e-6))
                #h_outs.append(d_s_r['br_rating'][-1] - (d_s_r['br_rating'][0]+d_s_r['br_rating'][1])*0.5)
                for omi, os_model in enumerate(os_models) :
                    m_out = os_model.perform(d_s_r["stim"])
                    m_outs[omi].append(m_out[-1,-1])
                    #m_outs[omi].append(sum(m_out[-1][:2]))
                    #m_outs[omi].append(m_out[-1,-1]-sum(m_out[-1][:2])*0.5)

            for omi, os_model in enumerate(os_models):
                try :
                    corr_val, p_val = stats.pearsonr(h_outs, m_outs[omi])

                    if np.isnan(corr_val) :
                        corr_val = -2.0
                        p_val = 0.0
                    #corr_val = -abs(np.stack(h_outs) - np.stack(m_outs[omi])).mean()
                    if p_val > 1.0 :
                        correl_vals[omi].append(-100.0)
                        p_vals[omi].append(0.0)
                        continue;
                except :
                    print("ss")
                correl_vals[omi].append(corr_val)
                p_vals[omi].append(p_val)

        tot_correls.append(correl_vals)
        tot_pvals.append(p_vals)

    return tot_correls, tot_pvals


var_tar = "recency"

if var_tar == "primacy" :
    os_models = [
        os_sangwan(primacy=0.0),
        os_sangwan(primacy=0.36),
        os_sangwan(primacy=1.0)
    ]
elif var_tar == "recency" :
    os_models = [
        os_sangwan(recency=0.0),
        os_sangwan(recency=0.18),
        os_sangwan(recency=0.36),
        os_sangwan(recency=0.54),
        os_sangwan(recency=0.72)
    ]
elif var_tar == "opt" :
    os_models = [
        os_sangwan(primacy=0.72, recency=0.0),
        os_sangwan(primacy=0.0, recency=0.72),
        os_sangwan(primacy=0.36, recency=0.36)
    ]

model_type = "LSTM_Attn2"
sbj_driving = "shuffle_backup_driving_LSTM_Attn2_Temp"
sbj_driving = os.path.join("../../../../", sbj_driving)

sbj_base = "shuffle_backup_no_driving"
sbj_base = os.path.join("../../../../", sbj_base)

data_root_driving = os.path.join(sbj_driving, "data")
meta_root_driving = os.path.join(sbj_driving, "metadata")
model_root_driving = os.path.join(sbj_driving, "incomplete")
meta_dir_driving = os.listdir(meta_root_driving);

data_root_base = os.path.join(sbj_base, "data")
meta_root_base = os.path.join(sbj_base, "metadata")
model_root_base = os.path.join(sbj_base, "incomplete")
meta_dir_base = os.listdir(meta_root_base);

sub_ids_dr, m_ids_dr = sbj_meta_extract(data_root_driving, meta_root_driving, meta_dir_driving)
sub_ids_bs, m_ids_bs = sbj_meta_extract(data_root_base, meta_root_base, meta_dir_base)

correl_drs, p_drs = correl_anal(sub_ids_dr, m_ids_dr, model_root_driving)
correl_bs, p_bs = correl_anal(sub_ids_bs, m_ids_bs, model_root_base)

dr_dict = {"CRMAX" :[], "CRCOUNT" :[], "Session":[], "Type":[]}
for s_idx in range(5) :
    for om_idx in range(len(os_models)) :
        cdrm = [cdr[om_idx][s_idx] for cdr in correl_drs]

        #dr_dict.update({"CR_{}".format(om_idx) : cdrm})
    cdrm_max = []
    for cdr in correl_drs :
        np_cdr = np.stack(cdr);
        if np.max(np_cdr[:, s_idx]) == -100.0 :
            continue;
        else :
            cdrm_max.append(np.argmax(np_cdr[:,s_idx]))

    cdrm_count = []
    for cdmax in cdrm_max :
        cdrm_count.append(len(np.where(cdrm_max == cdmax)[0]))
    dr_dict["CRCOUNT"].extend(cdrm_count)
    dr_dict["CRMAX"].extend(cdrm_max)
    dr_dict["Session"].extend([s_idx]*len(cdrm_max))
    dr_dict["Type"].extend([1]*len(cdrm_max))

bs_dict = {"CRMAX" :[], "CRCOUNT" :[], "Session":[], "Type":[]}
for s_idx in range(5) :
    for om_idx in range(len(os_models)) :
        cdrm = [cdr[om_idx][s_idx] for cdr in correl_bs]
        #bs_dict.update({"CR_{}".format(om_idx) : cdrm})
    cdrm_max = []
    for cdr in correl_bs:
        np_cdr = np.stack(cdr);
        if np.max(np_cdr[:, s_idx]) == -100.0:
            continue;
        else:
            cdrm_max.append(np.argmax(np_cdr[:,s_idx]))
    cdrm_count = []
    for cdmax in cdrm_max:
        cdrm_count.append(len(np.where(cdrm_max == cdmax)[0]))
    bs_dict["CRCOUNT"].extend(cdrm_count)
    bs_dict["CRMAX"].extend(cdrm_max)
    bs_dict["Session"].extend([s_idx] * len(cdrm_max))
    bs_dict["Type"].extend([0] * len(cdrm_max))

df_dr = pd.DataFrame.from_dict(dr_dict);
df_bs = pd.DataFrame.from_dict(bs_dict);
df_all = pd.concat([df_dr,df_bs])

import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2)

sns.barplot(df_bs, x="Session", y="CRCOUNT", hue="CRMAX", ax=axes[0])
sns.barplot(df_dr, x="Session", y="CRCOUNT", hue="CRMAX", ax=axes[1])

axes[0].set_title("base")
axes[1].set_title("dne")

fig.set_figheight(6)
fig.set_figwidth(12)
plt.suptitle(var_tar)
fig.tight_layout()
plt.show()
