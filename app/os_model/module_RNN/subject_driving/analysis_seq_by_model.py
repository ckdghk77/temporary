
import pickle, os
import numpy as np
import pandas as pd
from scipy.io import savemat
from app.os_model.data_loader import json_to_nps
from app.os_model.exp.Oneshot_sangwan import os_sangwan
from app.os_model.module_RNN.subject_driving.gen_alg import mlrose_OS
from tqdm import tqdm

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


def correl_fit(sub_ids, m_ids, model_root, fitting_type) :

    tot_params = []
    tot_fitness = []


    data_dict = {"sbj_id" :[],
                 "primacy" : [],
                 "recency" : [],
                 "learning rate": [],
                 "session": [],
                 "fitness": [],
                 "base_fitness": [],
                 }

    for sbj, m_id in tqdm(zip(sub_ids, m_ids)):
        try :
            data_sbj = json_to_nps(os.path.join(model_root, sbj, sbj + "_4.json"))
        except :
            data_sbj = json_to_nps(os.path.join(model_root, sbj + ".json"))

        sbj_params = []
        sbj_fitness = []
        tot_h_outs = []
        tot_h_stims = []

        for session in range(5):
            d_s = data_sbj["S{}".format(session)]

            h_outs = []
            h_stims = []
            for round in range(8):
                d_s_r = d_s["R{}".format(round)]
                stim_idx = d_s_r['stim'].argmax(-1);
                #if np.where(stim_idx==2)[0][0] !=  np.where(stim_idx==4)[0][0] :
                #    continue;
                h_outs.append(d_s_r['br_rating']/(sum(d_s_r['br_rating'])+1e-6))
                h_stims.append(d_s_r['stim'])

            tot_h_outs.extend(h_outs)
            tot_h_stims.extend(h_stims)

            optOS = mlrose_OS(fitting_type, init_state=[1.0, 1.0, 1.0],
                              max_attempts=100,
                              max_iters=1000, random_state=10)

            best_state, best_fitness, base_fitness = optOS.fit(h_stims, h_outs)
            data_dict['sbj_id'].append(sbj)
            data_dict['primacy'].append(best_state[0])
            data_dict['recency'].append(best_state[1])
            data_dict['learning rate'].append(best_state[2])
            data_dict['session'].append(session)
            data_dict['fitness'].append(best_fitness)
            data_dict['base_fitness'].append(base_fitness)

            sbj_params.append(best_state)
            sbj_fitness.append(best_fitness)

        tot_params.append(sbj_params)
        tot_fitness.append(sbj_fitness)
    return tot_params, tot_fitness, pd.DataFrame.from_dict(data_dict)


fitting_type = "rhill_climb" # genetic  //  annealing // rhill_climb

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

params_drs, fit_drs, drs_df = correl_fit(sub_ids_dr, m_ids_dr, model_root_driving, fitting_type)
params_bs, fit_bs, bs_df = correl_fit(sub_ids_bs, m_ids_bs, model_root_base, fitting_type)

drs_df["type"] = "DNE"
bs_df["type"] = "BASE"

tot_df = pd.concat([drs_df,bs_df])

tot_df.to_csv("./param (session indep)_{}.csv".format(fitting_type))
#import seaborn as sns
#import matplotlib.pyplot as plt

#fig, axes = plt.subplots(nrows=3, ncols=1)

#sns.catplot(data=tot_df, x="session", y="primacy", hue="type", kind='box', ax=axes[0])
#sns.catplot(data=tot_df, x="session", y="recency", hue="type", kind='box', ax=axes[1])
#sns.catplot(data=tot_df, x="session", y="learning rate", hue="type", kind='box', ax=axes[2])
#plt.show()
print('ss')