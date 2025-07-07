import pandas as pd
import numpy as np
import cv2
import pickle, os
from scipy.io import savemat
from app.os_model.data_loader import json_to_nps
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
# sns.set(style="whitegrid")

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


sub_ids = ["sangwan"]
# names = ["(0.0)", "(0.18)", "(0.36)",
#         "(0.54)", "(0.72)"] # primacy
# names = ["(0.05)", "(0.1)", "(0.2)",
#         "(0.3)", "(0.4)"] # lr

#names = ["_RAND", "_PRIMACY(-1.0)", "_RECENCY(-1.0)",
#         "_LR(2.0)", "_OS"]  # deviates
names = ["_GROUP1", "_GROUP2"]  # deviates


test_type = "os_cols"

var_tar = "recency"


groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#groups = [0, 5, 6, 7, 12, 18, 19, 21, 22, 3, 4, 9, 11, 14, 23]

model_type = "LSTM_Attn2"
sbj_tar = "shuffle_backup_driving_LSTM_Attn2_Temp"
sbj_tar = os.path.join("../../../../", sbj_tar)

data_root = os.path.join(sbj_tar, "data")
meta_root = os.path.join(sbj_tar, "metadata")
model_root = os.path.join(sbj_tar, "incomplete")

mout_root = "./shuffle_backup_driving_LSTM_Attn2_Temp"

meta_dir = os.listdir(meta_root);

sub_ids = [];
m_ids = [];

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


W = 128;
H = 128;
seed=0
tot_os_vectors_so = []
tot_osi_values_so = []
tot_m_osi_values_so = []
tot_outputs = []

primacy_points = []
recency_points = []

for i in range(5) :
    for j in range(5) :
        for m in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5] :
            for n in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5] :
                primacy_points.append(np.array([i,0,j, m,n]))
                recency_points.append(np.array([i,4,j, m,n]))

idxes = np.arange(2500);
np.random.shuffle(idxes)
primacy_points = [primacy_points[_] for _ in idxes[:100]]
np.random.shuffle(idxes)
recency_points = [recency_points[_] for _ in idxes[:100]]

for sbj in groups :

    data_sbj = json_to_nps(os.path.join(model_root, sub_ids[sbj], sub_ids[sbj]+"_4.json"))
    attns_sbj = np.load(os.path.join(mout_root, "attention_{}.npy".format(sub_ids[sbj])))
    outs_sbj = np.load(os.path.join(mout_root, "out_{}.npy".format(sub_ids[sbj])))

    os_vectors_so = []
    osi_values_so = []
    m_osi_values_so = []

    for s_idx, session_id in enumerate(sorted(data_sbj.keys())):
        os_vectors = []
        osi_values = []
        m_osi_values = []

        exp_s = data_sbj[session_id];

        for r_idx in range(8) :
            onehot_stim = exp_s["R{}".format(r_idx)]["stim"].argmax(-1);

            attn_sbj = attns_sbj[s_idx * 8 + r_idx]
            out_sbj = outs_sbj[s_idx*8 + r_idx]

            attn_size = attn_sbj.shape[0];
            attn_stim = np.zeros_like(onehot_stim)
            for at in range(attn_size) :
                attn = attn_sbj[at].reshape(5,6);
                attn = (attn > 0.9);

                attn_stim = attn_stim | attn;

            onehot_stim *= attn_stim

            if test_type in ["os_rows", "os_cols"] :
                os_map = onehot_stim==2;
            elif test_type in ["inc2_rows", "inc2_cols"] :
                os_map = onehot_stim==1;

            row_oh_stim, col_oh_stim = np.where(onehot_stim==2);
            row_oh_rew, col_oh_rew = np.where(onehot_stim==4);
            row_inc_stim, col_inc_stim = np.where(onehot_stim == 1);
            #row_inc_stim, col_inc_stim = np.where(onehot_stim == 0);

            #if row_oh_stim != row_oh_rew :
            #    continue;

            br_values = exp_s["R{}".format(r_idx)]['br_rating']
            os_idx = br_values[2] - (br_values[0] + br_values[1])/2.0
            out_sbj = out_sbj[1]
            point_vector = np.stack([row_oh_stim[0] if len(row_oh_stim) else -1,
                                     col_oh_stim[0] if len(col_oh_stim) else -1,
                                     row_oh_rew[0] if len(row_oh_rew) else -1,
                                     np.mean(row_inc_stim) if len(row_inc_stim) else -1,
                                     np.mean(col_inc_stim) if len(col_inc_stim) else -1]);

            os_vectors.append(point_vector)
            osi_values.append(os_idx)
            m_osi_values.append(out_sbj)

        os_vectors = np.stack(os_vectors)
        os_vectors_so.append(os_vectors)
        osi_values_so.append(np.stack(osi_values))
        m_osi_values_so.append(np.stack(m_osi_values))

    tot_os_vectors_so.append(os_vectors_so)
    tot_osi_values_so.append(osi_values_so)
    tot_m_osi_values_so.append(m_osi_values_so)

tot_os_vectors_so = np.stack(tot_os_vectors_so)
tot_osi_values_so = np.stack(tot_osi_values_so)
tot_m_osi_values_so = np.stack(tot_m_osi_values_so)

primacy_points = np.stack(primacy_points)
recency_points = np.stack(recency_points)

primacy_points = primacy_points / tot_os_vectors_so.reshape(-1,len(point_vector)).std(0)
recency_points = recency_points / tot_os_vectors_so.reshape(-1,len(point_vector)).std(0)

tot_os_vectors_so = tot_os_vectors_so/ tot_os_vectors_so.reshape(-1,len(point_vector)).std(0)

pca = PCA(n_components=2)
pca.fit(tot_os_vectors_so.reshape(-1,len(point_vector)))

pca2 = PCA(n_components=1)
pca2.fit(tot_os_vectors_so.reshape(-1,len(point_vector)))


subj_df = pd.DataFrame()

for session in range(5) :
    session_os_vectors = tot_os_vectors_so[:,session]
    session_osi_values = tot_osi_values_so[:,session]
    session_m_osi_values = tot_m_osi_values_so[:,session]

    S, R, P = session_os_vectors.shape

    session_subjs = np.arange(S)[:,np.newaxis].repeat(R, axis=1)

    transformed_vec = pca.transform(session_os_vectors.reshape(-1,P))
    transformed_vec_mean = transformed_vec.reshape((S,R,2)).mean(1,keepdims=True)
    transformed_vec_mean = transformed_vec_mean.repeat(R,axis=1).reshape(-1,2)

    transformed_vec2 = pca2.transform(session_os_vectors.reshape(-1,P))
    transformed_vec_mean2 = transformed_vec2.reshape((S,R)).mean(1,keepdims=True)
    transformed_vec_mean2 = transformed_vec_mean2.repeat(R,axis=1).reshape(-1)


    kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(transformed_vec_mean)
    kmeans2 = KMeans(n_clusters=3, random_state=0, n_init=5).fit(transformed_vec_mean2.reshape(-1,1))

    session_df = pd.DataFrame.from_dict({"x" : transformed_vec[:,0], "y" : transformed_vec[:,1],
                                         "x2" : transformed_vec2[:,0],
                                         "x_mean": transformed_vec_mean[:,0],
                                         "y_mean": transformed_vec_mean[:,1],
                                         "x2_mean" : transformed_vec_mean2,
                                         "session" : session,
                                         "sbj_id" : session_subjs.reshape(-1),
                                         "os_idx" : session_osi_values.reshape(-1),
                                         "os_idx_mean" : session_osi_values.mean(-1,keepdims=True).
                                        repeat(R, axis=1).reshape(-1),
                                         "m_os_idx": session_m_osi_values.reshape(-1),
                                         "m_os_idx_mean": session_m_osi_values.mean(-1, keepdims=True).
                                        repeat(R, axis=1).reshape(-1),
                                         "cluster" : kmeans.labels_,
                                         "cluster2" : kmeans2.labels_})

    subj_df = pd.concat([subj_df, session_df])

primacy_transformed = pca.transform(primacy_points)
recency_transformed = pca.transform(recency_points)

prdf = pd.DataFrame.from_dict({"x" : np.concatenate([primacy_transformed[:,0],recency_transformed[:,0]], axis=0),
                        "y" : np.concatenate([primacy_transformed[:,1],recency_transformed[:,1]], axis=0),
                        "type":np.concatenate([np.zeros(shape=len(primacy_transformed)),
                                               np.ones(shape=len(recency_transformed))], axis=0)})
x_min = subj_df.x.min()
x_max = subj_df.x.max()
y_min = subj_df.y.min()
y_max = subj_df.y.max()


import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1,ncols=5);
plt.subplots_adjust(wspace=0.1)
import seaborn as sns

for session in range(5) :
    session_sbj_df = subj_df.loc[(subj_df.session == session)];

    session_xys = np.concatenate([np.expand_dims(session_sbj_df["x"].values,-1),
                    np.expand_dims(session_sbj_df["y"].values,-1)], axis=1)
    session_os_idx = session_sbj_df["os_idx"].values
    session_m_os_idx = session_sbj_df["m_os_idx"].values
    session_sbj_ids = session_sbj_df['sbj_id'].values
    session_cluster_ids = session_sbj_df['cluster'].values

    delta_xys_all = []
    delta_osis_all = []
    sbj_ids_all = []
    cluster_ids_all = []

    for xys_1, osis_1, sbj_id_1, cluster_id_1 in zip(session_xys, session_os_idx, session_sbj_ids, session_cluster_ids) :
        for xys_2, osis_2, sbj_id_2, cluster_id_2 in zip(session_xys, session_os_idx, session_sbj_ids, session_cluster_ids):
            if sbj_id_1 != sbj_id_2 :
                continue
            delta_xys = (abs(xys_1 - xys_2)).mean()
            delta_osis = abs(osis_1 - osis_2)

            delta_xys_all.append(delta_xys)
            delta_osis_all.append(delta_osis)
            sbj_ids_all.append(sbj_id_1)
            cluster_ids_all.append(cluster_id_1)

    df_delta = pd.DataFrame.from_dict({"delta_xys" : delta_xys_all, "delta_osis" : delta_osis_all,
                                       "sbj_id" : sbj_ids_all, "cluster_id": cluster_ids_all})

    sns.regplot(x="delta_xys", y="delta_osis", data=df_delta.loc[df_delta.cluster_id==0], ax= axes[session])
    sns.regplot(x="delta_xys", y="delta_osis", data=df_delta.loc[df_delta.cluster_id==1], ax= axes[session])
    sns.regplot(x="delta_xys", y="delta_osis", data=df_delta.loc[df_delta.cluster_id==2], ax= axes[session])

    delta_xys_all = np.stack(delta_xys_all);
    delta_osis_all = np.stack(delta_osis_all)
    cluster_ids_all = np.stack(cluster_ids_all)

    #sns.scatterplot(x="delta_xys", y="delta_osis", data=df_delta, hue='cluster_id', ax= axes[session])
    res0 = stats.pearsonr(delta_xys_all[np.where(cluster_ids_all==0)], delta_osis_all[np.where(cluster_ids_all==0)])
    res1 = stats.pearsonr(delta_xys_all[np.where(cluster_ids_all == 1)], delta_osis_all[np.where(cluster_ids_all == 1)])
    res2 = stats.pearsonr(delta_xys_all[np.where(cluster_ids_all == 2)], delta_osis_all[np.where(cluster_ids_all == 2)])

    res0 = [round(_, 3) for _ in res0];
    res1 = [round(_, 3) for _ in res1];
    res2 = [round(_, 3) for _ in res2];

    axes[session].legend([],[],frameon=False)
    axes[session].set_xticks([])
    axes[session].set_yticks([])
    axes[session].set_title("{}\n{}\n{}".format(res0, res1,res2))

    axes[session].get_yaxis().set_visible(False)
    axes[session].get_xaxis().set_visible(False)
plt.show()


#############################################################
#####  PCA 1 component
#############################################################
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1,ncols=5);
plt.subplots_adjust(wspace=0.1)
import seaborn as sns

for session in range(5) :
    session_sbj_df = subj_df.loc[(subj_df.session == session)];

    session_xys = session_sbj_df["x2"].values
    session_os_idx = session_sbj_df["os_idx"].values
    session_sbj_ids = session_sbj_df['sbj_id'].values
    session_cluster_ids = session_sbj_df['cluster2'].values


    sns.regplot(x="x2", y="os_idx", data=session_sbj_df.loc[session_sbj_df.cluster2==0], ax= axes[session])
    sns.regplot(x="x2", y="os_idx", data=session_sbj_df.loc[session_sbj_df.cluster2==1], ax= axes[session])
    sns.regplot(x="x2", y="os_idx", data=session_sbj_df.loc[session_sbj_df.cluster2==2], ax= axes[session])

    #sns.scatterplot(x="delta_xys", y="delta_osis", data=df_delta, hue='cluster_id', ax= axes[session])
    res0 = stats.pearsonr(session_xys[np.where(session_cluster_ids==0)], session_os_idx[np.where(session_cluster_ids==0)])
    res1 = stats.pearsonr(session_xys[np.where(session_cluster_ids==1)], session_os_idx[np.where(session_cluster_ids==1)])
    res2 = stats.pearsonr(session_xys[np.where(session_cluster_ids==2)], session_os_idx[np.where(session_cluster_ids==2)])

    res0 = [round(_, 3) for _ in res0];
    res1 = [round(_, 3) for _ in res1];
    res2 = [round(_, 3) for _ in res2];

    axes[session].legend([],[],frameon=False)
    axes[session].set_xticks([])
    axes[session].set_yticks([])
    axes[session].set_title("{}\n{}\n{}".format(res0, res1,res2))

    axes[session].get_yaxis().set_visible(False)
    axes[session].get_xaxis().set_visible(False)
plt.show()

import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=1,ncols=5);
plt.subplots_adjust(wspace=0.1)
import seaborn as sns

for session in range(5) :
    sns.scatterplot(data=subj_df.loc[(subj_df.session == session)], x="x", y="y", hue="cluster",
                    ax=axes[session])
    axes[session].legend([],[],frameon=False)
    axes[session].set_xlim([x_min, x_max])
    axes[session].set_ylim([y_min, y_max])
    axes[session].set_xticks([])
    axes[session].set_yticks([])

    axes[session].get_yaxis().set_visible(False)
    axes[session].get_xaxis().set_visible(False)
plt.show()


import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=10,ncols=5);
plt.subplots_adjust(wspace=0.1)
import seaborn as sns

x_min = subj_df.x.min()
x_max = subj_df.x.max()
y_min = subj_df.y.min()
y_max = subj_df.y.max()

for sbj_idx in range(10) :

    for session in range(5) :
        sns.scatterplot(data=prdf, x="x", y="y", ax=axes[sbj_idx][session], hue="type", palette="coolwarm", alpha=0.4)
        sns.scatterplot(data=subj_df.loc[(subj_df.session == session) & (subj_df.sbj_id == sbj_idx)], x="x", y="y",
                        palette="seagreen",ax=axes[sbj_idx][session])

        axes[sbj_idx][session].legend([],[],frameon=False)
        axes[sbj_idx][session].set_xlim([x_min, x_max])
        axes[sbj_idx][session].set_ylim([y_min, y_max])
        axes[sbj_idx][session].set_xticks([])
        axes[sbj_idx][session].set_yticks([])

        axes[sbj_idx][session].get_yaxis().set_visible(False)
        axes[sbj_idx][session].get_xaxis().set_visible(False)

plt.show()

