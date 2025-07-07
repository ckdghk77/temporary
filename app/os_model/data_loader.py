import json
import numpy as np
from parse import parse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
sns.set(style="whitegrid")
import os

import torch, random
import torch.utils.data as data
from torch.utils.data import DataLoader


class CustomDataSet(data.Dataset) :

    def __init__(self, data, seed=None):
        self.data = data[0];
        self.data_oh = data[1];
        self.targets = data[2];
        self.vecs = data[3];

        if seed is not None :
            self.seed = seed;

            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(seed)

    def __getitem__(self, index):

        img, img_oh, target, vec = self.data[index], self.data_oh[index], self.targets[index], self.vecs[index]
        img_os = torch.where(img == 4)[0];

        return img, img_oh, img_os, target, vec

    def __len__(self):
        return len(self.data)

def triangle_vectorize(ratings, infos) :
    vector = np.zeros(shape=(3), dtype=np.float32);

    for rating in ratings["rating"] :

        img_idx = rating[0];
        img_idx = parse("{}.png", img_idx);
        img_idx = int(img_idx[0])
        rating = rating[1];

    if img_idx == int(infos["img_1"]):
        vector[2] = rating
    elif img_idx == int(infos["img_8"]):
        vector[1] = rating
    elif img_idx == int(infos["img_16"]):
        vector[0] = rating

    return vector


def scale_vectorize(ratings, infos) :

    vector = np.zeros(shape=(3), dtype=np.float32);

    for rating in ratings:

        img_idx = rating["question"][0];
        img_idx = parse("{}.png", img_idx);
        img_idx = int(img_idx[0])
        rating = rating["question"][1];


        if img_idx == int(infos["img_1"]) :
            vector[2] = rating
        elif img_idx == int(infos["img_8"]):
            vector[1] = rating
        elif img_idx == int(infos["img_16"]):
            vector[0] = rating

    return vector


def json_to_nps(path) :

    if type(path) == str :
        with open(path,"r") as f :
            info_js = json.load(f);

    in_nps = {}

    for idx, info in enumerate(info_js) :

        if "session" in info and "round" in info and "stim_and_reward" in info :

            session_str = "S{}".format(info["session"])
            round_str = "R{}".format(info["round"])

            if session_str not in in_nps :
                in_nps[session_str] = {}

            if round_str not in in_nps[session_str] :
                in_nps[session_str][round_str] = {}

            stim_tot = np.zeros(shape=(5,6,5),
                       dtype=np.float32)

            stim_and_rew = np.stack(info["stim_and_reward"]);

            stims = np.unique(stim_and_rew[:,:5]);

            for stim in stims :
                r,c = np.where(stim_and_rew[:,:5]==stim);
                onehot_v = np.zeros(shape=(5,), dtype=np.float32)

                if len(r) == 1 :
                    onehot_v[2] = 1.0
                    in_nps[session_str][round_str]["img_1"] = stim
                elif len(r) == 8 :
                    onehot_v[1] = 1.0
                    in_nps[session_str][round_str]["img_8"] = stim
                elif len(r) == 16 :
                    onehot_v[0] = 1.0
                    in_nps[session_str][round_str]["img_16"] = stim

                stim_tot[:,:5][r,c] = onehot_v

            rews = np.unique(stim_and_rew[:,5:]);

            for rew in rews:
                r,c = np.where(stim_and_rew[:,5:]==rew);
                onehot_v = np.zeros(shape=(5,), dtype=np.float32)

                if len(r) == 1:
                    onehot_v[4] = 1.0
                elif len(r) == 4:
                    onehot_v[3] = 1.0

                stim_tot[:,5:][r,c] = onehot_v

            in_nps[session_str][round_str]["stim"] = stim_tot

            gr_ratings = info_js[idx+1:idx+4];
            br_ratings = info_js[idx+4:idx+7]
            tr_ratings = info_js[idx+7]

            gr_vector = scale_vectorize(gr_ratings,
                                        in_nps[session_str][round_str]);
            br_vector = scale_vectorize(br_ratings,
                                           in_nps[session_str][round_str]);
            tr_vector = triangle_vectorize(tr_ratings,
                               in_nps[session_str][round_str]);

            in_nps[session_str][round_str]["gr_rating"] = gr_vector
            in_nps[session_str][round_str]["br_rating"] = br_vector
            in_nps[session_str][round_str]["tr_rating"] = tr_vector

            in_nps[session_str][round_str]["is_Pos"] = "50" in rews

    return in_nps


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def load_os_data(data_dicts, total_session=[0,1,2,3,4], train_session=[1,2,3,4],
                 val_session=[4], seed=0, batch_size=8, candidates=100):

    total_xs = []
    total_ys = []
    total_vecs = []

    train_xs = []
    train_ys = []
    train_vecs = []

    val_xs = []
    val_ys = []
    val_vecs = []

    task_vector = np.zeros(shape=(5,6,2), dtype=np.float32)
    task_vector[:,:5,0] = 1.0
    task_vector[:,5:,1] = 1.0

    for data_dict in data_dicts :

        for s in total_session :
            s_data = data_dict["S{}".format(s)]
            for r in s_data :
                dat = s_data[r]["stim"];
                lab = s_data[r]["br_rating"];

                #lab = np_softmax(lab);
                #lab = lab**2
                lab /= (lab.sum()+1e-8);
                lab = np.stack([sum(lab[:2]), lab[2]]);

                #print(lab[1])
                total_xs.append(dat)
                total_ys.append(lab)
                total_vecs.append(task_vector)

        for s in train_session :
            s_data = data_dict["S{}".format(s)]

            for r in s_data :
                dat = s_data[r]["stim"];
                lab = s_data[r]["br_rating"];

                #lab = np_softmax(lab);
                #lab = lab ** 2
                lab /= (lab.sum()+1e-8);
                lab = np.stack([sum(lab[:2]), lab[2]]);

                train_xs.append(dat)
                train_ys.append(lab)
                train_vecs.append(task_vector)

        for s in val_session:
            s_data = data_dict["S{}".format(s)]

            for r in s_data:
                dat = s_data[r]["stim"];
                lab = s_data[r]["br_rating"];

                #lab = np_softmax(lab);
                #lab = lab ** 2
                lab /= (lab.sum()+1e-8);
                lab = np.stack([sum(lab[:2]), lab[2]]);

                val_xs.append(dat)
                val_ys.append(lab)
                val_vecs.append(task_vector)

    train_oh_xs = [np.reshape(xs, (30, 5)) for xs in train_xs];
    train_long_xs = [np.argmax(xs, -1).reshape(30) for xs in train_xs];

    val_oh_xs = [np.reshape(xs, (30, 5)) for xs in val_xs];
    val_long_xs = [np.argmax(xs, -1).reshape(30) for xs in val_xs];

    total_oh_xs = [np.reshape(xs, (30, 5)) for xs in total_xs];
    total_long_xs = [np.argmax(xs, -1).reshape(30) for xs in total_xs];

    train_vecs = [vec.reshape(30,-1) for vec in train_vecs]
    val_vecs = [vec.reshape(30,-1) for vec in val_vecs]
    total_vecs = [vec.reshape(30, -1) for vec in total_vecs]

    total_oh_xs = np.stack(total_oh_xs);
    total_long_xs = np.stack(total_long_xs);
    total_ys = np.stack(total_ys);
    total_vecs = np.stack(total_vecs)

    train_oh_xs = np.stack(train_oh_xs);
    train_long_xs = np.stack(train_long_xs);
    train_ys = np.stack(train_ys);
    train_vecs = np.stack(train_vecs)

    val_oh_xs = np.stack(val_oh_xs);
    val_long_xs = np.stack(val_long_xs);
    val_ys = np.stack(val_ys);
    val_vecs = np.stack(val_vecs)

    total_long_xs = torch.LongTensor(total_long_xs)
    total_oh_xs = torch.FloatTensor(total_oh_xs)
    total_ys = torch.FloatTensor(total_ys)
    total_vecs = torch.LongTensor(total_vecs)

    train_long_xs = torch.LongTensor(train_long_xs)
    train_oh_xs = torch.FloatTensor(train_oh_xs)
    train_ys = torch.FloatTensor(train_ys);
    train_vecs = torch.LongTensor(train_vecs)

    val_long_xs = torch.LongTensor(val_long_xs)
    val_oh_xs = torch.FloatTensor(val_oh_xs)
    val_ys = torch.FloatTensor(val_ys);
    val_vecs = torch.LongTensor(val_vecs)

    #xs_all = torch.cat([train_xs, val_xs], dim=0);
    #ys_all = torch.cat([train_ys, val_ys], dim=0);
    #xs_all = total_xs
    #ys_all = total_ys
    #vecs_all = total_vecs

    incre_long_xs = val_long_xs[torch.where(val_ys.argmax(-1) == 0)];
    incre_oh_xs = val_oh_xs[torch.where(val_ys.argmax(-1) == 0)];
    incre_ys = val_ys[torch.where(val_ys.argmax(-1) == 0)];
    incre_vecs = val_vecs[torch.where(val_ys.argmax(-1) == 0)];

    os_long_xs = val_long_xs[torch.where(val_ys.argmax(-1) == 1)];
    os_oh_xs = val_oh_xs[torch.where(val_ys.argmax(-1) == 0)];
    os_ys = val_ys[torch.where(val_ys.argmax(-1) == 1)];
    os_vecs = val_vecs[torch.where(val_ys.argmax(-1) == 1)];

    train_dataset = CustomDataSet(data=(train_long_xs, train_oh_xs, train_ys, train_vecs), seed=seed);
    val_dataset = CustomDataSet(data=(val_long_xs, val_oh_xs, val_ys, val_vecs), seed=seed);

    incre_dataset = CustomDataSet(data=(incre_long_xs, incre_oh_xs, incre_ys, incre_vecs), seed=seed);
    os_dataset = CustomDataSet(data=(os_long_xs, os_oh_xs, os_ys, os_vecs), seed=seed);

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    incre_loader = DataLoader(incre_dataset, batch_size=batch_size)
    os_loader = DataLoader(os_dataset, batch_size=batch_size)


    return train_loader, val_loader, incre_loader, os_loader, (total_long_xs, total_oh_xs, total_ys, total_vecs)


def append_df(df_total, info_dict, sub_id, m_name) :


    S_range = range(5);
    R_range= range(8);

    rows = [];

    for S in S_range :
        for R in R_range :
            session_str = "S{}".format(S)
            round_str = "R{}".format(R)

            stim = np.argmax(info_dict[session_str][round_str]["stim"],-1);

            os_s_row, os_s_col = np.where(stim[:,:5] == 2);
            os_r_row, os_r_col = np.where(stim[:,5:] == 4);
            os_r_col += 5

            row = {"Session" : S, "Round" : R, "SR_idx" :S*8+R,
                   "Sub_id" : sub_id, "M_id":m_name, "OS_idx" : os_s_row[0] == os_r_row[0],
                   "ns_row" : os_s_row[0], "ns_col" : os_s_col[0], "nr_row" : os_r_row[0],
                   "nr_col" : os_r_col[0], "is_Pos" : info_dict[session_str][round_str]["is_Pos"]};

            gr_ratings = info_dict[session_str][round_str]["gr_rating"];
            br_ratings = info_dict[session_str][round_str]["br_rating"];
            tr_ratings = info_dict[session_str][round_str]["tr_rating"];

            for idx, (gr_rating, br_rating, tr_rating) in enumerate(zip(gr_ratings, br_ratings, tr_ratings)) :
                row.update({
                    "gr_rating_{}".format(idx) : gr_rating,
                    "br_rating_{}".format(idx) : br_rating,
                    "tr_rating_{}".format(idx) : tr_rating,
                })

            rows.append(row)

    df_row = pd.DataFrame(rows);

    response_var = np.concatenate(
        [df_row.br_rating_0.values, df_row.br_rating_1.values, df_row.br_rating_2.values]).std();
    df_row["resp_var_br"] = response_var
    df_total = pd.concat([df_total, df_row], ignore_index=True);

    return df_total

if __name__ == '__main__':

    df_total = pd.DataFrame();

    per_plot = 6;

    save_name = "AMT_shuffle_driving.csv"
    data_root = "../../shuffle_backup_driving_LSTM_Attn2_Temp/data";
    meta_root = "../../shuffle_backup_driving_LSTM_Attn2_Temp/metadata";

    reject_ids = []

    meta_dir = os.listdir(meta_root);

    sub_ids = [];
    m_ids = [];

    for sub_files in os.listdir(data_root) :

        if sub_files.endswith(".json") :
            sub_id = sub_files[:-5]

            sub_ids.append(sub_id)
            for m in meta_dir:

                with open(os.path.join(meta_root, m), "r") as f:

                    lines = f.readlines();
                    if lines:
                        try :
                            sub_id_meta = lines[3].split("subId\t")[1];
                        except :
                            continue
                        if sub_id == sub_id_meta[:-1]:
                            m_name = m
                            m_ids.append(m_name)
                            break;

    for idx, (sub_id, m_id) in enumerate(zip(sub_ids, m_ids)) :

        inf_nps = json_to_nps(os.path.join(data_root,"{}".format(sub_id+".json")))

        df_total = append_df(df_total, inf_nps, sub_id, m_id);

    #df_total = df_total.loc[df_total.is_Pos == 1];

    df_total.to_csv(save_name)

    df_INCRE = df_total.loc[df_total.OS_idx==False];
    df_OS = df_total.loc[df_total.OS_idx==True];


    incre_0 = df_INCRE.br_rating_0/df_INCRE.resp_var_br;
    incre_1 = df_INCRE.br_rating_1/df_INCRE.resp_var_br;
    incre_2 = df_INCRE.br_rating_2/df_INCRE.resp_var_br;

    fig, axes = plt.subplots();

    test_01 = stats.ttest_rel(incre_0.values, incre_1.values)
    test_12 = stats.ttest_rel(incre_1.values, incre_2.values)

    print("Incre P-value 12 : {}".format(test_01.pvalue))
    print("Incre P-value 23 : {}".format(test_12.pvalue))
    x = ['Image1', 'Image2', "Image3"]
    y = [incre_0.mean(), incre_1.mean(), incre_2.mean()]
    sns.barplot(x,y)
    axes.set_ylim([0,10])
    axes.set_title("Incre")
    plt.show()

    os_0 = df_OS.br_rating_0/df_OS.resp_var_br;
    os_1 = df_OS.br_rating_1/df_OS.resp_var_br;
    os_2 = df_OS.br_rating_2/df_OS.resp_var_br;

    fig, axes = plt.subplots();

    test_01 = stats.ttest_rel(os_0.values, os_1.values)
    test_12 = stats.ttest_rel(os_1.values, os_2.values)

    print("OS P-value 12 : {}".format(test_01.pvalue))
    print("OS P-value 23 : {}".format(test_12.pvalue))
    x = ['Image1', 'Image2', "Image3"]
    y = [os_0.mean(), os_1.mean(), os_2.mean()]
    sns.barplot(x, y)
    axes.set_ylim([0, 10])
    axes.set_title("OS")
    plt.show()

    incre_os_idx = incre_2 - (incre_0 + incre_1)/2.0
    os_os_idx = os_2 - (os_0 + os_1) / 2.0

    x = ['Type 1', 'Type 2']
    y = [incre_os_idx.mean(), os_os_idx.mean()]
    sns.barplot(x, y)
    axes.set_title("OS_IDX")
    plt.show()

    df_total = df_total.loc[df_total.OS_idx==True];

    gr_bin = np.histogram_bin_edges(np.linspace(-5, 5, num=10), bins=10);
    br_bin = np.histogram_bin_edges(np.linspace(0, 10, num=10), bins=10);
    tr_bin = np.histogram_bin_edges(np.linspace(0.0, 1.0, num=10), bins=10);


    fig_num = len(sub_ids)//per_plot;

    sub_idx = 0;
    for fig_i in range(fig_num) :

        fig, axes = plt.subplots(per_plot,3);

        for f_i in range(per_plot) :

            df_sub = df_total.loc[df_total.Sub_id == sub_ids[sub_idx]]

            sns.histplot(x="gr_rating_0", data=df_sub,
                            color="C0", ax=axes[f_i,0], bins=gr_bin, alpha=0.5)
            sns.histplot(x="gr_rating_1", data=df_sub,
                            color="C1", ax=axes[f_i,0], bins=gr_bin,alpha=0.5)
            sns.histplot(x="gr_rating_2", data=df_sub,
                            color="C2", ax=axes[f_i,0], bins=gr_bin,alpha=0.5)

            sns.histplot(x="br_rating_0", data=df_sub,
                            color="C0", ax=axes[f_i,1], bins=br_bin, alpha=0.5)
            sns.histplot(x="br_rating_1", data=df_sub,
                            color="C1", ax=axes[f_i,1], bins=br_bin,alpha=0.5)
            sns.histplot(x="br_rating_2", data=df_sub,
                            color="C2", ax=axes[f_i,1], bins=br_bin,alpha=0.5)

            sns.histplot(x="tr_rating_0", data=df_sub,
                            color="C0", ax=axes[f_i,2], bins=tr_bin, alpha=0.5)
            sns.histplot(x="tr_rating_1", data=df_sub,
                            color="C1", ax=axes[f_i,2], bins=tr_bin,alpha=0.5)
            sns.histplot(x="tr_rating_2", data=df_sub,
                            color="C2", ax=axes[f_i,2], bins=tr_bin, alpha=0.5)

            axes[f_i,0].set_ylabel("gr_rating")
            axes[f_i, 0].set_xlabel("")
            axes[f_i,1].set_ylabel("br_rating")
            axes[f_i, 1].set_xlabel("")
            axes[f_i,2].set_ylabel("tr_rating")
            axes[f_i, 2].set_xlabel("")

            sub_idx+=1

            #axes[f_i,1].set_title("M_ID : {} || Sub_ID : {}".format(m_ids[sub_idx], sub_ids[sub_idx]))

        fig.suptitle("per_subjects {}".format(fig_i))
        plt.show()
    

    for reject_id in reject_ids :
        df_total = df_total.drop(df_total[df_total.Sub_id == reject_id].index)

    fig, axes = plt.subplots(1, 3);
    sns.histplot(x="gr_rating_0", data=df_total,
                 color="C0", ax=axes[0], bins=gr_bin, alpha=0.5)
    sns.histplot(x="gr_rating_1", data=df_total,
                 color="C1", ax=axes[0], bins=gr_bin, alpha=0.5)
    sns.histplot(x="gr_rating_2", data=df_total,
                 color="C2", ax=axes[0], bins=gr_bin, alpha=0.5)

    sns.histplot(x="br_rating_0", data=df_total,
                 color="C0", ax=axes[1], bins=br_bin, alpha=0.5)
    sns.histplot(x="br_rating_1", data=df_total,
                 color="C1", ax=axes[1], bins=br_bin, alpha=0.5)
    sns.histplot(x="br_rating_2", data=df_total,
                 color="C2", ax=axes[1], bins=br_bin, alpha=0.5)

    sns.histplot(x="tr_rating_0", data=df_total,
                 color="C0", ax=axes[2], bins=tr_bin, alpha=0.5)
    sns.histplot(x="tr_rating_1", data=df_total,
                 color="C1", ax=axes[2], bins=tr_bin, alpha=0.5)
    sns.histplot(x="tr_rating_2", data=df_total,
                 color="C2", ax=axes[2], bins=tr_bin, alpha=0.5)

    axes[0].set_ylabel("gr_rating")
    axes[0].set_xlabel("")
    axes[1].set_ylabel("br_rating")
    axes[1].set_xlabel("")
    axes[2].set_ylabel("tr_rating")
    axes[2].set_xlabel("")

    fig.suptitle("total_subjects")
    plt.show()

    fig, axes = plt.subplots(1, 5);

    for S in range(5) :

        sns.histplot(x="br_rating_0", data=df_total.loc[df_total.Session == S],
                     color="C0", ax=axes[S], bins=br_bin, alpha=0.5)
        sns.histplot(x="br_rating_1", data=df_total.loc[df_total.Session == S],
                     color="C1", ax=axes[S], bins=br_bin, alpha=0.5)
        sns.histplot(x="br_rating_2", data=df_total.loc[df_total.Session == S],
                     color="C2", ax=axes[S], bins=br_bin, alpha=0.5)

        axes[S].set_ylabel("br_rating")
        axes[S].set_xlabel("")
        axes[S].set_ylim([0,45])

    fig.suptitle("total_subjects_sequential")
    plt.show()


