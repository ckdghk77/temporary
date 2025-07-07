import pandas as pd
import numpy as np
import cv2
import pickle, os
from scipy.io import savemat
from app.os_model.data_loader import json_to_nps

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
var_tar = "temp"

if var_tar == "primacy" :
    groups = [[0, 5, 6, 7, 12, 18, 19, 21, 22],
              [3, 4, 9, 11, 14, 23]]
elif var_tar == "recency" :
    groups = [[3, 4, 9, 11, 12, 19, 23],
              [0, 5, 6, 7, 14, 18, 21, 22]]
elif var_tar == "lr" :
    groups = [[12, 14, 23],
              [0, 3, 5, 6, 7, 9, 18, 19, 21, 22]]
elif var_tar == "temp" :
    groups = [[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
              ]

model_type = "LSTM_Attn2"
sbj_tar = "shuffle_backup_driving_LSTM_Attn2_Temp"
sbj_tar = os.path.join("../../../../", sbj_tar)

data_root = os.path.join(sbj_tar, "data")
meta_root = os.path.join(sbj_tar, "metadata")
model_root = os.path.join(sbj_tar, "incomplete")
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

tot_attn_correls = []
tot_attn_ps = []

tot_out_correls = []
tot_out_ps = []
tot_os_maps = []
tot_os_cols = []
tot_os_rows = []

W = 128;
H = 128;
seed=0

for g_idx, group in enumerate(groups):

    group_maps_so = []
    group_maps_cols = []
    group_maps_rows = []

    for sbj in group :

        data_sbj = json_to_nps(os.path.join(model_root, sub_ids[sbj], sub_ids[sbj]+"_4.json"))

        os_maps_so = []
        os_maps_cols = []
        os_maps_rows = []

        for s_idx, session_id in enumerate(sorted(data_sbj.keys())):
            os_maps_s = []

            exp_s = data_sbj[session_id];

            for r_idx in range(8) :

                if test_type in ["os_rows", "os_cols"] :
                    os_map = exp_s["R{}".format(r_idx)]["stim"].argmax(-1)==2;
                elif test_type in ["inc2_rows", "inc2_cols"] :
                    os_map = exp_s["R{}".format(r_idx)]["stim"].argmax(-1)==1;

                onehot_stim = exp_s["R{}".format(r_idx)]["stim"].argmax(-1);

                row_oh_stim, col_oh_stim = np.where(onehot_stim == 2);
                row_oh_rew, col_oh_rew = np.where(onehot_stim == 4);

                if row_oh_stim != row_oh_rew :
                    continue;

                os_maps_s.append(os_map)

            os_maps_s = np.stack(os_maps_s).mean(0);

            os_maps_so.append(os_maps_s)
            os_maps_cols.append(os_maps_s.sum(0))
            os_maps_rows.append(os_maps_s.sum(1))

        group_maps_so.append(np.stack(os_maps_so))
        group_maps_cols.append(np.stack(os_maps_cols))
        group_maps_rows.append(np.stack(os_maps_rows))

    tot_os_maps.append(np.stack(group_maps_so).mean(0))
    tot_os_cols.append(np.stack(group_maps_cols).mean(0))
    tot_os_rows.append(np.stack(group_maps_rows).mean(0))

nSession = len(os_maps_so);
nExps = len(tot_os_maps);

w_per_one = W
h_per_one = H * nSession + 10*nSession

tot_imgs = []
tot_nps = []
for os_maps_so in tot_os_maps :
    cat_img = np.zeros(shape=(w_per_one,
                              h_per_one, 3), dtype=np.uint8);
    cat_nps = []
    for s_idx, os_map in enumerate(os_maps_so) :
        os_map = ((os_map/os_map.max()) * 255.0).astype(np.uint8)
        cat_nps.append(os_map)

        hmap_c = cv2.applyColorMap(os_map, cv2.COLORMAP_BONE);
        hmap_c = cv2.resize(hmap_c, dsize=(W, H),
                            interpolation=cv2.INTER_NEAREST);

        cat_img[:, s_idx*H + 10*s_idx:(s_idx+1)*H + 10*s_idx] = hmap_c

    tot_imgs.append(cat_img)
    tot_nps.append(np.stack(cat_nps))

tot_nps = np.stack(tot_nps)
tot_img = np.zeros(shape=((w_per_one + 10) * nExps,
                          h_per_one, 3), dtype=np.uint8);

for row in range(nExps) :
    img_idx = row;

    tot_img[img_idx * 10 + w_per_one * img_idx:
            img_idx * 10 + w_per_one * img_idx + w_per_one,
    :] = tot_imgs[img_idx]

# tot_img = Image.fromarray(tot_img);
cv2.imwrite("./{}/heatmap_{}.png".format(var_tar, test_type), tot_img)

import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=nExps,ncols=nSession);
plt.subplots_adjust(wspace=0.1)

tot_ys = []

for row in range(nExps) :
    row_ys = []
    for s in range(nSession) :
        if test_type in ["os_cols", "inc2_cols"] :
            col_infos = tot_os_cols[row][s];
        elif test_type in ["os_rows", "inc2_rows"] :
            col_infos = tot_os_rows[row][s];

        row_ys.append(col_infos)
        img = tot_nps[row,s]
        hmap = cv2.applyColorMap(img, cv2.COLORMAP_BONE);


        axes[row, s].plot((np.arange(len(col_infos))+0.5)*8, col_infos*30, c="r", alpha=0.6)
        axes[row,s].imshow(hmap, extent=[0,48, 0, 30])
        axes[row, s].scatter((np.arange(len(col_infos))+0.5)*8, col_infos*30, c="r",s=4, alpha=1.0)

        axes[row, s].set_ylim([0,30]);
        axes[row, s].set_xlim([0,len(col_infos)*8]);
        axes[row, s].set_xticks([]);
        axes[row, s].set_yticks([]);
        #axes[s].plot(col_infos)
        #axes[s].set_ylim([0,1]);
        #axes[s].set_xticks([]);
        #axes[s].set_yticks([]);
    tot_ys.append(np.stack(row_ys))

tot_ys = np.stack(tot_ys);

savemat(file_name="./{}/info_mat_{}_{}.mat".format(var_tar, model_type, test_type), mdict={"imgs" : tot_nps, "ys" : tot_ys})
#np.save("./{}/np_imgs_{}.npy".format(var_tar, test_type), tot_nps)
#np.save("./{}/np_ys_{}.npy".format(var_tar, test_type), tot_ys)

plt.savefig("./{}/quant_{}.png".format(var_tar, test_type), transparent=True)




