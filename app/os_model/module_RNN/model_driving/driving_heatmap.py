import pandas as pd
import numpy as np
import cv2
import pickle, os
from scipy.io import savemat

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
names = ["_PRIMACY(0.0)", "_PRIMACY(0.18)", "_PRIMACY(0.36)",
          "_PRIMACY(0.54)", "_PRIMACY(0.72)"]  # deviates
names = ["_PRIMACY(0.0)", "_PRIMACY(0.36)", "_PRIMACY(0.72)"]  # deviates
names = ["_RECENCY(0.72)", "_PRIMACY(0.72)"]  # deviates

names = [
    "_PRIMACY(1.0)", "_PRIMACY(0.72)", "_PRIMACY(0.36)",
    "_PRIMACY(0.0), RECENCY(0.0)",
    "_RECENCY(0.36)", "_RECENCY(0.72)","_RECENCY(1.0)"]  # deviates

#names = ["_PRIMACY(0.0)"]  # deviates


test_type = "os_cols"

var_tar = "contrast"
model_type = "LSTM_Attn2"
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

for outer_level, name in enumerate(names):

    with open("{}/exps_{}_{}_{}.pickle".format(var_tar, model_type, outer_level,seed), 'rb') as fp:
        exp_dict = pickle.load(fp);
    fname = "{}/control_exp_{}_{}_{}.csv".format(var_tar,model_type, outer_level, seed)

    df = pd.read_csv(fname)
    os_maps_so = []
    os_maps_cols = []
    os_maps_rows = []

    for s_idx, session_id in enumerate(sorted(exp_dict.keys())):
        os_maps_s = []

        exp_s = exp_dict[session_id];
        df_sub = df.loc[(df.Session_id==s_idx)];

        for r_idx in df_sub["Round_id"].values :
            if test_type in ["os_rows", "os_cols"] :
                os_map = exp_s["R{}".format(r_idx)]["stim"].argmax(-1)==2;
            elif test_type in ["inc2_rows", "inc2_cols"] :
                os_map = exp_s["R{}".format(r_idx)]["stim"].argmax(-1)==1;

            os_maps_s.append(os_map)

        os_maps_s = np.stack(os_maps_s).mean(0);

        os_maps_so.append(os_maps_s)
        os_maps_cols.append(os_maps_s.sum(0))
        os_maps_rows.append(os_maps_s.sum(1))

    tot_os_maps.append(os_maps_so)
    tot_os_cols.append(os_maps_cols)
    tot_os_rows.append(os_maps_rows)


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




