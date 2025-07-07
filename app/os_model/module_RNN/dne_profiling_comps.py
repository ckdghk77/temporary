
import torch
import torch.nn.functional as F

from app.os_model.visualizer import vis_results_attn, vis_results_feature
import numpy as np
from torch.autograd import Variable

import random


if __name__ == '__main__':
    import argparse
    import os
    from app.os_model.data_loader import json_to_nps, load_os_data, append_df
    import torch.optim as optim
    from app.os_model.exp.Oneshot import Oneshot
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4, help='Random seed.')
    parser.add_argument('--var_tar', type=str, default="ablation", help='primacy|recency|lr|ablation')

    args = parser.parse_args();

    seed = args.seed;
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    var_tar = args.var_tar

    feature_dim=5;
    stim_ratio=[16,8,1];
    stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
    cue_ratio = [4,1]
    cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]

    if var_tar == "primacy":

        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.18, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.36, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.54, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.72, model="sangwan2014"),
        ]
    elif var_tar == "recency":
        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=0.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=0.18, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=0.36, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=0.54, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=0.72, model="sangwan2014"),
        ]

    elif var_tar == "lr":
        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=0.05, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=0.1, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=0.2, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=0.3, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=0.4, model="sangwan2014"),
        ]
    elif var_tar == "ablation":
        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, model="random"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=-1.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, recency=-1.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, lr=2.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, model="sangwan2014")
        ]

    for os_i, os_exp_fit in enumerate(os_exps) :
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

        df_result = pd.DataFrame();

        gen_data = dict();
        session_num=40;
        round_num=8;
        for sn in range(session_num) :
            gen_data["S{}".format(sn)] = dict()
            for rn in range(round_num) :
                round_dict = dict()
                x, _, _ = os_exp_fit.gen_episode(target=rn%2)
                round_dict['stim'] = x;

                for sub_level in range(len(os_exps)) :
                    y = os_exps[sub_level].model.perform(x);
                    round_dict['br_rating_{}'.format(sub_level)] = y[-1];
                    if os_i == sub_level :
                        round_dict['br_rating'] = y[-1];

                gen_data["S{}".format(sn)]["R{}".format(rn)] = round_dict

                pd_dict = {
                    "SR_idx" : sn*session_num+rn,
                    "OS_idx" : rn%2,
                    "Sub_id" : "sangwan",
                }

                for sub_level in range(len(os_exps)) :
                    sub_y = round_dict["br_rating_{}".format(sub_level)];
                    pd_dict["br_rating_{}_0".format(sub_level)] = sub_y[0]
                    pd_dict["br_rating_{}_1".format(sub_level)] = sub_y[1]
                    pd_dict["br_rating_{}_2".format(sub_level)] = sub_y[2]

                    if os_i == sub_level :
                        pd_dict["out_1"] = sub_y[2]
                        pd_dict["out_0"] = (sub_y[0] + sub_y[1])/2.0

                round_pd = pd.DataFrame(pd_dict, index=[0])
                df_result = pd.concat([df_result, round_pd], ignore_index=True);

            df_result.to_csv(os.path.join("model_profiling",
                                          var_tar,
                                          "profiling_exp_ans_{}_{}_{}.csv".format(var_tar, os_i, seed)))
