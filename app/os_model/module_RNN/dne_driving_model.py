
import torch
import torch.nn.functional as F

import os,sys
path = os.getcwd()
sys.path.append(path)
sys.path.append('/home/chlee/webserver/os-mturk')

from app.os_model.visualizer import vis_results_attn
import numpy as np
from torch.autograd import Variable
import pickle
import random


def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1)


def iterate_epoch(data_loader, network=None, optim=None, eval=False, save_img=False, device="cuda",
                  save_feature=False, root_img_dir="incre") :

    if network is not None :
        net = network
    if optim is not None :
        optimizer = optim

    net.eval() if eval else net.train()

    accs = []
    ces = []

    for batch_idx, (data, data_oh, idx_os, label, task_vec) in enumerate(data_loader):

        data = data.contiguous()
        data, data_oh, label, task_vec = data.to(device), data_oh.to(device), label.to(device), task_vec.to(device)

        #data, data_oh, label = Variable(data), Variable(data_oh), Variable(label)

        logits = net(data, data_oh, idx_os, task_vec);
        #logits = F.softmax(logits)
        #loss = softXEnt(logits, label).mean();
        loss = F.mse_loss(logits[:,1], label[:,1])

        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();

        if save_img :
            with torch.no_grad() :

                attention = net.get_attention(data, idx_os, task_vec);

                B, A, S = attention.shape
                attention = attention.reshape(B,A, 5,6).cpu().data.numpy();

                B, S = data.shape;

                inputs = data.reshape(B,5,6).cpu().data.numpy()

                for h_i in range(attention.shape[1]):
                    vis_results_attn(inputs, attention, head_idx=h_i, save=True,
                                     fname="im_{}_{}.png".format(batch_idx, h_i), root_dir=root_img_dir)


        prediction = torch.argmax(logits, dim=-1);
        correct = (prediction == label.argmax(dim=-1));
        #prediction = logits > 0.5
        #correct = (prediction == (label>0.5));
        del data, data_oh, label, task_vec
        accs.extend(correct.detach().cpu().numpy());
        ces.append(loss.detach().cpu().numpy());

    accs = 100 * np.mean(accs);
    return accs, np.mean(ces),


import torch.optim as optim
import pandas as pd


from app.os_model.module_RNN.sublayers import CosineAnnealingWarmupRestarts
from app.os_model.data_loader import json_to_nps, load_os_data, append_df
from app.os_model.exp.Oneshot import Oneshot



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model', type=str, default="LSTM_Attn2", help='model')
    parser.add_argument('--var_tar', type=str, default="lr", help='contrast')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

    args = parser.parse_args();

    seed = args.seed;
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = args.model
    var_tar = args.var_tar

    feature_dim=5;
    stim_ratio=[16,8,1];
    stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
    cue_ratio = [4,1]
    cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]

    if var_tar == "contrast"  :
        opt_os_exp = Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                             cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0,
                             model="sangwan2014")

        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=1.0, recency=0.0,  model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.72, recency=0.0,  model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.36, recency=0.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.36, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.72, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=1.0, model="sangwan2014"),]

    #lr=0.1, primacy=0.36, recency=0.36,
    elif var_tar == "lr" :
        opt_os_exp = Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                             cue_ratio=cue_ratio, cue_vectors=cue_vectors,
                             model="sangwan2014")
        os_exps = [
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, lr=0.0, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, lr=0.05, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, lr=0.1, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, lr=0.15, model="sangwan2014"),
            Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.0, recency=0.0, lr=0.2, model="sangwan2014"),]

    for os_i, os_exp_fit in enumerate(os_exps) :
        net = None
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

        df_result = pd.DataFrame();

        gen_data = dict();
        train_data = dict()

        session_num=8;
        round_num=8;
        for sn in range(session_num) :
            gen_data["S{}".format(sn)] = dict()
            train_data["S{}".format(sn)] = dict()
            for rn in range(100) :
                round_dict = dict()

                x, _, (cue_alloc, rew_alloc) = os_exp_fit.gen_episode(net, target=rn%2, opt_model=opt_os_exp)
                round_dict['stim'] = x;

                for sub_level in range(len(os_exps)) :
                    y = os_exps[sub_level].model.perform(x);
                    round_dict['br_rating_{}'.format(sub_level)] = y[-1];
                    if os_i == sub_level :
                        round_dict['br_rating'] = y[-1];

                pd_dict = {
                    "SR_idx": sn * session_num + rn,
                    "Session_id": sn,
                    "Round_id": rn,
                    "OS_idx": rn % 2,
                    "Sub_id": "sangwan",
                }

                for sub_level in range(len(os_exps)):
                    sub_y = round_dict["br_rating_{}".format(sub_level)];
                    pd_dict["br_rating_{}_0".format(sub_level)] = sub_y[0]
                    pd_dict["br_rating_{}_1".format(sub_level)] = sub_y[1]
                    pd_dict["br_rating_{}_2".format(sub_level)] = sub_y[2]

                if rn < round_num :
                    train_data["S{}".format(sn)]["R{}".format(rn)] = round_dict

                gen_data["S{}".format(sn)]["R{}".format(rn)] = round_dict
                round_pd = pd.DataFrame(pd_dict, index=[0])
                df_result = pd.concat([df_result, round_pd], ignore_index=True);

            train_loader, _, _, _, _ = load_os_data([train_data],
                                                                               total_session=np.arange(sn+1),
                                                                                train_session=np.arange(sn+1),
                                                                                val_session=np.arange(sn+1),
                                                                                batch_size=8)
            _, valid_loader, incre_loader, os_loader, (exp_xs, exp_oh_xs, exp_ys, exp_vecs) = load_os_data( [gen_data],
                                                                        total_session=np.arange(sn + 1),
                                                                        train_session=np.arange(sn + 1),
                                                                        val_session=np.arange(sn + 1),
                                                                        batch_size=8)

            del net
            if model == "Transformer":
                from app.os_model.module_RNN.transformer import Transformer

                net = Transformer(in_channel=5, d_embed=5, d_inner=6, d_k=4, d_v=4, e_layers=2, d_layers=1,
                                  n_head=4, dropout=args.dropout, n_position=30, class_num=2);
            elif model == "MLP":
                from app.os_model.module_RNN.dnns import MLP

                net = MLP(in_channel=5, n_position=30, dropout=args.dropout, class_num=2)

            elif model == "SLP":
                from app.os_model.module_RNN.dnns import SLP

                net = SLP(in_channel=5, n_position=30, dropout=args.dropout, class_num=2)

            elif model == "LSTM":
                from app.os_model.module_RNN.dnns import LSTM

                net = LSTM(in_channel=5, dropout=args.dropout, class_num=2)

            elif model == "BiLSTM":
                from app.os_model.module_RNN.dnns import BiLSTM

                net = BiLSTM(in_channel=5, dropout=args.dropout, class_num=2)

            elif model == "BiLSTM_Attn":
                from app.os_model.module_RNN.dnns import BiLSTM_Attn

                net = BiLSTM_Attn(in_channel=5, dropout=args.dropout, class_num=2)

            elif model == "LSTM_Attn":
                from app.os_model.module_RNN.dnns import LSTM_Attn

                net = LSTM_Attn(in_channel=5, dropout=args.dropout, class_num=2)

            elif model == "BiLSTM_Attn2":
                from app.os_model.module_RNN.dnns import BiLSTM_Attn2

                net = BiLSTM_Attn2(in_channel=5, dropout=args.dropout, class_num=2)

            elif model == "LSTM_Attn2":
                from app.os_model.module_RNN.dnns import LSTM_Attn2

                net = LSTM_Attn2(in_channel=5, dropout=args.dropout, class_num=2)


            elif model == "CNN":
                from app.os_model.module_RNN.dnns import CNN

                net = CNN(in_channel=5, dropout=args.dropout, class_num=2)

            #net = net.cuda()
            net = net.to("cuda:0")
            lr = 0.01
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                         cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                         warmup_steps=150, gamma=0.5)
            for epoch in range(1000):
                lr_scheduler.step()
                acc_train, ce_train = iterate_epoch(train_loader, network=net, optim=optimizer, eval=False)

                if epoch % 200 == 0:
                    acc_val, ce_val = iterate_epoch(valid_loader, network=net, optim=optimizer, eval=True)

                    print("E{} || Train ACCs : {:3f}, Train CEs : {:3f}, "
                          " Val ACCs : {:3f},  Val CEs : {:3f}".format(epoch, acc_train, ce_train,
                                                                       acc_val, ce_val))

                    '''
                    acc_val_incre, ce_val_incre = iterate_epoch(incre_loader, network=net, optim=optimizer, eval=True, save_img=True,
                                                          root_img_dir="./fig_attn/incre");
                    acc_val_os, ce_val_os = iterate_epoch(os_loader, network=net, optim=optimizer, eval=True, save_img=True,
                                                    root_img_dir="./fig_attn/os");
                    '''

                if epoch % 200 == 0 :
                    with torch.no_grad() :

                        exp_xs = exp_xs.cuda();
                        exp_oh_xs = exp_oh_xs.cuda();
                        exp_vecs = exp_vecs.cuda()
                        idx_os = torch.where(exp_xs == 4);
                        idx_os = torch.stack(idx_os)
                        idx_os = idx_os[-1, :];

                        #output = F.softmax(net(exp_xs, idx_os, exp_vecs), -1);
                        output = net(exp_xs, exp_oh_xs, idx_os, exp_vecs);
                        '''
                        attention = net.get_attention(exp_xs, idx_os, task_vec=exp_vecs, t_layer=0);

                        exp_xs_reshape = exp_xs.reshape(len(exp_xs), 5, 6).cpu().data.numpy()
                        attn_xs_reshape = attention.reshape(len(exp_xs), attention.shape[1], 5, 6).cpu().data.numpy()

                        for exp_idx in range(len(exp_xs_reshape)) :
                            for h_i in range(attention.shape[1]):
                                vis_results_attn(exp_xs_reshape[exp_idx][np.newaxis],
                                                 attn_xs_reshape[exp_idx][np.newaxis], head_idx=h_i, save=True,
                                                 fname="im_{}_{}.png".format(exp_idx, h_i),
                                                 root_dir="fig_attn/{}".format("sangwan"))

                        for cc in range(3) : #0,1,2
                            cc_attn_tot = []
                            for h_i in range(attention.shape[1]):
                                cc_attn = torch.stack([at[torch.where(ex==cc)] for e_idx, (ex, at)
                                                          in enumerate(zip(exp_xs, attention[:,h_i]))]);
                                cc_attn_tot.append(cc_attn)

                            df_result["attn_{}".format(cc)] = torch.stack(cc_attn_tot).sum(0).sum(-1).cpu().data.numpy()
                        '''
                        for oi in range(2) :
                            df_result["out_{}".format(oi)] = output[:,oi].cpu().data.numpy()

                        df_result.to_csv(os.path.join("model_driving",
                                     var_tar,
                                    "control_exp_{}_{}_{}.csv".format(model, os_i, seed)))
                        with open(os.path.join("model_driving",
                                               var_tar,
                                               "exps_{}_{}_{}.pickle".format(model, os_i, seed)), 'wb') as fp:
                            pickle.dump(gen_data, fp)

        del net
        del optimizer
        torch.cuda.empty_cache()
