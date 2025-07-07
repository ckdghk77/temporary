
import torch
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

import random
from tqdm import tqdm
import os,sys
path = os.getcwd()
sys.path.append(path)
sys.path.append('/home/chlee/webserver/os-mturk')



def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1)


def forward(data_loader, eval=False, save_img=False, device="cuda", save_feature=False, root_img_dir="incre") :

    net.eval() if eval else net.train()

    accs = []
    ces = []

    for batch_idx, (data, data_oh, idx_os, label, task_vec) in enumerate(data_loader):

        data = data.contiguous()

        data, data_oh, label, task_vec = data.to(device), data_oh.to(device), label.to(device), task_vec.to(device)

        data, data_oh, label = Variable(data), Variable(data_oh), Variable(label)

        logits = net(data, data_oh, idx_os, task_vec);
        #logits = F.softmax(logits)
        #loss = softXEnt(logits, label).mean();
        loss = F.mse_loss(logits[:,1], label[:,1])

        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();

        '''
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
        '''

        prediction = torch.argmax(logits, dim=-1);
        correct = (prediction == label.argmax(dim=-1));
        #prediction = logits > 0.5
        #correct = (prediction == (label>0.5));

        accs.extend(correct.detach().cpu().numpy());
        ces.append(loss.detach().cpu().numpy());

    accs = 100 * np.mean(accs);
    return accs, np.mean(ces),


if __name__ == '__main__':

    import argparse
    import os
    from app.os_model.data_loader import json_to_nps, load_os_data, append_df
    from app.os_model.module_RNN.sublayers import CosineAnnealingWarmupRestarts
    import torch.optim as optim
    from app.os_model.exp.Oneshot import Oneshot
    import pandas as pd


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model', type=str, default="LSTM_Attn2", help='model')
    parser.add_argument('--sbj_tar', type=str, default="shuffle_backup_driving_LSTM_Attn2_Temp",
                        help='shuffle_backup_driving_LSTM_Attn2_Temp')

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
    sbj_tar = os.path.join("../../../", args.sbj_tar)

    feature_dim=5;
    stim_ratio=[16,8,1];
    stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                   np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                   np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
    cue_ratio = [4,1]
    cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]


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

    for s_i, (sub_id, m_id) in enumerate(tqdm(zip(sub_ids, m_ids))):
        model_file = os.path.join(model_root, m_id, "_model_5.pt")

        df_result = pd.DataFrame();

        gen_data = dict();
        session_num=5;
        round_num=8;
        data_subjs = json_to_nps(os.path.join(model_root, sub_id, sub_id+"_4.json"))
        for sn in range(session_num) :

            gen_data["S{}".format(sn)] = dict()
            for rn in range(round_num) :
                round_dict = dict()
                x = data_subjs["S{}".format(sn)]["R{}".format(rn)]['stim']
                round_dict['stim'] = x
                round_dict['br_rating'] = data_subjs["S{}".format(sn)]["R{}".format(rn)]['br_rating']

                gen_data["S{}".format(sn)]["R{}".format(rn)] = round_dict

        train_loader, valid_loader, incre_loader, os_loader, \
        (exp_xs, exp_oh_xs, exp_ys, exp_vecs) = load_os_data([gen_data],
                                                               total_session=np.arange(session_num),
                                                                train_session=np.arange(session_num),
                                                                val_session=np.arange(session_num),
                                                                batch_size=8)

        if model == "Transformer" :
            from app.os_model.module_RNN.transformer import Transformer
            net = Transformer(in_channel=5, d_embed=5, d_inner=6, d_k=4, d_v=4, e_layers=2, d_layers=1,
                              n_head=4, dropout=args.dropout, n_position=30, class_num=2);
        elif model == "MLP" :
            from app.os_model.module_RNN.dnns import MLP
            net = MLP(in_channel=5, n_position=30, dropout=args.dropout, class_num=2)

        elif model == "SLP" :
            from app.os_model.module_RNN.dnns import SLP
            net = SLP(in_channel=5, n_position=30, dropout=args.dropout, class_num=2)

        elif model == "LSTM" :
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

        elif model == "CNN" :
            from app.os_model.module_RNN.dnns import CNN
            net = CNN(in_channel=5, dropout=args.dropout, class_num=2)

        lr = 0.01
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                     cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                     warmup_steps=150, gamma=0.5)

        net.load_model(fname=model_file, optimizer=optimizer, lr_scheduler=lr_scheduler)

        net = net.cuda()
        with torch.no_grad() :
            exp_xs = exp_xs.cuda();
            exp_oh_xs = exp_oh_xs.cuda();
            exp_vecs = exp_vecs.cuda()
            idx_os = torch.where(exp_xs == 4);
            idx_os = torch.stack(idx_os)
            idx_os = idx_os[-1, :];

            #output = F.softmax(net(exp_xs, idx_os, exp_vecs), -1);
            output = net(exp_xs, exp_oh_xs, idx_os, exp_vecs);
            output_attn = net.forward_attn(exp_xs, exp_oh_xs, idx_os, exp_vecs);


            for oi in range(2) :
                df_result["out_{}".format(oi)] = output[:,oi].cpu().data.numpy()

            #df_result["sub_id"] = sub_id
            #df_result["sub_idx"] = s_i
            np.save(os.path.join("subject_profiling",
                                          args.sbj_tar,
                                          "attention_{}.npy".format(sub_id)),
                                output_attn.cpu().data.numpy())

            np.save(os.path.join("subject_profiling",
                                          args.sbj_tar,
                                          "out_{}.npy".format(sub_id)),
                                output.cpu().data.numpy())

        del net
        del optimizer
        torch.cuda.empty_cache()
