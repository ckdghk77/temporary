
import torch
import torch.nn.functional as F

from app.os_model.visualizer import vis_results_attn, vis_results_feature
import numpy as np
from torch.autograd import Variable



def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1)


def forward(data_loader, eval=False, save_img=False, save_feature=False, root_img_dir="incre") :

    net.eval() if eval else net.train()

    accs = []
    ces = []

    for batch_idx, (data, idx_os, label, task_vec) in enumerate(data_loader):

        data = data.contiguous()

        data, label, task_vec = data.cuda(), label.cuda(), task_vec.cuda()

        data, label = Variable(data), Variable(label)

        logits = net(data, idx_os, task_vec);

        #loss = softXEnt(logits, label).mean();
        loss = F.mse_loss(logits, label)

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

        accs.extend(correct.detach().cpu().numpy());
        ces.append(loss.detach().cpu().numpy());

    accs = 100 * np.mean(accs);
    return accs, np.mean(ces),


if __name__ == '__main__':
    import os
    from app.os_model.data_loader import json_to_nps, load_os_data, append_df
    import torch.optim as optim
    from app.os_model.exp.Oneshot import Oneshot
    import pandas as pd

    os_exp = Oneshot(feature_dim=5, stim_ratio=[16, 8, 1], stim_vectors=
    [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
     np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
     np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])],
             cue_ratio=[4, 1], cue_vectors=[np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])], model="sangwan2014");

    data_root = "../../../data_0901";
    meta_root = "../../../meta_0901";
    meta_dir = os.listdir(meta_root);
    sub_ids = os.listdir(data_root);

    #sub_ids = sub_ids[3:4]

    m_ids = []
    inf_np_all = []

    df_subs = pd.read_csv('../data_0901.csv')

    for sub_id in sub_ids :

        sub_id = sub_id.split(".json")[0]

        for m in meta_dir:

            with open(os.path.join(meta_root, m), "r") as f:

                lines = f.readlines();
                if lines:
                    sub_id_meta = lines[3].split("subId\t")[1];
                    if sub_id == sub_id_meta[:-1]:
                        m_name = m
                        m_ids.append(m_name)
                        break;

    for idx, (sub_id, m_id) in enumerate(zip(sub_ids, m_ids)):
        print(sub_id)
        inf_nps = json_to_nps(os.path.join(data_root, "{}".format(sub_id)))
        #inf_np_all.append(inf_nps)

        train_loader, valid_loader, incre_loader, os_loader, (exp_xs, exp_ys, exp_vecs) = load_os_data([inf_nps],
                                                                            total_session=[0,1,2,3,4],
                                                                            train_session=[1,2,3,4],
                                                                            val_session=[4],
                                                                            batch_size=8)

        from app.os_model.module_RNN.transformer import Transformer
        net = Transformer(in_channel=5, d_embed=3, d_model=3,
                          d_inner=3, d_k=3, d_v=3, e_layers=1, d_layers=1,
                          n_head=2, dropout=0.0, n_position=30, class_num=2);

        net = net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.98));

        for epoch in range(3000):

            acc_train, ce_train = forward(train_loader, eval=False)

            if epoch % 500 == 0:
                acc_val, ce_val = forward(valid_loader, eval=True)

                print("E{} || Train ACCs : {:3f}, Train CEs : {:3f}, "
                      " Val ACCs : {:3f},  Val CEs : {:3f}".format(epoch, acc_train, ce_train,
                                                                   acc_val, ce_val))

                acc_val_incre, ce_val_incre = forward(incre_loader, eval=True, save_img=True,
                                                      root_img_dir="./fig_attn/incre");
                acc_val_os, ce_val_os = forward(os_loader, eval=True, save_img=True,
                                                root_img_dir="./fig_attn/os");

            if epoch % 200 == 0 :
                with torch.no_grad() :

                    exp_xs = exp_xs.cuda();
                    exp_vecs = exp_vecs.cuda()
                    idx_os = torch.where(exp_xs == 4);
                    idx_os = torch.stack(idx_os)
                    idx_os = idx_os[-1, :];

                    #output = F.softmax(net(exp_xs, idx_os, exp_vecs),-1);
                    output = net(exp_xs, idx_os, exp_vecs);
                    attention = net.get_attention(exp_xs, idx_os, task_vec=exp_vecs, t_layer=0);

                    exp_xs_reshape = exp_xs.reshape(len(exp_xs), 5, 6).cpu().data.numpy()
                    attn_xs_reshape = attention.reshape(len(exp_xs), attention.shape[1], 5, 6).cpu().data.numpy()

                    for exp_idx in range(len(exp_xs_reshape)) :
                        for h_i in range(attention.shape[1]):
                            vis_results_attn(exp_xs_reshape[exp_idx][np.newaxis],
                                             attn_xs_reshape[exp_idx][np.newaxis], head_idx=h_i, save=True,
                                             fname="im_{}_{}.png".format(exp_idx, h_i),
                                             root_dir="fig_attn/{}".format(sub_id.split(".")[0]))

                    for cc in range(3):  # 0,1,2
                        cc_attn_tot = []
                        for h_i in range(attention.shape[1]):
                            cc_attn = torch.stack([at[torch.where(ex == cc)] for e_idx, (ex, at)
                                                   in enumerate(zip(exp_xs, attention[:, h_i]))]);
                            cc_attn_tot.append(cc_attn)

                        df_subs.loc[df_subs.Sub_id == sub_id.split(".")[0],
                                    "attn_{}".format(cc)] = torch.stack(cc_attn_tot).sum(0).mean(-1).cpu().data.numpy()

                    for oi in range(2) :
                        df_subs.loc[df_subs.Sub_id == sub_id.split(".")[0],
                                    "out_{}".format(oi)] = output[:,oi].cpu().data.numpy()

                df_subs.to_csv("data_0901_me.csv")





