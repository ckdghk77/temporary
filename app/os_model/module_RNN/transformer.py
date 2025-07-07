
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.os_model.module_RNN.sublayers import MultiHeadAttention, PositionwiseFeedForward
from app.os_model.visualizer import vis_results_attn, vis_results_feature
import numpy as np
from torch.autograd import Variable

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_hid, n_row=5, n_col=6):
        super(PositionalEncoding2D, self).__init__()
        self.n_row = n_row
        self.n_col = n_col

        # Not a parameter
        self.register_buffer('pos_table_row', self._get_sinusoid_encoding_table(n_row, d_hid))
        self.register_buffer('pos_table_col', self._get_sinusoid_encoding_table(n_col, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        pos_row = self.pos_table_row[:, np.arange(x.size(1)) // self.n_col];
        #pos_col = self.pos_table_col[:,np.arange(x.size(1)) - self.n_row*(np.arange(x.size(1))//self.n_row)];
        pos_col = self.pos_table_col[:,np.arange(x.size(1)) - self.n_col*(np.arange(x.size(1))//self.n_col)];

        return torch.cat([x, pos_row.repeat(x.shape[0],1,1), pos_col.repeat(x.shape[0],1,1)], dim=-1)


class TaskEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TaskEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('task_table', self._get_sinusoid_encoding_table(n_row, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        pos_row = self.pos_table_row[:, np.arange(x.size(1)) // self.n_col];
        pos_col = self.pos_table_col[:,np.arange(x.size(1)) - self.n_row*(np.arange(x.size(1))//self.n_row)];

        return torch.cat([x, pos_row.repeat(x.shape[0],1,1), pos_col.repeat(x.shape[0],1,1)], dim=-1)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn,


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_stim, d_embed, n_layers, n_head, d_k, d_v,
            d_inner, dropout=0.1, n_position=200, position_dim=1, task_dim=1, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_stim, d_embed, padding_idx=None)
        self.position_enc = PositionalEncoding2D(position_dim, n_row=5, n_col=6)
        #self.task_enc = TaskEncoding(task_dim);

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_embed + position_dim*2 + task_dim, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_embed + position_dim*2 + task_dim, eps=1e-6)
        self.scale_emb = scale_emb

    def forward(self, src_seq, task_vec, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = src_seq
        #if self.scale_emb:
        #    enc_output *= self.d_model ** 0.5
        enc_output = self.position_enc(enc_output)
        enc_output = torch.cat([enc_output, task_vec], dim=-1);
        #enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 None)
            #enc_output, enc_slf_attn = enc_layer(enc_output)

            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_embed, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(d_embed, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        #dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    def __init__(self, in_channel=1, d_embed=32, d_inner=64, d_k=16, d_v=16,
                 e_layers=2, d_layers=2, n_head=4, dropout=0.1,
                 n_position=30, class_num=2,
                 trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
                 scale_emb_or_prj='prj', device="cuda"):
        super(Transformer, self).__init__()

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.device = device

        position_dim=2
        task_dim= 2

        self.encoder = Encoder(
            n_src_stim = in_channel, n_position=n_position, position_dim=position_dim,
            task_dim=task_dim,
            d_embed = d_embed, d_inner= d_inner,
            n_layers = e_layers, n_head = n_head, d_k = d_k, d_v = d_v,
            dropout=dropout, scale_emb=scale_emb);

        self.out = nn.Linear((d_embed+position_dim*2+task_dim)*1, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

        '''
        for m in self.i_embed.modules():
            if isinstance(m, nn.Linear):
                m.weight.data[:, -self.ICovar_dim:].fill_(0.01)
        '''

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):
        enc_feature = self.forward_feature(src_seq_oh, task_vec);
        tar_idx = tar_idx.squeeze(-1)
        tar_feature = [ef[ti] for ef, ti in zip(enc_feature, tar_idx)]
        tar_feature = torch.stack(tar_feature)
        logits = self.out(tar_feature);

        return logits

    '''
    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):
        enc_feature = self.forward_feature(src_seq_oh, task_vec);
        tar_idx = tar_idx.squeeze(-1)
        tar_feature = [ef[ti] for ef, ti in zip(enc_feature, tar_idx)]
        tar_feature = torch.stack(tar_feature)
        #tar_feature = enc_feature.mean(1);
        outcomes_all = []
        for ti in tar_idx:
            outcome_idxes = [5, 11, 17, 23, 29];
            outcome_idxes.remove(ti.item())
            outcomes_all.append(outcome_idxes)
        nontar_feature = [ef[oa] for ef, oa in zip(enc_feature, outcomes_all)]
        nontar_feature = torch.stack(nontar_feature);
        #nontar_feature = nontar_feature.view(nontar_feature.shape[0], -1)
        nontar_feature = nontar_feature.mean(1)

        tot_feature = torch.cat([tar_feature, nontar_feature], dim=1);

        logits = self.out(tot_feature);

        return logits
    '''
    def forward_feature(self, src_seq, task_vec):
        enc_output, *_ = self.encoder(src_seq, task_vec, return_attns=False)

        return enc_output

    def get_attention(self, src_seq, src_seq_oh, tar_idx, task_vec, t_layer=0) :

        _, attention = self.encoder(src_seq_oh, task_vec, return_attns=True)
        tar_idx = tar_idx.squeeze(-1)

        t_attention = attention[t_layer];
        t_attention = [ta[:,ti] for ta, ti in zip(t_attention, tar_idx)]

        return torch.stack(t_attention)

    def save_model(self, fname, optimizer=None):
        torch.save({'model' : self.state_dict(),
                    'optimizer' : optimizer.state_dict()}, fname);

    def load_model(self, fname, optimizer=None):

        chkpt = torch.load(fname);
        self.load_state_dict(chkpt['model']);
        optimizer.load_state_dict(chkpt['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)



def forward(data_loader, eval=False, save_img=False, save_feature=False, root_img_dir="incre") :

    net.eval() if eval else net.train()

    accs = []
    ces = []

    for batch_idx, (data, label) in enumerate(data_loader):

        data = data.contiguous()

        data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        logits = net(data);

        loss = softXEnt(logits, label).mean();

        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();

        if save_img :
            with torch.no_grad() :

                attention = net.get_attention(data);
                b, r = torch.where(data == 4);
                attention = attention[b, :, r];
                B, A, S = attention.shape
                attention = attention.reshape(B,A, 5,6).cpu().data.numpy();

                B, S = data.shape;

                inputs = data.reshape(B,5,6).cpu().data.numpy()

                for h_i in range(attention.shape[1]):
                    vis_results_attn(inputs, attention, label, head_idx=h_i, save=True,
                                     fname="im_{}_{}.png".format(batch_idx, h_i), root_dir=root_img_dir)


        prediction = torch.argmax(logits, dim=-1);
        correct = (prediction == label.argmax(dim=-1));

        accs.extend(correct.detach().cpu().numpy());
        ces.append(loss.detach().cpu().numpy());

    accs = 100 * np.mean(accs);
    return accs, np.mean(ces),


if __name__ == '__main__':
    import os
    from app.os_model.data_loader import json_to_nps, load_data
    import torch.optim as optim
    from app.os_model.exp.Oneshot import Oneshot
    import pandas as pd

    os_exp = Oneshot(feature_dim=5, stim_ratio=[16, 8, 1], stim_vectors=
    [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
     np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
     np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])],
                     cue_ratio=[4, 1], cue_vectors=[np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                                                    np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]);


    data_root = "../../../data_0901";
    meta_root = "../../../meta_0901";
    meta_dir = os.listdir(meta_root);

    sub_ids = os.listdir(data_root);

    df_subs = pd.read_csv('../data_0901.csv')

    for sub_id in sub_ids :

        sub_id = sub_id.split(".json")[0]

        net = Transformer(in_channel=5, d_embed=4, d_model=4,
                          d_inner=4, d_k=3 , d_v=3, e_layers=2, d_layers=3,
                          n_head=2, dropout=0.0, n_position=30, class_num=2);


        for m in meta_dir:

            with open(os.path.join(meta_root, m), "r") as f:

                lines = f.readlines();
                if lines:
                    sub_id_meta = lines[3].split("subId\t")[1];
                    if sub_id == sub_id_meta[:-1]:
                        m_name = m
                        break;

        inf_nps = json_to_nps(os.path.join(data_root, "{}".format(sub_id + ".json")))

        train_loader, valid_loader, incre_loader, os_loader, \
        c_incre_loader, c_os_loader, (exp_xs, exp_ys), (cand_incre_xs, cand_incre_ys), (cand_os_xs, cand_os_ys) = load_data(inf_nps, os_exp,
                                                                                                                            batch_size=4)

        net = net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.98),
                               eps = 1.0e-9);

        for epoch in range(5000):

            acc_train, ce_train = forward(train_loader, eval=False)

            if epoch % 20 == 0:
                acc_val, ce_val = forward(valid_loader, eval=True)

                print("E{} || Train ACCs : {:3f}, Train CEs : {:3f}, "
                      " Val ACCs : {:3f},  Val CEs : {:3f}".format(epoch, acc_train, ce_train,
                                                                   acc_val, ce_val))

                acc_val_incre, ce_val_incre = forward(incre_loader, eval=True, save_img=True,
                                                      root_img_dir="./fig_attn/incre");
                acc_val_os, ce_val_os = forward(os_loader, eval=True, save_img=True,
                                                root_img_dir="./fig_attn/os");


            if epoch % 100 == 0 :
                with torch.no_grad() :

                    if epoch == 4000 :
                        verbose=True
                    else:
                        verbose = False
                    exp_xs = exp_xs.cuda();

                    exp_feature = net.forward_feature(exp_xs);

                    attention = net.get_attention(exp_xs);

                    perturbed_exps_0 = os_exp.perturb_episode(exp_xs[:], attention[:], num=100, perturb_type=0, verbose=verbose)
                    perturbed_exps_1 = os_exp.perturb_episode(exp_xs[:], attention[:], num=100, perturb_type=1, verbose=verbose)

                    D, N = perturbed_exps_0.shape[:2];

                    perturbed_exps_0 = perturbed_exps_0.reshape(D*N, -1);
                    perturbed_exps_1 = perturbed_exps_1.reshape(D*N, -1);

                    perturbed_feature_0 = net.forward_feature(torch.LongTensor(perturbed_exps_0).cuda());
                    perturbed_feature_1 = net.forward_feature(torch.LongTensor(perturbed_exps_1).cuda());

                    perturbed_feature_0 = perturbed_feature_0.reshape(D, N, -1);
                    perturbed_feature_1 = perturbed_feature_1.reshape(D, N, -1);

                    inc_feature = net.forward_feature(cand_incre_xs.cuda());
                    os_feature = net.forward_feature(cand_os_xs.cuda());

                    metric1 = perturbed_feature_0.std()/exp_feature.std()
                    metric2 = perturbed_feature_1.std()/exp_feature.std();
                    metric3 = perturbed_feature_1.std()/perturbed_feature_0.std()

                    metric4 = F.mse_loss(perturbed_feature_0, exp_feature.unsqueeze(1))
                    metric5 = F.mse_loss(perturbed_feature_1, exp_feature.unsqueeze(1))
                    metric6 = metric4/metric5

                    '''
                    tot_features = torch.cat([exp_feature, perturbed_feature_0, perturbed_feature_1], dim=0)

                    rf_type = np.concatenate([np.stack([1] * len(exp_feature)),
                                              np.stack([0] * (len(perturbed_feature_0) + len(perturbed_feature_1)))]);

                    os_type = np.concatenate([(exp_ys.argmax(-1) == 2).long().data.numpy(),
                                              np.stack([2] * len(perturbed_exps_0)),
                                              np.stack([2] * len(perturbed_exps_1))]);

                    perturb_type = np.concatenate([np.stack([0] * len(exp_feature)),
                                              np.stack([1] * len(perturbed_feature_0)),
                                              np.stack([2] * len(perturbed_feature_1))]);

                    '''
                    '''
                    pca = PCA(n_components=2)
                    low_x = pca.fit_transform(tot_features.cpu().data.numpy());

                    vis_results_feature(low_x, rf_types=rf_type,
                                               os_types=os_type,
                                                perturb_types = perturb_type,
                                                fname="im_{}.png".format(epoch),
                                                root_dir = "./fig_attn/feature")
                    '''


        df_subs.loc[df_subs.Sub_id == sub_id, "metric1"] = metric1.item()
        df_subs.loc[df_subs.Sub_id == sub_id, "metric2"] = metric2.item()
        df_subs.loc[df_subs.Sub_id == sub_id, "metric3"] = metric3.item()
        df_subs.loc[df_subs.Sub_id == sub_id, "metric4"] = metric4.item()
        df_subs.loc[df_subs.Sub_id == sub_id, "metric5"] = metric5.item()
        df_subs.loc[df_subs.Sub_id == sub_id, "metric6"] = metric6.item()

    df_subs.to_csv("data_0901_me.csv")





