
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.os_model.module_RNN.transformer import Transformer

class MLP(nn.Module) :
    def __init__(self, in_channel=1, n_position=30, dropout=0.0, class_num=2):
        super(MLP, self).__init__()

        self.in_net = nn.Linear(in_channel,1)


        self.encoder = nn.Sequential(
            nn.Linear(n_position, n_position//2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(n_position//2, n_position//4),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.out_net = nn.Linear(n_position//4,class_num)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        in_x = self.in_net(src_seq_oh).squeeze(-1);
        enc_x = self.encoder(in_x);
        out = self.out_net(enc_x);

        return out


class SLP(nn.Module) :
    def __init__(self, in_channel=1, n_position=30, dropout=0.0, class_num=2):
        super(SLP, self).__init__()

        self.in_net = nn.Linear(in_channel,1)

        self.encoder = nn.Sequential(
            nn.Linear(n_position, n_position//4),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.out_net = nn.Linear(n_position//4,class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        in_x = self.in_net(src_seq_oh).squeeze(-1);
        enc_x = self.encoder(in_x);
        out = self.out_net(enc_x);

        return out

class LSTM(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0, class_num=2):
        super(LSTM, self).__init__()

        self.encoder = nn.LSTM(input_size=in_channel, hidden_size=6,
                               num_layers=2, batch_first=True, dropout=dropout)

        self.out_net = nn.Linear(6, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        out, (hidden, cell) = self.encoder(src_seq_oh);

        out = torch.stack([out[o_idx,t] for o_idx, t in enumerate(tar_idx.squeeze())])
        out = self.out_net(out);

        return out


class BiLSTM(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0,hidden_size=6, class_num=2):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=in_channel, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.out_net = nn.Linear(hidden_size*2, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        B, S, _ = src_seq_oh.shape;

        h = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)

        out, (hidden, cell) = self.encoder(src_seq_oh, (h,c));

        out = torch.stack([out[o_idx,t] for o_idx, t in enumerate(tar_idx.squeeze())])
        out = self.out_net(out);

        return out


class BiLSTM_Attn(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0,hidden_size=6, class_num=2, device="cuda"):
        super(BiLSTM_Attn, self).__init__()

        self.hidden_size = hidden_size

        self.tr_net = Transformer(in_channel=in_channel, d_embed=in_channel, d_inner=6,
                                  d_k=4, d_v=4, e_layers=1, d_layers=1,
                          n_head=4, dropout=0.0, n_position=30, class_num=2, device=device);

        self.encoder = nn.LSTM(input_size=15, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.out_net = nn.Linear(hidden_size*2, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        tr_enc = self.tr_net.forward_feature(src_seq_oh, task_vec);

        tr_enc = self.tr_net.encoder.position_enc(tr_enc)
        B, S, _ = src_seq_oh.shape;

        h = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)

        lstm_out, (hidden, cell) = self.encoder(tr_enc, (h,c));

        out = torch.stack([lstm_out[o_idx,t] for o_idx, t in enumerate(tar_idx.squeeze())])
        out = self.out_net(out)

        return out


class LSTM_Attn(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0,hidden_size=6, class_num=2, device="cuda"):
        super(LSTM_Attn, self).__init__()

        self.hidden_size = hidden_size

        self.tr_net = Transformer(in_channel=in_channel, d_embed=in_channel, d_inner=6,
                                  d_k=4, d_v=4, e_layers=2, d_layers=1,
                          n_head=4, dropout=0.0, n_position=30, class_num=2, device=device);

        self.encoder = nn.LSTM(input_size=15, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, bidirectional=False, dropout=dropout)

        self.out_net = nn.Linear(hidden_size, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        tr_enc = self.tr_net.forward_feature(src_seq_oh, task_vec);

        tr_enc = self.tr_net.encoder.position_enc(tr_enc)
        B, S, _ = src_seq_oh.shape;

        h = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)

        lstm_out, (hidden, cell) = self.encoder(tr_enc, (h,c));

        out = torch.stack([lstm_out[o_idx,t] for o_idx, t in enumerate(tar_idx.squeeze())])
        out = self.out_net(out)

        return out



class BiLSTM_Attn2(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0,hidden_size=6, class_num=2, device="cuda"):
        super(BiLSTM_Attn2, self).__init__()

        self.hidden_size = hidden_size


        self.encoder = nn.LSTM(input_size=in_channel+4, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.tr_net = Transformer(in_channel=hidden_size*2, d_embed=hidden_size*2, d_inner=6,
                                  d_k=4, d_v=4, e_layers=1, d_layers=1,
                          n_head=4, dropout=0., n_position=30, class_num=2, device=device);


        self.out_net = nn.Linear(hidden_size*2, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):
        B, S, _ = src_seq_oh.shape;

        src_seq_oh = self.tr_net.encoder.position_enc(src_seq_oh)

        h = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(4, B, self.hidden_size)).to(src_seq_oh.device)

        lstm_out, (hidden, cell) = self.encoder(src_seq_oh, (h,c));

        out = self.tr_net(src_seq, lstm_out, tar_idx, task_vec)

        return out



class LSTM_Attn2(nn.Module) :
    def __init__(self, in_channel=1, dropout=0.0,hidden_size=6, class_num=2, device="cuda"):
        super(LSTM_Attn2, self).__init__()

        self.hidden_size = hidden_size

        self.device = device
        self.encoder = nn.LSTM(input_size=in_channel+4, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, bidirectional=False, dropout=dropout)

        self.tr_net = Transformer(in_channel=hidden_size, d_embed=hidden_size, d_inner=6,
                                  d_k=4, d_v=4, e_layers=1, d_layers=1,
                                n_head=4, dropout=0., n_position=30, class_num=2, device=device);


        self.out_net = nn.Linear(hidden_size, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def save_model(self, fname, optimizer=None, lr_scheduler=None):
        torch.save({'model' : self.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict()}, fname);

    def load_model(self, fname, optimizer=None, lr_scheduler=None):

        chkpt = torch.load(fname,map_location=torch.device("cpu"));
        self.load_state_dict(chkpt['model']);
        optimizer.load_state_dict(chkpt['optimizer'])
        lr_scheduler.load_state_dict(chkpt['lr_scheduler'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def forward_attn(self, src_seq, src_seq_oh, tar_idx, task_vec):
        B, S, _ = src_seq_oh.shape;

        src_seq_oh = self.tr_net.encoder.position_enc(src_seq_oh)

        h = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)

        lstm_out, (hidden, cell) = self.encoder(src_seq_oh, (h, c));

        attention = self.tr_net.get_attention(src_seq, lstm_out, tar_idx, task_vec)

        return attention

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):
        B, S, _ = src_seq_oh.shape;

        src_seq_oh = self.tr_net.encoder.position_enc(src_seq_oh)

        h = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)
        c = torch.zeros(size=(2, B, self.hidden_size)).to(src_seq_oh.device)

        lstm_out, (hidden, cell) = self.encoder(src_seq_oh, (h,c));

        out = self.tr_net(src_seq, lstm_out, tar_idx, task_vec)

        return out

class CNN(nn.Module) :
    def __init__(self, in_channel=1, n_position=30, dropout=0.0, class_num=2):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=12, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(in_channels=12, out_channels=n_position//4, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.out_net = nn.Linear(n_position//4,class_num)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None :
                    m.bias.data.fill_(0.0)

    def forward(self, src_seq, src_seq_oh, tar_idx, task_vec):

        enc_x = self.encoder(src_seq_oh.permute(0,2,1));
        out = self.out_net(enc_x.squeeze(-1));

        return out