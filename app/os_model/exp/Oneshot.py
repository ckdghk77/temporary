import numpy as np
import torch
import random
from app.os_model.visualizer import vis_results_shuffle
from app.os_model.data_loader import CustomDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

class Oneshot(object) :
    def __init__(self, feature_dim=5, stim_ratio=[16,8,1], stim_vectors=[0.5, 0.25, 0.1],
                 cue_ratio=[4,1], cue_vectors=[0.6, 0.3], lr=0.1, primacy=0.36, recency=0.36, model="sangwan2014") :

        self.tot_stim=25
        self.stim_ratio = stim_ratio
        self.stim_vectors = stim_vectors

        self.tot_cue=5
        self.cue_ratio = cue_ratio
        self.cue_vectors = cue_vectors

        if model == "naive" :
            from app.os_model.exp.Oneshot_Naive import os_naive
            self.model = os_naive();
        elif model == "sangwan2014" :
            from app.os_model.exp.Oneshot_sangwan import os_sangwan
            self.model = os_sangwan(lr=lr, primacy=primacy, recency = recency);
        elif model == "random" :
            from app.os_model.exp.Oneshot_Random import os_random
            self.model = os_random();


        self.tot_dim=feature_dim;


    def spawn_stims(self, idx=None) :

        candidates = np.arange(self.tot_stim);
        np.random.shuffle(candidates)

        stims = np.zeros(shape=(self.tot_stim, self.tot_dim), dtype=np.float32);

        for sr, sr_value in zip(self.stim_ratio, self.stim_vectors) :
            idxes = np.random.choice(candidates, size = sr, replace=False);
            stims[idxes] = sr_value;

            for id in idxes :
                cand_id = np.where(candidates == id);
                candidates = np.delete(candidates, cand_id)

        stims = np.reshape(stims, (5,5, self.tot_dim));

        return stims

    def spawn_cues(self, idx=None):

        cues = np.zeros(shape=(self.tot_cue, self.tot_dim), dtype=np.float32);

        for c_idx, c in enumerate(cues) :

            if c_idx == idx :
                cues[c_idx] = self.cue_vectors[-1];
            else :
                cues[c_idx] = self.cue_vectors[0]

        return cues

    def gen_episode(self, net=None, target=None, opt_model=None, type="fake", device="cuda"):

        if target is None :
            target = np.random.choice([0,1], size = 1, replace=False)

        if net is None :
            stims = self.spawn_stims();

            novel_stim_r = np.where(np.argmax(stims, -1) == 2)[0][0];

            if target == 1 :
                novel_rew_row = novel_stim_r
            else :
                candidates = np.arange(5);
                candidates = np.delete(candidates, novel_stim_r)
                novel_rew_row = np.random.choice(candidates, size=1, replace=False)

            cues = self.spawn_cues(novel_rew_row)
            ep = np.concatenate([stims, np.expand_dims(cues, 1)], axis=1);

            labels = self.model.perform(ep)

            return ep, labels, (None, None)
        else :

            net.eval()
            eps = []
            ep_ys = []
            task_vecs = []

            for c in range(100) :
                task_vector = np.zeros(shape=(5, 6, 2), dtype=np.float32)
                task_vector[:, :5, 0] = 1.0
                task_vector[:, 5:, 1] = 1.0

                task_y = np.zeros(shape=(2), dtype=np.float32)
                task_y[target] = 1.0;
                stims = self.spawn_stims();
                # A (4, 0, 0, ), B (1,2,0,4,3)

                novel_stim_r = np.where(np.argmax(stims, -1) == 2)[0][0];

                if target == 1:
                    novel_rew_row = novel_stim_r
                else:
                    candidates = np.arange(5);
                    candidates = np.delete(candidates, novel_stim_r)
                    novel_rew_row = np.random.choice(candidates, size=1, replace=False)

                cues = self.spawn_cues(novel_rew_row)
                ep = np.concatenate([stims, np.expand_dims(cues, 1)], axis=1);

                #print(np.where(ep.argmax(-1)==2))
                eps.append(ep)
                ep_ys.append(task_y)
                task_vecs.append(task_vector)
            #print("******************")
            np_eps = np.stack(eps);
            np_ep_ys = np.stack(ep_ys);
            task_vecs = np.stack(task_vecs);

            np_oh_eps = np.stack([np.reshape(xs, (30,5)) for xs in np_eps]);
            np_eps = np.stack([np.argmax(xs, -1).reshape(30) for xs in np_eps]);

            task_vecs = np.stack([xs.reshape(30,-1) for xs in task_vecs])

            dataset = CustomDataSet(data=(torch.LongTensor(np_eps),
                                torch.FloatTensor(np_oh_eps),
                                torch.FloatTensor(np_ep_ys),
                                torch.LongTensor(task_vecs)),
                                    seed=None)

            test_loader = DataLoader(dataset, batch_size=len(ep_ys), shuffle=False);

            data, data_oh, idx_os, label, task_vec = next(iter(test_loader));

            data, data_oh, label, task_vec = data.to(device), data_oh.to(device), label.to(device), task_vec.to(device)

            logits = net(data, data_oh, idx_os, task_vec);
            #probs = F.softmax(logits,-1);

            pred = logits[:,1];
            opt_results = [opt_model.model.perform(ep)[-1,:] for ep in eps];
            opt_results = np.stack(opt_results)
            if target == 1 :
                opt_results = opt_results[:,-1]
            else :
                opt_results = opt_results[:,:2].sum(1)
            td_opt = np.abs(pred.cpu().data.numpy() - opt_results);

            #rank = (rank - rank.min()) / (rank.max() - rank.min());
            rank = td_opt/td_opt.sum()

            #max_idx = np.random.choice(np.arange(len(rank)),
            #                           p=rank.cpu().data.numpy())
            #max_idx = torch.argmax(rank).item()
            max_idx = np.argmax(rank)

            if target==1 :
                print(np.where(eps[max_idx].argmax(-1)==2))
            #print(rank)

            eps_onehot = eps[max_idx].argmax(-1);
            eps_onehot = eps_onehot.reshape(-1);

            cue_allocs = np.concatenate([eps_onehot[0:5],eps_onehot[6:6+5],eps_onehot[12:12+5]
                               ,eps_onehot[18:18+5],eps_onehot[24:24+5]])
            rew_allocs = np.concatenate([eps_onehot[5:6],eps_onehot[11:12], eps_onehot[17:18],
                                         eps_onehot[23:24],eps_onehot[29:30]])

            return eps[max_idx], self.model.perform(eps[max_idx]), (cue_allocs, rew_allocs)


    def shuffle_stims_rews(self, orig_exp, unfixed_idxes, num=100):

        rew_idxes = set([5,11,17,23,29]);
        stim_idxes = set(np.arange(30)) - rew_idxes;

        orig_stim_idxes = list(stim_idxes.intersection(unfixed_idxes));
        orig_rew_idxes = list(rew_idxes.intersection(unfixed_idxes));

        shuffled_exps = [];

        for i in range(num) :

            shuffled_idxes = orig_stim_idxes.copy();
            random.shuffle(shuffled_idxes)

            shuffled_exp = orig_exp.copy();

            for oi, si in zip(orig_stim_idxes, shuffled_idxes) :
                shuffled_exp[oi] = orig_exp[si];

            shuffled_idxes = orig_rew_idxes.copy();
            random.shuffle(shuffled_idxes)

            for oi, si in zip(orig_rew_idxes, shuffled_idxes):
                shuffled_exp[oi] = orig_exp[si];

            shuffled_exps.append(shuffled_exp)

        return np.stack(shuffled_exps)

    def shuffle_rews(self, orig_exp, unfixed_idxes):

        stim_idxes = set([5,11,17,23,29]);

        orig_idxes = list(stim_idxes.intersection(unfixed_idxes));
        shuffled_idxes = orig_idxes.copy();
        random.shuffle(shuffled_idxes)

        shuffled_exp = orig_exp.copy();

        for oi, si in zip(orig_idxes, shuffled_idxes) :
            shuffled_exp[oi] = orig_exp[si];

        return shuffled_exp


    def perturb_input(self, tasks, num=100, verbose=False, perturb_type=0):

        if type(tasks) == torch.Tensor :
            tasks = tasks.cpu().data.numpy();

        s_task_all = np.zeros(shape=(len(tasks), num, tasks.shape[-1]), dtype=np.long)

        pass


    def perturb_episode(self, tasks, attentions, num=100, verbose=False, perturb_type=0):

        if type(tasks) == torch.Tensor :
            tasks = tasks.cpu().data.numpy();

        if type(attentions) == torch.Tensor:
            attentions = attentions.cpu().data.numpy();

        s_task_all = np.zeros(shape=(len(tasks),num, tasks.shape[-1]), dtype=np.long)

        for t_i, (task, attention) in enumerate(zip(tasks, attentions)) :

            fixed_idxes = []
            for att in attention :
                tar = att.argmax();

                fixed_idxes.append(tar)

            tot_idxes = set(np.arange(len(task)))
            fixed_idxes = set(fixed_idxes);
            unfixed_idxes = tot_idxes - fixed_idxes

            s_tasks = self.shuffle_stims_rews(task, num=num,
                                        unfixed_idxes=unfixed_idxes if perturb_type == 0 else fixed_idxes)

            s_task_all[t_i] = s_tasks
            fixed_loc = np.zeros_like(task, dtype=np.float32);
            for f in fixed_idxes :
                fixed_loc[f] = 1.0

            if verbose :
                vis_results_shuffle(np.expand_dims(task,0).repeat(len(s_tasks),axis=0).reshape(len(s_tasks), 5,6),
                                    s_tasks.reshape(len(s_tasks),5,6),
                                    fixed_idxes=fixed_loc.reshape(1,5,6).repeat(len(s_tasks),axis=0),
                                    fname="im_{}.png".format(perturb_type),
                                    root_dir="./fig_attn/shuffle",
                                    nrows=5, ncols=1)

        return s_task_all

    '''
    
    
    def gen_episode(self, target=None, type="fake"):

        if target is None :
            target = np.random.choice([0,1], size = 1, replace=False)

        stims = self.spawn_stims();
        candidates = np.arange(5);
        novel_row = np.random.choice(candidates, size=1, replace=False)
        self.spawn_cues(novel_row)


        os_row = np.where(np.argmax(stims,axis=-1) == 2)[0]

        if type == "fake" :

            if target == 1 :
                cues = self.spawn_cues(os_row);
            else :
                candidates = np.arange(5);
                candidates = np.delete(candidates, os_row);
                inc_row = np.random.choice(candidates, size=1, replace=False)

                cues = self.spawn_cues(inc_row);

        ep = np.concatenate([stims, np.expand_dims(cues, 1)], axis=1);

        return ep, target
        
    '''





