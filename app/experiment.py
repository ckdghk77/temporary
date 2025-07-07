from itertools import count
from flask import (Blueprint, redirect, render_template, request, session, url_for, jsonify)
from .io import write_data, write_metadata
import json
import time, os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from app.os_model.data_loader import json_to_nps, load_os_data
from app.os_model.module_RNN.dne_driving_model import iterate_epoch
from app.os_model.module_RNN.sublayers import CosineAnnealingWarmupRestarts
from app.os_model.module_RNN.dnns import LSTM_Attn2

## Initialize blueprint.
bp = Blueprint('experiment', __name__)
total_epoch = 20
available_gpus = ['cuda:0','cuda:1','cuda:2','cuda:3','cuda:4','cuda:5','cuda:6','cuda:7']
#available_gpus = ['cuda:0']

@bp.route('/experiment')
def experiment():

    """Present jsPsych experiment to participant."""

    ## Error-catching: screen for missing session.
    if not 'workerId' in session:

        ## Redirect participant to error (missing workerId).
        return redirect(url_for('error.error', errornum=1000))

    ## Case 1: previously completed experiment.
    elif 'complete' in session:
        
        ## Redirect participant to complete page.
        return redirect(url_for('complete.complete'))

    ## Case 2: repeat visit.
    elif not session['allow_restart'] and 'experiment' in session:

        ## Update participant metadata.
        session['ERROR'] = "1004: Revisited experiment."
        session['complete'] = 'error'
        write_metadata(session, ['ERROR','complete'], 'a')

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1004))

    ## Case 3: first visit.
    else:
            
        ## Update participant metadata.
        session['experiment'] = True
        
        incomp_dir = session['incomplete'].replace('\\\\','\\')
        incomp_file = os.path.join(incomp_dir, session['subId']+".json");
        model_file = os.path.join(incomp_dir, session['subId'], "_model_{}.pt");
        
        try :
            os.makedirs(os.path.join(incomp_dir, session['subId']))
            print("Successfully created subjects model dir")
        except :
            pass        

        session['incomp_file'] = incomp_file
        session['model_file'] = model_file
        session['s_count'] = 0
        session['gpu_n'] = np.random.choice(available_gpus);

        print(session['gpu_n'])
        net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
        lr = 0.01
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                     cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                     warmup_steps=150, gamma=0.5)
        net.save_model(fname=session['model_file'].format(session['s_count']), 
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)
        
        write_metadata(session, ['experiment'], 'a')
        del net
        del optimizer
        del lr_scheduler
        try :
            print("WorkerId : " + session['workerId'])
            print("AssignmentId : " + session['assingmentId'])
        except :
            pass
        ## Present experiment.
        return render_template('experiment.html', workerId=session['workerId'], assignmentId=session['assignmentId'], hitId=session['hitId'], a=session['a'], tp_a=session['tp_a'], b=session['b'], tp_b=session['tp_b'], c=session['c'], tp_c=session['tp_c'])

@bp.route('/experiment', methods=['POST'])
def pass_message():
    """Write jsPsych message to metadata."""
    
    if request.is_json:

        ## Retrieve jsPsych data.
        msg = request.get_json()

        ## Update participant metadata.
        session['MESSAGE'] = msg
        write_metadata(session, ['MESSAGE'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/incomplete_save', methods=['POST'])
def incomplete_save():
    """Save incomplete jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='incomplete')

    ## Flag partial data saving.
    #session['MESSAGE'] = 'incomplete dataset saved'
    #write_metadata(session, ['MESSAGE'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/redirect_success', methods = ['POST'])
def redirect_success():
    """Save complete jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='pass')

    ## Flag experiment as complete.
    session['complete'] = 'success'
    write_metadata(session, ['complete'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/redirect_reject', methods = ['POST'])
def redirect_reject():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'reject'
    write_metadata(session, ['complete'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/redirect_error', methods = ['POST'])
def redirect_error():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'error'
    write_metadata(session, ['complete'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

#manager = Manager()
#cur_epoch = manager.Value("d", 0) 

## model function can be defined in experiment.py or other python file
def model_train(dloader, net, optimizer, lr_scheduler, device="cuda"):
    #print("Model_train_In! : {}".format(session["cur_epoch"]))
    
    time.sleep(1)
    for e in range (50) :
        acc, loss = iterate_epoch(data_loader=dloader, 
                      network=net, optim=optimizer, eval=False, device=device)
        lr_scheduler.step()
        #print(lr_scheduler.get_lr())
    
    print("E{} || Train ACCs : {:3f}, Train CEs : {:3f}".format(session["cur_epoch"], 
                                                                acc, loss))

    session["cur_epoch"] = session["cur_epoch"] + 1
    #print("Model_train_Out : {}".format(session["cur_epoch"]))
   
    return

## function to data from browser and train model one epoch then return model progress
## At end of training, return model result
@bp.route('/train_iter', methods = ['GET'])
def train_iter():
    
    while True :
        try :
            data_np = json_to_nps(os.path.join(session['incomplete'], session['subId'],
                             '{}_{}.json'.format(session['subId'],session['s_count'])))
        except :
            #print("Wating for incomplete file")
            continue;
        break
   
    sessions = list(data_np.keys())
    train_loader, _, _, _, _ = load_os_data([data_np],
                total_session=np.arange(len(sessions)),
                train_session=np.arange(len(sessions)),
                val_session=np.arange(len(sessions)),
                batch_size=8)
    
    
    net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
    lr = 0.01
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                            cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                            warmup_steps=150, gamma=0.5)
    
    net.load_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
    net = net.to(session['gpu_n'])
    lr_scheduler.set_optimizer(optimizer)
    if not "cur_epoch" in session:
        session["cur_epoch"] = 0
    
    model_train(train_loader, net, optimizer, lr_scheduler, device=session['gpu_n'])

    net.save_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
       
    if session["cur_epoch"] >= total_epoch : 
        session["cur_epoch"] = 0
        session['s_count']+=1
        net.save_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
        del net
        del optimizer
        return "1"
    else :
        del net 
        del optimizer
        return str(session["cur_epoch"]/total_epoch)
    
@bp.route('/dummy_iter', methods = ['GET'])
def dummy_iter():
    if not "cur_epoch" in session:
        session["cur_epoch"] = 0
    #print("Dummy train_In! : {}".format(session["cur_epoch"]))
    
    time.sleep(3)
    
    session["cur_epoch"] = session["cur_epoch"] + 1

    #print("Dummy train_Out : {}".format(session["cur_epoch"]))
    
    if session["cur_epoch"] >= total_epoch : 
        info = {"progress" : "1"}
        return json.dumps(info)
    else :
        info = {"progress" : str(session["cur_epoch"]/total_epoch)}
        return json.dumps(info)


@bp.route('/gen_exps', methods = ['GET'])
def gen_exps():
    #print("GEN EPISODES!!!!")
    from app.os_model.exp.Oneshot import Oneshot

    os_type = int(request.args.get('ostype'))

    feature_dim=5;
    stim_ratio=[16,8,1];
    stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                                       np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
    cue_ratio = [4,1]
    cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]

    os_exp = Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.36, recency=0.36, 
                model="sangwan2014")
    
    net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
    lr = 0.01
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                 cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                 warmup_steps=150, gamma=0.5)

    net.load_model(fname=session['model_file'].format(session['s_count']),
                   optimizer = optimizer, lr_scheduler = lr_scheduler)
    net = net.to(session['gpu_n'])
    with torch.no_grad() :
        x, _, (cue_alloc, rew_alloc) = os_exp.gen_episode(net, target=os_type, opt_model=os_exp, device=session['gpu_n'])

    info = {"cue_alloc" : cue_alloc.tolist(),
     "rew_alloc" : rew_alloc.tolist()}
    del net
    
    return json.dumps(info)
