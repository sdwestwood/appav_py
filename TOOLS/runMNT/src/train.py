import torch
from torch import nn
import sys
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import csv   

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from src.eval_metrics import *
torch.autograd.set_detect_anomaly(True)

####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    p2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_p, out_seq_len=hyp_params.l_len)
    r2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_r, out_seq_len=hyp_params.l_len)
    return p2l_module, r2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr, weight_decay=1e-5)
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_p2l_module, ctc_r2l_module = None, None
        ctc_p2l_optimizer, ctc_r2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_p2_module, ctc_r2_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_p2_module, ctc_r2_module = ctc_p2_module.cuda(), ctc_r2_module.cuda()
        ctc_p2_optimizer = getattr(optim, hyp_params.optim)(ctc_p2_module.parameters(), lr=hyp_params.lr)
        ctc_r2_optimizer = getattr(optim, hyp_params.optim)(ctc_r2_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_p2_module': ctc_p2_module,
                'ctc_r2_module': ctc_r2_module,
                'ctc_p2_optimizer': ctc_p2_optimizer,
                'ctc_r2_optimizer': ctc_r2_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_p2_module = settings['ctc_p2_module']
    ctc_r2_module = settings['ctc_r2_module']
    ctc_p2_optimizer = settings['ctc_p2_optimizer']
    ctc_r2_optimizer = settings['ctc_r2_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']
    

    def train(model, optimizer, criterion, ctc_p2_module, ctc_r2_module, ctc_p2_optimizer, ctc_r2_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            if hyp_params.modality_count == 2:
                sample_ind, audio, features = batch_X
            elif hyp_params.modality_count == 3:
                sample_ind, text, audio, features = batch_X
            elif hyp_params.modality_count == 4:
                sample_ind, text, audio, features, head = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if ctc_criterion is not None:
                ctc_p2_module.zero_grad()
                ctc_r2_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    if hyp_params.modality_count == 4:
                        text, audio, features, head, eval_attr = text.cuda(), audio.cuda(), features.cuda(), head.cuda(), eval_attr.cuda()
                    elif hyp_params.modality_count == 3:
                        text, audio, features, eval_attr = text.cuda(), audio.cuda(), features.cuda(), eval_attr.cuda()
                    elif hyp_params.modality_count == 2:
                        audio, features, eval_attr = audio.cuda(), features.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap' or hyp_params.label_type=='binary':
                        eval_attr = eval_attr.long()
            
            batch_size = audio.size(0)
            batch_chunk = hyp_params.batch_chunk
            
            ######## CTC STARTS ######## Do not worry about this if not working on CTC
            if ctc_criterion is not None:
                ctc_p2_net = nn.DataParallel(ctc_p2_module) if batch_size > 10 else ctc_p2_module
                ctc_r2_net = nn.DataParallel(ctc_r2_module) if batch_size > 10 else ctc_r2_module

                audio, p2_position = ctc_p2_net(audio) # audio now is the aligned to text
                features, r2_position = ctc_r2_net(features)
                
                ## Compute the ctc loss
                l_len, p_len, r_len, h_len = hyp_params.l_len, hyp_params.p_len, hyp_params.r_len, hyp_params.h_len
                # Output Labels
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                # Specifying each output length
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                # Specifying each input length
                p_length = torch.tensor([p_len]*batch_size).int().cpu()
                r_length = torch.tensor([r_len]*batch_size).int().cpu()
                
                ctc_p2_loss = ctc_criterion(p2_position.transpose(0,1).cpu(), l_position, p_length, l_length)
                ctc_r2_loss = ctc_criterion(r2_position.transpose(0,1).cpu(), l_position, r_length, l_length)
                ctc_loss = ctc_p2_loss + ctc_r2_loss
                ctc_loss = ctc_loss.cuda() if hyp_params.use_cuda else ctc_loss
            else:
                ctc_loss = 0
            ######## CTC ENDS ########
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                if hyp_params.modality_count > 2:
                    text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                features_chunks = features.chunk(batch_chunk, dim=0)
                if hyp_params.modality_count == 4:
                    head_chunks = head.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    if hyp_params.modality_count == 4:
                        text_i, audio_i, features_i, head_i = text_chunks[i], audio_chunks[i], features_chunks[i], head_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        preds_i, hiddens_i = net(text_i, audio_i, features_i, head_i)
                    elif hyp_params.modality_count == 3:
                        text_i, audio_i, features_i = text_chunks[i], audio_chunks[i], features_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        preds_i, hiddens_i = net(text_i, audio_i, features_i)
                    elif hyp_params.modality_count == 2:
                        audio_i, features_i = audio_chunks[i], features_chunks[i]
                        eval_attr_i = eval_attr_chunks[i]
                        preds_i, hiddens_i = net(audio_i, features_i)
                    
                    if hyp_params.dataset == 'iemocap' or hyp_params.label_type=='binary':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    elif 'deap' in hyp_params.dataset and hyp_params.label_type!='binary':
                        eval_attr_i = eval_attr_i.view(-1)
                        
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                    
                ctc_loss.backward()
                combined_loss = raw_loss + ctc_loss
                
            else:
                if hyp_params.modality_count == 4:
                    preds, hiddens = net(text, audio, features, head)
                elif hyp_params.modality_count == 3:
                    preds, hiddens = net(text, audio, features)
                elif hyp_params.modality_count == 2:
                    preds, hiddens = net(audio, features)
                if hyp_params.dataset == 'iemocap' or hyp_params.label_type=='binary':
                    #preds = preds.view(-1, 2)
                    preds = preds.view(-1)
                    eval_attr = eval_attr.view(-1).float()
                elif 'deap' in hyp_params.dataset  and hyp_params.label_type!='binary':
                    preds = preds.view(-1)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + ctc_loss
                combined_loss.backward()
            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_p2_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_r2_module.parameters(), hyp_params.clip)
                ctc_p2_optimizer.step()
                ctc_r2_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_p2_module, ctc_r2_module, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                if hyp_params.modality_count == 4:
                    sample_ind, text, audio, features, head = batch_X
                elif hyp_params.modality_count == 3:
                    sample_ind, text, audio, features = batch_X
                elif hyp_params.modality_count == 2:
                    sample_ind, audio, features = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        if hyp_params.modality_count == 4:
                            text, audio, features, head, eval_attr = text.cuda(), audio.cuda(), features.cuda(), head.cuda(), eval_attr.cuda()
                        elif hyp_params.modality_count == 3:
                            text, audio, features, eval_attr = text.cuda(), audio.cuda(), features.cuda(), eval_attr.cuda()
                        elif hyp_params.modality_count == 2:
                            audio, features, eval_attr = audio.cuda(), features.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap' or hyp_params.label_type=='binary':
                            eval_attr = eval_attr.long()
                        
                batch_size = audio.size(0)
                
                if (ctc_p2_module is not None) and (ctc_r2_module is not None):
                    ctc_p2_net = nn.DataParallel(ctc_p2_module) if batch_size > 10 else ctc_p2_module
                    ctc_r2_net = nn.DataParallel(ctc_r2_module) if batch_size > 10 else ctc_r2_module
                    audio, _ = ctc_p2_net(audio)     # audio aligned to text
                    features, _ = ctc_r2_net(features)   # features aligned to text
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                if hyp_params.modality_count == 4:
                    preds, _ = net(text, audio, features, head)
                elif hyp_params.modality_count == 3:
                    preds, _ = net(text, audio, features)
                elif hyp_params.modality_count == 2:
                    preds, _ = net(audio, features)
                if hyp_params.dataset == 'iemocap' or hyp_params.label_type=='binary':
                    #preds = preds.view(-1, 2)
                    preds = preds.view(-1)
                    eval_attr = eval_attr.view(-1).float()
                elif 'deap' in hyp_params.dataset and hyp_params.label_type!='binary':
                    preds = preds.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size
                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        class_accuracy = None
        if hyp_params.label_type == 'binary': 
                #arg_preds = torch.argmax(results, dim=-1)
                arg_preds = torch.round(nn.Sigmoid()(results))
                if test:
                    cond = 'test'
                    unique, counts = torch.unique(arg_preds, return_counts=True)
                    print('predicted', dict(zip(unique, counts)))
                    unique, counts = torch.unique(truths, return_counts=True)
                    print('truths', dict(zip(unique, counts)))
                else:
                    cond = 'validation'
                class_accuracy = accuracy_score(truths.cpu().detach().numpy(), arg_preds.cpu().detach().numpy())
                # note that for AUC calculation we use sigmoid vals
                fpr, tpr, thresholds = roc_curve(truths.cpu().detach().numpy(), nn.Sigmoid()(results).cpu().detach().numpy())
                auc_out = auc(fpr, tpr)
                print(cond, 'accuracy', class_accuracy, 'auc', auc_out)
        return avg_loss, results, truths, class_accuracy, auc_out

    best_valid = 1e8
    early_stopping = 0
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion, ctc_p2_module, ctc_r2_module, ctc_p2_optimizer, ctc_r2_optimizer, ctc_criterion)
        val_loss, _, _, val_class_accuracy, val_auc = evaluate(model, ctc_p2_module, ctc_r2_module, criterion, test=False)
        test_loss, _, _, test_class_accuracy, test_auc = evaluate(model, ctc_p2_module, ctc_r2_module, criterion, test=True)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        col_vals = [hyp_params.dataset_type, hyp_params.label_type, hyp_params.criterion, 
                    hyp_params.raw_kernel_size, hyp_params.raw_stride, hyp_params.feat_kernel_size, hyp_params.feat_stride, 
                    hyp_params.batch_size, epoch, train_loss, val_loss, test_loss,
                   val_class_accuracy, test_class_accuracy, val_auc, test_auc]
        if epoch == 1: # overwrite existing
            col_headers = ['dataset', 'label_type', 'criterion', 
                       'raw_kernel_size', 'raw_stride', 'feat_kernel_size', 'feat_stride', 
                       'batch_size', 'epoch', 'train_loss', 'val_loss', 'test_loss',
                          'val_accuracy','test_accuracy', 'val_auc', 'test_auc']
            if hyp_params.dataset == 'nidyn':
                col_vals.append('hyp_params.h5_filename')
                col_headers.append(hyp_params.h5_filename)
                with open(f'pre_trained_models/{hyp_params.dataset_type}_{hyp_params.h5_filename}.csv','w') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(col_headers)
                    writer.writerow(col_vals)
            else:
                with open(f'pre_trained_models/{hyp_params.dataset_type}_{hyp_params.valence_arousal}.csv','w') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(col_headers)
                    writer.writerow(col_vals)
        else:
            if hyp_params.dataset == 'nidyn':
                col_vals.append('hyp_params.h5_filename')
                col_headers.append(hyp_params.h5_filename)
                with open(f'pre_trained_models/{hyp_params.dataset_type}_{hyp_params.h5_filename}.csv','a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(col_vals)
            else:
                with open(f'pre_trained_models/{hyp_params.dataset_type}_{hyp_params.valence_arousal}.csv','a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(col_vals)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.dataset_type}_{hyp_params.valence_arousal}.pt!")
            save_model(hyp_params, model, name=hyp_params.dataset_type+'_'+hyp_params.valence_arousal)
            best_valid = val_loss
            early_stopping = 0
        else:
            if early_stopping==10:
                print(f"Saved model at pre_trained_models/{hyp_params.dataset_type}_{hyp_params.valence_arousal}_early_stopped.pt!")
                save_model(hyp_params, model, name=f"{hyp_params.dataset_type}_{hyp_params.valence_arousal}_early_stopped")
            early_stopping += 1

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, ctc_p2_module, ctc_r2_module, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)
    elif 'deap' in hyp_params.dataset:
        eval_deap(results, truth)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
 