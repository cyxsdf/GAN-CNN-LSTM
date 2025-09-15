import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.modalities != 'AV':
        if hyp_params.modalities == 'A':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator = translator.cuda()
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'V':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator = translator.cuda()
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        trans_criterion = getattr(nn, 'MSELoss')()
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    if hyp_params.modalities != 'AV':
        if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            settings = {'model': model,
                        'translator': translator,
                        'translator_optimizer': translator_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        else:
            raise ValueError('Unknown modalities type')
    elif hyp_params.modalities == 'AV':
        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
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

    scheduler = settings['scheduler']

    if hyp_params.modalities != 'AV':
        trans_criterion = settings['trans_criterion']
        if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']
        else:
            raise ValueError('Unknown modalities type')
    else:
        translator = None

    # 对比损失权重（可根据需要调整）
    contrast_weight = 0.1 if hasattr(hyp_params, 'contrast_weight') else 0.1

    def train(model, translator, optimizer, criterion):
        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'AV':
            if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator.train()
            else:
                raise ValueError('Unknown modalities type')
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (audio, video, masks, labels) in enumerate(train_loader):

            model.zero_grad()
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    translator.zero_grad()
                else:
                    raise ValueError('Unknown modalities type')

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    video, audio, masks, labels = video.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
            masks_video = masks.unsqueeze(-1).expand(-1, hyp_params.v_len, hyp_params.orig_d_v)
            audio = audio * masks_audio
            video = video * masks_video
            batch_size = audio.size(0)  # 获取并保存当前批次大小

            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = torch.tensor(0.0, device=audio.device)  # 初始化翻译损失
            contrast_loss = torch.tensor(0.0, device=audio.device)  # 初始化对比损失

            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    if hyp_params.modalities == 'A':
                        # 获取所有返回值，只使用前两个
                        trans_outputs = trans_net(audio, video, 'train')
                        fake_v = trans_outputs[0]
                        contrast_loss = trans_outputs[1] if len(trans_outputs) > 1 else 0

                        # 确保fake_v与video维度和批次大小一致
                        if fake_v.dim() != video.dim():
                            fake_v = fake_v.unsqueeze(0) if fake_v.dim() < video.dim() else fake_v.squeeze(0)
                        if fake_v.size(0) != batch_size:
                            fake_v = fake_v.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                        trans_loss = trans_criterion(fake_v, video)
                    elif hyp_params.modalities == 'V':
                        # 获取所有返回值，只使用前两个
                        trans_outputs = trans_net(video, audio, 'train')
                        fake_a = trans_outputs[0]
                        contrast_loss = trans_outputs[1] if len(trans_outputs) > 1 else 0

                        # 确保fake_a与audio维度和批次大小一致
                        if fake_a.dim() != audio.dim():
                            fake_a = fake_a.unsqueeze(0) if fake_a.dim() < audio.dim() else fake_a.squeeze(0)
                        if fake_a.size(0) != batch_size:
                            fake_a = fake_a.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                        trans_loss = trans_criterion(fake_a, audio)
                    else:
                        raise ValueError('Unknown modalities type')

            # 处理主模型输出
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'A':
                    # 确保输入维度和批次大小完全匹配
                    if fake_v.dim() != video.dim():
                        fake_v = fake_v.view_as(video)
                    if fake_v.size(0) != video.size(0):
                        fake_v = fake_v[:video.size(0)]  # 截断到正确的批次大小
                    outputs = net(audio, fake_v)
                    preds = outputs[0]  # 只使用第一个返回值作为预测结果
                elif hyp_params.modalities == 'V':
                    # 确保输入维度和批次大小完全匹配
                    if fake_a.dim() != audio.dim():
                        fake_a = fake_a.view_as(audio)
                    if fake_a.size(0) != audio.size(0):
                        fake_a = fake_a[:audio.size(0)]  # 截断到正确的批次大小
                    outputs = net(fake_a, video)
                    preds = outputs[0]  # 只使用第一个返回值作为预测结果
                else:
                    raise ValueError('Unknown modalities type')
            elif hyp_params.modalities == 'AV':
                outputs = net(audio, video)
                preds = outputs[0]  # 只使用第一个返回值作为预测结果
            else:
                raise ValueError('Unknown modalities type')

            raw_loss = criterion(preds.transpose(1, 2), labels)
            if hyp_params.modalities != 'AV':
                # 结合原始损失、翻译损失和对比损失
                combined_loss = raw_loss + trans_loss + contrast_weight * contrast_loss
            else:
                combined_loss = raw_loss
            combined_loss.backward()

            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()
                else:
                    raise ValueError('Unknown modalities type')

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        model.eval()
        if hyp_params.modalities != 'AV':
            if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator.eval()
            else:
                raise ValueError('Unknown modalities type')
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, video, masks, labels) in enumerate(loader):

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        video, audio, masks, labels = video.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
                masks_video = masks.unsqueeze(-1).expand(-1, hyp_params.v_len, hyp_params.orig_d_v)
                audio = audio * masks_audio
                video = video * masks_video
                batch_size = audio.size(0)  # 保存当前批次大小

                net = nn.DataParallel(model) if hyp_params.distribute else model
                trans_loss = torch.tensor(0.0, device=audio.device)

                if hyp_params.modalities != 'AV':
                    if not test:
                        if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'A':
                                # 获取所有返回值，只使用第一个
                                trans_outputs = trans_net(audio, video, 'valid')
                                fake_v = trans_outputs[0]

                                # 确保fake_v与video维度和批次大小一致
                                if fake_v.dim() != video.dim():
                                    fake_v = fake_v.unsqueeze(0) if fake_v.dim() < video.dim() else fake_v.squeeze(0)
                                if fake_v.size(0) != batch_size:
                                    fake_v = fake_v.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                                trans_loss = trans_criterion(fake_v, video)
                            elif hyp_params.modalities == 'V':
                                # 获取所有返回值，只使用第一个
                                trans_outputs = trans_net(video, audio, 'valid')
                                fake_a = trans_outputs[0]

                                # 确保fake_a与audio维度和批次大小一致
                                if fake_a.dim() != audio.dim():
                                    fake_a = fake_a.unsqueeze(0) if fake_a.dim() < audio.dim() else fake_a.squeeze(0)
                                if fake_a.size(0) != batch_size:
                                    fake_a = fake_a.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                                trans_loss = trans_criterion(fake_a, audio)
                            else:
                                raise ValueError('Unknown modalities type')
                    else:
                        if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'A':
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        trans_outputs = trans_net(audio, video, 'test', eval_start=True)
                                        fake_v_token = trans_outputs[0][:, [-1]]
                                    else:
                                        trans_outputs = trans_net(audio, fake_v, 'test')
                                        fake_v_token = trans_outputs[0][:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                                # 确保fake_v与video维度和批次大小一致
                                if fake_v.dim() != video.dim():
                                    fake_v = fake_v.unsqueeze(0) if fake_v.dim() < video.dim() else fake_v.squeeze(0)
                                if fake_v.size(0) != batch_size:
                                    fake_v = fake_v.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                            elif hyp_params.modalities == 'V':
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        trans_outputs = trans_net(video, audio, 'test', eval_start=True)
                                        fake_a_token = trans_outputs[0][:, [-1]]
                                    else:
                                        trans_outputs = trans_net(video, fake_a, 'test')
                                        fake_a_token = trans_outputs[0][:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)

                                # 确保fake_a与audio维度和批次大小一致
                                if fake_a.dim() != audio.dim():
                                    fake_a = fake_a.unsqueeze(0) if fake_a.dim() < audio.dim() else fake_a.squeeze(0)
                                if fake_a.size(0) != batch_size:
                                    fake_a = fake_a.repeat(batch_size, 1, 1)[:batch_size]  # 调整批次大小
                            else:
                                raise ValueError('Unknown modalities type')
                        else:
                            raise ValueError('Unknown modalities type')

                # 处理主模型输出
                if hyp_params.modalities != 'AV':
                    if hyp_params.modalities == 'A':
                        # 确保输入维度和批次大小完全匹配
                        if fake_v.dim() != video.dim():
                            fake_v = fake_v.view_as(video)
                        if fake_v.size(0) != video.size(0):
                            fake_v = fake_v[:video.size(0)]  # 截断到正确的批次大小
                        outputs = net(audio, fake_v)
                        preds = outputs[0]
                    elif hyp_params.modalities == 'V':
                        # 确保输入维度和批次大小完全匹配
                        if fake_a.dim() != audio.dim():
                            fake_a = fake_a.view_as(audio)
                        if fake_a.size(0) != audio.size(0):
                            fake_a = fake_a[:audio.size(0)]  # 截断到正确的批次大小
                        outputs = net(fake_a, video)
                        preds = outputs[0]
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'AV':
                    outputs = net(audio, video)
                    preds = outputs[0]
                else:
                    raise ValueError('Unknown modalities type')

                raw_loss = criterion(preds.transpose(1, 2), labels)
                if hyp_params.modalities != 'AV' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss
                total_loss += combined_loss.item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    if hyp_params.modalities != 'AV':
        if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        else:
            raise ValueError('Unknown modalities type')
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')
    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()
        train(model, translator, optimizer, criterion)
        val_loss, _, _, _ = evaluate(model, translator, criterion, test=False)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        if val_loss < best_valid:
            if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
        translator = load_model(hyp_params, name='TRANSLATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, mask = evaluate(model, translator, criterion, test=True)

    acc = eval_ur_funny(results, truths, mask)

    return acc
