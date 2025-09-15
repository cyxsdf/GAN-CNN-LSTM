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
# GAN Modules for Missing Modality Generation
#
####################################################################

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, seq_len=33, cnn_filters=64, kernel_size=3):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # CNN层：提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
        )

        # LSTM层：捕捉序列特征
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 输出层：映射到目标模态维度
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2因为双向LSTM

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # 转换为(batch_size, input_dim, seq_len)以适应CNN输入
        x = self.cnn(x)  # (batch_size, cnn_filters*2, seq_len)
        x = x.transpose(1, 2)  # 转换回(batch_size, seq_len, cnn_filters*2)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)

        # 输出层
        x = self.fc(lstm_out)  # (batch_size, seq_len, output_dim)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len=33, hidden_dim=256, cnn_filters=32, kernel_size=3):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=1),
            nn.LeakyReLU(0.2),
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 输出层（二分类：真实/伪造）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)  # (batch_size, cnn_filters*2, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, cnn_filters*2)

        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        x = self.fc(lstm_out)  # (batch_size, seq_len, 1)
        return x


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 初始化GAN组件
    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L':
            # 文本模态输入，生成音频模态
            input_dim = hyp_params.orig_d_l  # 文本特征维度
            output_dim = hyp_params.orig_d_a  # 音频特征维度
            generator = Generator(
                input_dim,
                output_dim,
                seq_len=hyp_params.l_len,
                cnn_filters=64,
                kernel_size=3
            )
            discriminator = Discriminator(
                output_dim,
                seq_len=hyp_params.a_len,
                cnn_filters=32,
                kernel_size=3
            )
        elif hyp_params.modalities == 'A':
            # 音频模态输入，生成文本模态
            input_dim = hyp_params.orig_d_a  # 音频特征维度
            output_dim = hyp_params.orig_d_l  # 文本特征维度
            generator = Generator(
                input_dim,
                output_dim,
                seq_len=hyp_params.a_len,
                cnn_filters=64,
                kernel_size=3
            )
            discriminator = Discriminator(
                output_dim,
                seq_len=hyp_params.l_len,
                cnn_filters=32,
                kernel_size=3
            )

        # 优化器设置
        gen_optimizer = optim.Adam(generator.parameters(), lr=hyp_params.gen_lr, betas=(0.5, 0.999))
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=hyp_params.dis_lr, betas=(0.5, 0.999))
        gan_criterion = nn.BCELoss()  # GAN损失函数
        recon_criterion = nn.MSELoss()  # 重构损失函数

    # 主任务模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
        if hyp_params.modalities != 'LA':
            generator = generator.cuda()
            discriminator = discriminator.cuda()

    # 主模型优化器
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 配置训练参数
    if hyp_params.modalities != 'LA':
        settings = {
            'model': model,
            'generator': generator,
            'discriminator': discriminator,
            'gen_optimizer': gen_optimizer,
            'dis_optimizer': dis_optimizer,
            'gan_criterion': gan_criterion,
            'recon_criterion': recon_criterion,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }
    elif hyp_params.modalities == 'LA':
        settings = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }
    else:
        raise ValueError('Unknown modalities type')

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

    # GAN组件
    if hyp_params.modalities != 'LA':
        generator = settings['generator']
        discriminator = settings['discriminator']
        gen_optimizer = settings['gen_optimizer']
        dis_optimizer = settings['dis_optimizer']
        gan_criterion = settings['gan_criterion']
        recon_criterion = settings['recon_criterion']
    else:
        generator = None
        discriminator = None

    def train():
        model.train()
        epoch_loss = 0
        if hyp_params.modalities != 'LA':
            generator.train()
            discriminator.train()

        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            # 应用掩码
            masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
            masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600) if hyp_params.dataset == 'meld_senti' \
                else masks.unsqueeze(-1).expand(-1, 33, 300)
            text = text * masks_text
            audio = audio * masks_audio
            batch_size = text.size(0)

            # 主模型梯度清零
            model.zero_grad()

            if hyp_params.modalities != 'LA':
                # GAN梯度清零
                generator.zero_grad()
                discriminator.zero_grad()

                # 生成缺失模态
                if hyp_params.modalities == 'L':
                    # 文本生成音频
                    fake_audio = generator(text)
                    real_data = audio
                    gen_data = fake_audio
                elif hyp_params.modalities == 'A':
                    # 音频生成文本
                    fake_text = generator(audio)
                    real_data = text
                    gen_data = fake_text

                # 1. 训练判别器
                # 真实数据
                real_label = torch.ones_like(real_data[..., :1])  # (batch, seq_len, 1)
                fake_label = torch.zeros_like(real_data[..., :1])

                real_pred = discriminator(real_data)
                dis_loss_real = gan_criterion(real_pred, real_label)

                # 生成数据（detach避免更新生成器）
                fake_pred = discriminator(gen_data.detach())
                dis_loss_fake = gan_criterion(fake_pred, fake_label)

                dis_loss = (dis_loss_real + dis_loss_fake) * 0.5
                dis_loss.backward(retain_graph=True)
                dis_optimizer.step()

                # 2. 训练生成器
                # GAN损失（希望生成数据被判别为真实）
                gen_pred = discriminator(gen_data)
                gen_loss_gan = gan_criterion(gen_pred, real_label)

                # 重构损失（生成数据与真实数据的相似度）
                gen_loss_recon = recon_criterion(gen_data, real_data)

                # 联合损失
                gen_loss = gen_loss_gan * hyp_params.gan_weight + gen_loss_recon

            # 3. 主任务训练
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L':
                    preds, _ = model(text, fake_audio)
                elif hyp_params.modalities == 'A':
                    preds, _ = model(fake_text, audio)
            else:
                preds, _ = model(text, audio)

            # 主任务损失
            task_loss = criterion(preds.transpose(1, 2), labels)

            # 总损失计算
            if hyp_params.modalities != 'LA':
                total_loss = task_loss + gen_loss
            else:
                total_loss = task_loss

            total_loss.backward()

            # 参数更新
            if hyp_params.modalities != 'LA':
                gen_optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += total_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(test=False):
        model.eval()
        if hyp_params.modalities != 'LA':
            generator.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                # 应用掩码
                masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600) if hyp_params.dataset == 'meld_senti' \
                    else masks.unsqueeze(-1).expand(-1, 33, 300)
                text = text * masks_text
                audio = audio * masks_audio
                batch_size = text.size(0)

                # 生成缺失模态（评估阶段）
                if hyp_params.modalities != 'LA':
                    if hyp_params.modalities == 'L':
                        fake_audio = generator(text)
                        preds, _ = model(text, fake_audio)
                    elif hyp_params.modalities == 'A':
                        fake_text = generator(audio)
                        preds, _ = model(fake_text, audio)
                else:
                    preds, _ = model(text, audio)

                # 计算损失
                loss = criterion(preds.transpose(1, 2), labels)
                total_loss += loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 模型参数统计
    if hyp_params.modalities != 'LA':
        gen_params = sum([param.nelement() for param in generator.parameters()])
        dis_params = sum([param.nelement() for param in discriminator.parameters()])
        print(f'Generator Parameters: {gen_params}, Discriminator Parameters: {dis_params}...')
    mum_params = sum([param.nelement() for param in model.parameters()])
    print(f'Multimodal Understanding Model Parameters: {mum_params}...')

    # 训练主循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练
        train_loss = train()

        # 验证
        val_loss, _, _, _ = evaluate(test=False)
        end = time.time()
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            best_valid = val_loss
            save_model(hyp_params, model, name=hyp_params.name)
            if hyp_params.modalities != 'LA':
                save_model(hyp_params, generator, name='GENERATOR')
                save_model(hyp_params, discriminator, name='DISCRIMINATOR')

    # 加载最佳模型并测试
    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.modalities != 'LA':
        generator = load_model(hyp_params, name='GENERATOR')
    _, results, truths, mask = evaluate(test=True)
    acc = eval_meld(results, truths, mask)

    return acc