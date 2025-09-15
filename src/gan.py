import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    """基础生成器类，可根据需求扩展"""

    def __init__(self, input_dim, output_dim, hidden_dim=512, seq_len=33):
        super().__init__()
        self.seq_len = seq_len
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(seq_len),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(seq_len),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerGenerator(Generator):
    """基于Transformer的生成器，用于更复杂的模态生成"""

    def __init__(self, hyp_params, missing):
        super().__init__(
            input_dim=hyp_params.embed_dim,
            output_dim=self._get_output_dim(hyp_params, missing),
            hidden_dim=hyp_params.hidden_dim
        )
        self.hyp_params = hyp_params
        self.missing = missing
        self.embed_dim = hyp_params.embed_dim

        # 输入投影
        self.input_dim = self._get_input_dim(hyp_params, missing)
        self.proj_in = nn.Linear(self.input_dim, self.embed_dim)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=hyp_params.num_heads,
            layers=hyp_params.trans_layers,
            attn_dropout=hyp_params.attn_dropout,
            relu_dropout=hyp_params.relu_dropout,
            res_dropout=hyp_params.res_dropout
        )

        # 位置和模态嵌入
        self.position_embeddings = nn.Embedding(100, self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

    def _get_input_dim(self, hyp_params, missing):
        if missing == 'L':
            return hyp_params.orig_d_a + (
                hyp_params.orig_d_v if hyp_params.dataset not in ['meld_senti', 'meld_emo'] else 0)
        elif missing == 'A':
            return hyp_params.orig_d_l + (
                hyp_params.orig_d_v if hyp_params.dataset not in ['meld_senti', 'meld_emo'] else 0)
        elif missing == 'V':
            return hyp_params.orig_d_l + hyp_params.orig_d_a
        raise ValueError(f"Unknown missing modality: {missing}")

    def _get_output_dim(self, hyp_params, missing):
        return {
            'L': hyp_params.orig_d_l,
            'A': hyp_params.orig_d_a,
            'V': hyp_params.orig_d_v
        }[missing]

    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        x = F.dropout(F.relu(self.proj_in(src)), p=0.1, training=self.training)
        x = x.transpose(0, 1)

        # 添加位置和模态嵌入
        pos_ids = torch.arange(seq_len, device=src.device).unsqueeze(1).expand(-1, batch_size)
        x = x + self.position_embeddings(pos_ids)
        modal_type = 0 if self.missing != 'L' else 1
        x = x + self.modal_type_embeddings(torch.full_like(pos_ids, modal_type))

        x = self.transformer(x)
        return self.layers(x.transpose(0, 1))


class Discriminator(nn.Module):
    """通用判别器，支持不同输入维度"""

    def __init__(self, input_dim, hidden_dim=256, use_seq_avg=True):
        super().__init__()
        self.use_seq_avg = use_seq_avg  # 是否对序列维度平均
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.use_seq_avg and x.dim() == 3:
            x = x.mean(dim=1)  # 序列维度平均
        return self.model(x)