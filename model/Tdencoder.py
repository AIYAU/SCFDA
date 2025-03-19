import torch
import torch.nn as nn
import torch.fft

from model.low_rank_fusion import LMF
from model.moe import MultiFeatureMOE


def scan_and_extract_blocks(input_tensor, block_shape):
    """
    批量扫描并提取块。
    Args:
        input_tensor (Tensor): 输入张量，形状为 (batch_size, height, width, depth)。
        block_shape (tuple): 块的形状 (blk_height, blk_width, blk_depth)。
    Returns:
        Tensor: 扫描得到的所有块，形状为 (batch_size, num_blocks, blk_height, blk_width, blk_depth)。
    """
    # 获取输入和块的大小
    batch_size, in_height, in_width, in_depth = input_tensor.shape
    blk_height, blk_width, blk_depth = block_shape
    assert blk_depth == in_depth, "The block depth must be the same as the input depth"
    # 计算扫描中心位置
    center_h = in_height // 2
    center_w = in_width // 2
    # 定义扫描的相对偏移
    offsets = torch.tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])  # 9 个扫描位置 (i, j)
    # 初始化用于存储块的列表
    blocks = []
    # 遍历每个偏移量，提取对应的块
    for offset in offsets:
        # 计算起始和结束位置
        start_h = center_h + offset[0] - blk_height // 2
        end_h = start_h + blk_height
        start_w = center_w + offset[1] - blk_width // 2
        end_w = start_w + blk_width
        # 检查边界条件
        if 0 <= start_h < in_height - blk_height + 1 and 0 <= start_w < in_width - blk_width + 1:
            # 提取每个样本中的块
            block = input_tensor[:, start_h:end_h, start_w:end_w,
                    :]  # 形状 (batch_size, blk_height, blk_width, blk_depth)
            blocks.append(block)

    # 堆叠所有块 (batch_size, num_blocks, blk_height, blk_width, blk_depth)
    blocks = torch.stack(blocks, dim=1)
    return blocks

class SpectralEncoder(nn.Module): # 光谱特征
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpectralEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        self.conv1 = nn.Conv3d(1, self.inter_size, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                               bias=True)
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                               padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                               padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn3 = nn.BatchNorm3d(self.inter_size)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(self.inter_size, self.feature_dim,
                               kernel_size=(((self.input_channels - 7 + 2 * 1) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.feature_dim)
        self.activation4 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))
        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        # print(f"光谱未池化：{x1.shape}")
        x1 = self.avgpool(x1)
        # print(f"光谱未展平：{x1.shape}")
        x1 = x1.reshape((x1.size(0), -1))
        # print(f"光谱全连接前：{x1.shape}")
        return x1


class SpatialEncoder(nn.Module): # 空间特征
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpatialEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, self.inter_size, kernel_size=(self.input_channels, 1, 1))
        self.bn5 = nn.BatchNorm3d(self.inter_size)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv8 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 1, 1))

        self.conv6 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn6 = nn.BatchNorm3d(self.inter_size)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn7 = nn.BatchNorm3d(self.inter_size)
        self.activation7 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(self.inter_size, out_features=self.feature_dim))

    def forward(self, x):
        #print(f'拓展前{x.shape}')
        x = x.unsqueeze(1)
        #print(f'拓展后{x.shape}')
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))
        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2
        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        # print(f"空间未池化：{x2.shape}")
        x2 = self.avgpool(x2)
        # print(f"空间未展平：{x2.shape}")
        x2 = x2.reshape((x2.size(0), -1))
        # print(f"空间全连接前：{x2.shape}")
        x2 = self.fc(x2)
        return x2


class WordEmbTransformers(nn.Module):
    def __init__(self, feature_dim, dropout):
        super(WordEmbTransformers, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.fc = nn.Sequential(nn.Linear(in_features=768,
                                          out_features=128,
                                          bias=True),
                                nn.ReLU(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(in_features=128,
                                          out_features=self.feature_dim,
                                          bias=True)
                                )

    def forward(self, x):
        # 0-1
        x = self.fc(x)
        return x


class AttentionWeight(nn.Module):
    def __init__(self, feature_dim, hidden_layer, dropout):
        super(AttentionWeight, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_layer = hidden_layer
        self.dropout = dropout

        self.getAttentionWeight = nn.Sequential(nn.Linear(in_features=self.feature_dim,
                                                          out_features=self.hidden_layer),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(in_features=self.hidden_layer,
                                                          out_features=1),
                                                nn.Sigmoid()
                                                )

    def forward(self, x):
        x = self.getAttentionWeight(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_dimension, patch_size, sub_patch_size,emb_size, dropout=0.5):
        super(Encoder, self).__init__()
        self.n_dimension = n_dimension
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.emb_size = emb_size
        self.dropout = dropout

        self.spectral_encoder = SpectralEncoder(input_channels=self.n_dimension, patch_size=self.patch_size,
                                                feature_dim=self.emb_size)# spectral encoder

        self.sub_spatial_encoder = SpatialEncoder(input_channels=self.n_dimension, patch_size=self.sub_patch_size,
                                              feature_dim=self.emb_size) # sub_block encoder

        self.spatial_encoder = SpatialEncoder(input_channels=self.n_dimension, patch_size=self.patch_size,
                                              feature_dim=self.emb_size) # all_block encoder

        self.freq_encoder = FrequencyFeatureExtractor() # freq encoder

        self.word_emb_transformers = WordEmbTransformers(feature_dim=self.emb_size, dropout=self.dropout)
        self.moe = MultiFeatureMOE(feature_dim = 128, num_features = 9, num_experts = 2, expert_hidden_dim = 128, output_dim = 128)

        input_dims = (128, 128, 128)
        self.fusion = LMF(128, 128, 20, use_softmax=True)
        # Forward pass


    def forward(self, x, semantic_feature="", s_or_q="query"):
        spatial_feature = self.spatial_encoder(x) # 提取空间特征
        spectral_feature = self.spectral_encoder(x) # 提取光谱特征
        # get freq feature
       
        magnitude_features = self.freq_encoder(x) # 提取频域特征
        # 提取分块特征(16,9,3,3,100)===>(9,16,3,3,100)
        sub_block = scan_and_extract_blocks(x.permute(0, 2, 3, 1), block_shape = (3, 3, 100))
        # 对9个子块分别提取，能得到一个9个16,128的特征
        sub_block = sub_block.permute(1, 0, 4, 2, 3) # 维度变换
        sub_features = []
        for block in sub_block:
            # 提取分特征
            sub_features.append(self.sub_spatial_encoder(block)) # 得到分特征（9,16,128）
        moe_feature = self.moe(sub_features) #经过n个专家筛选的 上下文特征 16,128

        image_feature = self.fusion(moe_feature, spatial_feature)
        spatial_spectral_fusion_feature = 0.9 * image_feature + 0.1 * spectral_feature
        # spatial_spectral_fusion_feature = image_feature
        # spatial_spectral_fusion_feature = self.fusion(spectral_feature, spatial_feature,magnitude_features)# 空间光谱频域通过低秩融合
        # support set extract fusion_feature
        if s_or_q == "support":  # semantic_feature = (9, 768)
            semantic_feature = self.word_emb_transformers(semantic_feature)  # (9, 128) 提取语义特征
            return spatial_spectral_fusion_feature, semantic_feature,magnitude_features # 返回
        # query set extract spatial_spectral_fusion_feature
        return spatial_spectral_fusion_feature
class FrequencyFeatureExtractor(nn.Module):
    def __init__(self):
        super(FrequencyFeatureExtractor, self).__init__()
        # 用于频域特征的卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128)
        )
    def forward(self, x):
        # 计算频域的幅值
        freq_data = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(freq_data)
        # 进一步提取频域特征
        # print(f"初始频域 shape{freq_data.shape}")
        magnitude_features = self.conv(magnitude)
        return magnitude_features

if __name__ == '__main__':
    from torchinfo import summary
    encoder = Encoder(n_dimension=100,sub_patch_size=3, patch_size=7, emb_size=128, dropout=0.3)
    print(encoder)
    output = encoder(torch.randn(16,100,7,7))
    # summary(encoder, input_size=(16, 100, 7, 7))
    # 打印输出
    print(f"输出：{output.shape}")


