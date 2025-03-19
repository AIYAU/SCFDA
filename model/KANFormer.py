import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from KAN import KANLinear


class KANBlock(nn.Module):
    def __init__(self,input_dim,spline_order=3,grid_size=5):
        super(KANBlock, self).__init__()
        self.kan_layer_01 = KANLinear(input_dim,input_dim,spline_order=spline_order,grid_size=grid_size)
        self.kan_layer_02 = KANLinear(input_dim,input_dim,spline_order=spline_order,grid_size=grid_size)

    def forward(self, x):
        shorcut = x
        score = F.adaptive_avg_pool2d(x,(1,1))
        score = score.squeeze(-1).squeeze(-1)
        score = self.kan_layer_01(score)
        score = self.kan_layer_02(score)
        score = rearrange(score, 'b c -> b c 1 1')
        x = x * score
        return x + shorcut

    


class Fusion(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=256,scale=4,depth=4,image_size=64):
        super(Fusion, self).__init__()
        self.hsi_kan = KANLinear(HSI_bands,hidden_dim)
        self.msi_kan = KANLinear(MSI_bands,hidden_dim)
        self.align_kan = KANLinear(hidden_dim*2,hidden_dim)
        self.scale = scale
        self.image_size = image_size

    def forward(self, LRHSI, HRMSI):
        up_LRHSI = F.interpolate(LRHSI, scale_factor=self.scale, mode='bicubic', align_corners=True)
        lrhsi_feats = rearrange(up_LRHSI, 'b c h w -> b (h w) c')
        hrmsi_feats = rearrange(HRMSI, 'b c h w -> b (h w) c')
        lrhsi_feats = self.hsi_kan(lrhsi_feats)
        hrmsi_feats = self.msi_kan(hrmsi_feats)
        feats = torch.cat([lrhsi_feats, hrmsi_feats], dim=-1)  
        feats = self.align_kan(feats)  
        feats = rearrange(feats, 'b (h w) c -> b c h w', h=self.image_size)
        return feats
    
class KANFormer(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=256,scale=4,depth=4,image_size=64):
        super(KANFormer, self).__init__()
        self.HSI_bands = HSI_bands
        self.MSI_bands = MSI_bands
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.fusion = Fusion(HSI_bands=HSI_bands,MSI_bands=MSI_bands,hidden_dim=hidden_dim,scale=scale,depth=depth,image_size=image_size)
        self.layers = nn.ModuleList([KANBlock(hidden_dim, hidden_dim) \
                                     for i in range(depth)])
        
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, HSI_bands, 3,1,1)
        )

    def forward(self, LRHSI, HRMSI):
        up_HSI = F.interpolate(LRHSI, scale_factor=self.scale, mode='bicubic', align_corners=True)
        x = self.fusion(LRHSI, HRMSI)
        print(x.shape)
        for layer in self.layers:
            x = layer(x)
        x = self.refine(x)

        print(x.shape)
        print(up_HSI.shape)

        return x + up_HSI

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum([layer.kan_layer_01.regularization_loss(regularize_activation, regularize_entropy)  + \
                    layer.kan_layer_02.regularization_loss(regularize_activation, regularize_entropy)    \
                    for layer in self.layers] )

if __name__ == '__main__':
    import torch
    from torchinfo import summary

    # 设置模型参数
    HSI_bands = 31  # 高光谱图像的通道数
    MSI_bands = 3  # 多光谱图像的通道数
    hidden_dim = 256
    scale = 4
    depth = 4
    image_size = 64  # 输入低分辨率图像的尺寸

    # 初始化模型
    model = KANFormer(HSI_bands=HSI_bands, MSI_bands=MSI_bands, hidden_dim=hidden_dim, scale=scale, depth=depth,
                      image_size=image_size)

    # 模拟输入
    batch_size = 1
    LRHSI = torch.rand(batch_size, HSI_bands, image_size, image_size)  # 低分辨率高光谱图像
    HRMSI = torch.rand(batch_size, MSI_bands, image_size * scale, image_size * scale)  # 高分辨率多光谱图像
    print("LRHSI:",LRHSI.shape)
    print("HRMSI",HRMSI.shape)
    # 使用torchinfo查看模型信息
    preHSI = model(LRHSI, HRMSI)

    print("preHSI:",preHSI.shape)
    # summary(model, input_data=(LRHSI, HRMSI), depth=3,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size"])
