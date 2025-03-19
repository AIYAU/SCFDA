import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims
        self.video_in = input_dims
        # self.text_in = input_dims

        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        # define the post-fusion layers
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        # self.text_factor = Parameter(torch.Tensor(self.rank, self.text_in + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # initialize the factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        # xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x: torch.Tensor, video_x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, text_in)
        '''
        batch_size = audio_x.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        DTYPE = torch.cuda.FloatTensor if audio_x.is_cuda else torch.FloatTensor

        _audio_h = torch.cat((torch.ones(batch_size, 1, dtype=audio_x.dtype, device=audio_x.device), audio_x), dim=1)
        _video_h = torch.cat((torch.ones(batch_size, 1, dtype=video_x.dtype, device=video_x.device), video_x), dim=1)
        # _text_h = torch.cat((torch.ones(batch_size, 1, dtype=text_x.dtype, device=text_x.device), text_x), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        # fusion_text = torch.matmul(_text_h, self.text_factor)
        # fusion_zy = fusion_audio * fusion_video * fusion_text
        fusion_zy = fusion_audio * fusion_video

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output, dim=-1)
        return output

if __name__ == '__main__':
    def test_lmf():
        # Assume input shape is (batch_size=16, feature_dim=128) for each modality
        batch_size = 16
        feature_dim = 128
        output_dim = 128
        rank = 10
        # Create random inputs for audio, video, and text
        audio_input = torch.randn(batch_size, feature_dim)
        video_input = torch.randn(batch_size, feature_dim)
        text_input = torch.randn(batch_size, feature_dim)
        # Instantiate LMF model
        input_dims = (feature_dim, feature_dim, feature_dim)
        model = LMF(128, output_dim, rank, use_softmax=True)
        # Forward pass
        output = model(audio_input, video_input)
        print("Output shape:", output.shape)
        print("Output:", output)

    # Run the test function
    test_lmf()