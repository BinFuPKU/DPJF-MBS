import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WORD_EMBED_DIM = 32
USER_EMBED_DIM = 16

class Profile2Vec(torch.nn.Module):
    def __init__(self, word_embeddings, name):
        super(Profile2Vec, self).__init__()
        self.name = name
        # profile: word embeddings for look_up
        # embedding_matrix = [[0...0], [...], ...[]]
        self.Word_Embeds = word_embeddings

        self.ConvNet = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=WORD_EMBED_DIM, out_channels=WORD_EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            torch.nn.BatchNorm1d(WORD_EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3),

            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=WORD_EMBED_DIM, out_channels=USER_EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            torch.nn.BatchNorm1d(USER_EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=50)
        )

    # profiles: [batch_size, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50), word idx
    # return [batch_size, USER_EMBED_DIM]
    def forward(self, profiles):
        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size, MAX_PROFILELEN * MAX_TERMLEN](40 * 15, 50)
        shape = profiles.shape
        profiles_ = profiles.view([shape[0], -1])

        # embeddings: [batch_size, MAX_PROFILELEN * MAX_TERMLEN, EMBED_DIM]
        profiles_wordembed = self.Word_Embeds(profiles_).float()

        # permute for conv1d
        # embeddings: [batch_size, EMBED_DIM, MAX_PROFILELEN * MAX_TERMLEN]
        profiles_wordembed_ = profiles_wordembed.permute(0, 2, 1)

        # [batch_size, EMBED_DIM, x]
        profiles_convs_out = self.ConvNet(profiles_wordembed_)

        # [batch_size, EMBED_DIM, x] -> [batch_size, EMBED_DIM, 1]
        profiles_len = profiles_convs_out.shape[-1]
        profiles_final_out = torch.nn.MaxPool1d(kernel_size=profiles_len)(profiles_convs_out).squeeze(-1)
        return profiles_final_out