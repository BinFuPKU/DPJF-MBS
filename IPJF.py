
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src.Preprocess.profile_word2vec import EMBED_DIM


class IPJF(torch.nn.Module):
    def __init__(self, word_embeddings,):
        super(IPJF, self).__init__()

        # embedding_matrix = [[0...0], [...], ...[]]
        self.Word_Embeds = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.Word_Embeds.weight.requires_grad = False

        self.Expect_ConvNet = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3),

            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            # torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=50)
        )

        self.Job_ConvNet = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=5),
            # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2),

            # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,...
            # torch.nn.Conv1d(in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=3),
            # # BatchNorm1d只处理第二个维度
            # torch.nn.BatchNorm1d(EMBED_DIM),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=50)
        )

        # match mlp
        self.Match_MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * EMBED_DIM, EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(EMBED_DIM, 1),
            torch.nn.Sigmoid()
        )

    # [batch_size *2, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50)
    # term: padding same, word: padding 0
    # expects_sample, jobs_sample are in same format
    def forward(self, expects, jobs):

        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size, MAX_PROFILELEN * MAX_TERMLEN](40 * 15, 50)
        shape = expects.shape
        expects_, jobs_ = expects.view([shape[0], -1]), jobs.view([shape[0], -1])

        # embeddings: [batch_size, MAX_PROFILELEN * MAX_TERMLEN, EMBED_DIM]
        expects_wordembed = self.Word_Embeds(expects_).float()
        jobs_wordembed = self.Word_Embeds(jobs_).float()

        # permute for conv1d
        # embeddings: [batch_size, EMBED_DIM, MAX_PROFILELEN * MAX_TERMLEN]
        expects_wordembed_ = expects_wordembed.permute(0, 2, 1)
        jobs_wordembed_ = jobs_wordembed.permute(0, 2, 1)

        # [batch_size, EMBED_DIM, x]
        expect_convs_out = self.Expect_ConvNet(expects_wordembed_)
        job_convs_out = self.Job_ConvNet(jobs_wordembed_)

        # [batch_size, EMBED_DIM, x] -> [batch_size, EMBED_DIM, 1]
        expect_len, job_len = expect_convs_out.shape[-1], job_convs_out.shape[-1]
        expect_final_out = torch.nn.AvgPool1d(kernel_size=expect_len)(expect_convs_out).squeeze(-1)
        job_final_out = torch.nn.MaxPool1d(kernel_size=job_len)(job_convs_out).squeeze(-1)

        return self.Match_MLP(torch.cat([expect_final_out, job_final_out], dim=-1)).squeeze(-1)