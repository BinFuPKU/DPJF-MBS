
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src.Preprocess.profile_word2vec import EMBED_DIM

USER_EMBED_DIM = 100

class NCF(torch.nn.Module):
    def __init__(self, n_expects, n_jobs):
        super(NCF, self).__init__()
        self.n_expects = n_expects
        self.n_jobs = n_jobs

        # embeddings: learnable
        self.expect_embeddings = torch.nn.Embedding(n_expects, USER_EMBED_DIM)
        self.expect_embeddings.weight.requires_grad = True
        self.job_embeddings = torch.nn.Embedding(n_jobs, USER_EMBED_DIM)
        self.job_embeddings.weight.requires_grad = True


        # MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM),
        )

        # MLP
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Sigmoid(),
            torch.nn.Linear(USER_EMBED_DIM, 1),
            torch.nn.Sigmoid()
        )

    def predict(self, ids1, ids2, e2j_flag=True):
        expect_embed, job_embed = None, None
        if e2j_flag:
            expect_embed = self.expect_embeddings(ids1)
            job_embed = self.job_embeddings(ids2)
        else:
            job_embed = self.job_embeddings(ids1)
            expect_embed = self.expect_embeddings(ids2)

        gmf_features = expect_embed * job_embed
        mlp_features = self.mlp(torch.cat([expect_embed, job_embed], dim=-1))
        return self.final_mlp(torch.cat([gmf_features, mlp_features], dim=-1)).squeeze(-1)


    def forward(self, ids, ids_pair, e2j_flag=True):
        scores_pos, scores_neg = None, None
        if e2j_flag:
            scores_pos = self.predict(ids, ids_pair[:, 0], True)
            scores_neg = self.predict(ids, ids_pair[:, 1], True)
        else:
            scores_pos = self.predict(ids, ids_pair[:, 0], False)
            scores_neg = self.predict(ids, ids_pair[:, 1], False)
        return torch.sigmoid(scores_pos - scores_neg)
