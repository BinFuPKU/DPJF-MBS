import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WORD_EMBED_DIM = 32
USER_EMBED_DIM = 32

MAX_PROFILELEN = 20
MAX_TERMLEN = 50

MEMORY_SLOT = 5

class Model(torch.nn.Module):
    def __init__(self, word_embeddings, behavior_typenum):
        super(Model, self).__init__()
        self.behavior_typenum = behavior_typenum

        print('MEMORY_SLOT=', MEMORY_SLOT)

        # profile: word embeddings for look_up
        # embedding_matrix = [[0...0], [...], ...[]]
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.word_embeddings.weight.requires_grad = False

        # bilstm: int(USER_EMBED_DIM/2) * 2 = USER_EMBED_DIM
        self.words_gru = torch.nn.GRU(input_size=WORD_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=True, dropout=0.)
        # attention layer
        self.words_attention_layer = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Tanh(), # LeakyReLU
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),
        )
        # attention layer
        self.terms_attention_layer = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(USER_EMBED_DIM, 1, bias=False),
        )

        self.KEY_MATs = []
        for i in range(self.behavior_typenum):
            # [MEMORY_SLOT, USER_EMBED_DIM]
            self.KEY_MATs.append(Variable(torch.randn((MEMORY_SLOT, USER_EMBED_DIM)), requires_grad = True).float().to(DEVICE))

        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Sigmoid(),
        )

        self.e2j_MTL_MLPs = []
        for i in range(self.behavior_typenum-1):
            self.e2j_MTL_MLPs.append(torch.nn.Sequential(
                                        torch.nn.Linear( 4 *USER_EMBED_DIM,  USER_EMBED_DIM),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear( USER_EMBED_DIM, 1),
                                        torch.nn.Sigmoid()).to(DEVICE))
        self.j2e_MTL_MLPs = []
        for i in range(self.behavior_typenum-1):
            self.j2e_MTL_MLPs.append(torch.nn.Sequential(
                                        torch.nn.Linear( 4 *USER_EMBED_DIM,  USER_EMBED_DIM),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear( USER_EMBED_DIM, 1),
                                        torch.nn.Sigmoid()).to(DEVICE))
        self.match_MLP = torch.nn.Sequential(
                                        torch.nn.Linear( 4 *USER_EMBED_DIM,  USER_EMBED_DIM),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear( USER_EMBED_DIM, 1),
                                        torch.nn.Sigmoid()).to(DEVICE)
        self.e2j_MTL_MLPs.append(self.match_MLP)
        self.j2e_MTL_MLPs.append(self.match_MLP)

    # profiles: [batch_size, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50), word idx
    # return: [batch_size, USER_EMBED_DIM]
    def get_profile_embeddings(self, profiles, isexpect):
        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        shape = profiles.shape
        profiles_ = profiles.view([-1, shape[-1]])

        # sort expects_sample_: large to small
        # sorted [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        lens = (profiles_ > 0).sum(dim=-1)
        lens_sort, ind_sort = lens.sort(dim=0, descending=True)
        profiles_sort = profiles_[ind_sort]

        # embeddings: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        profile_embed = self.word_embeddings(profiles_sort).float()
        # compress: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        profile_pack = pack_padded_sequence(profile_embed, lens_sort, batch_first=True)

        words_output, _ = self.words_gru(profile_pack)

        # output: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2]
        words_output_, _ = torch.nn.utils.rnn.pad_packed_sequence(words_output, batch_first=True)

        # attention: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2]
        words_attention = F.softmax(self.words_attention_layer(words_output_), dim=-2)

        # [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2] ->
        # [batch_size * MAX_PROFILELEN, words_lstm_hidden_dims * 2]
        terms_ = (words_attention * words_output_).sum(-2)\
            .view([shape[0], MAX_PROFILELEN, USER_EMBED_DIM])

        # (batch, 1, USER_EMBED_DIM) * (batch_size, MAX_PROFILELEN, hidden_size) ->
        # (batch, MAX_PROFILELEN, USER_EMBED_DIM) -> softmax(batch, MAX_PROFILELEN, 1)
        attention = torch.softmax(self.terms_attention_layer(terms_), dim=-2)

        # (batch, MAX_PROFILELEN, 1) * (batch_size, MAX_PROFILELEN, hidden_size) ->
        # (batch, hidden_size)
        profile_embeddings = torch.sum(attention * terms_, dim=1)

        return profile_embeddings

    # memory: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    # b_embedding： [batch, USER_EMBED_DIM]
    # col_label: []
    # return: [batch, USER_EMBED_DIM]
    def read(self, memory, b_embedding, col_label):
        preference = None
        for i in range(self.behavior_typenum):
            # [batch, USER_EMBED_DIM] mm [MEMORY_SLOT, USER_EMBED_DIM].T -> [batch, MEMORY_SLOT]
            attention = torch.softmax(torch.matmul(b_embedding, torch.transpose(self.KEY_MATs[i], dim0=1, dim1=0)), dim=-1)
            # [batch, 1, MEMORY_SLOT] bmm [batch, MEMORY_SLOT, USER_EMBED_DIM] -> [batch, 1, USER_EMBED_DIM]
            # -> [batch, USER_EMBED_DIM]
            embedding = torch.squeeze(torch.bmm(torch.unsqueeze(attention, 1), memory), 1)

            mask_label = torch.from_numpy(np.array([label == i for label in col_label])).float().to(DEVICE)
            if i==0:
                preference = embedding * torch.unsqueeze(mask_label, -1)
            else:
                preference += embedding * torch.unsqueeze(mask_label, -1)
        return preference

    # memory: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    # b_embedding： [batch, USER_EMBED_DIM]
    # col_label: []
    # col_mask: tensor([])
    # return: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    def update(self, memory, b_embedding, col_label, col_mask, isaexpect):

        for i in range(self.behavior_typenum):
            # [batch, USER_EMBED_DIM] mm [MEMORY_SLOT, USER_EMBED_DIM].T -> [batch, MEMORY_SLOT]
            attention = torch.softmax(torch.matmul(b_embedding, torch.transpose(self.KEY_MATs[i], dim0=1, dim1=0)), dim=-1)
            # [batch, MEMORY_SLOT, 1] * [batch, 1, USER_EMBED_DIM] -> [batch, MEMORY_SLOT, USER_EMBED_DIM
            update = torch.unsqueeze(attention, -1) * torch.unsqueeze(self.update_mlp(b_embedding), 1)
            # [batch, MEMORY_SLOT, USER_EMBED_DIM] * [batch, MEMORY_SLOT, 1]
            new_memory = memory * (1 - torch.unsqueeze(attention.to(DEVICE), -1)) + update
            # [batch]
            mask_label = torch.from_numpy(np.array([label==i for label in col_label])).float().to(DEVICE)
            mask = mask_label * col_mask
            memory = memory * (1-mask.view([len(col_label),1,1])) + new_memory * mask.view([len(col_label),1,1])
        return memory

    # b_profiless: [batch, max_seq_len, sent, word] [1, 3, 20, 50], gpu tensor
    # b_seq_lens: [], list
    # b_seq_labels: [[], ...], 0,1,2,3
    # return: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    def process_seq(self, b_seq_profiless, b_seq_lens, b_seq_tlabels, isaexpect=True):
        # memory
        batch_memory = torch.from_numpy(np.zeros((len(b_seq_lens), MEMORY_SLOT, USER_EMBED_DIM))).float().to(DEVICE)
        for i in range(max(b_seq_lens)):
            # [1,0,... ]
            col_mask = torch.from_numpy((np.array(b_seq_lens)-i>0)+0.).float().to(DEVICE)
            col_label = [bstls[i] if len(bstls)>i+1 else 0 for bstls in b_seq_tlabels]
            # [batch, USER_EMBED_DIM]
            batch_b_embedding = self.get_profile_embeddings(b_seq_profiless[:, i, :, :].contiguous(), not isaexpect)
            batch_memory = self.update(batch_memory, batch_b_embedding, col_label, col_mask, isaexpect)
        return batch_memory

    # expect_memory: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    # job_profiles: [batch, sent, word]
    # job_memory: [batch, MEMORY_SLOT, USER_EMBED_DIM]
    # expect_profiles: [batch, sent, word]
    # label: [], list
    def predict(self, expect_memory, job_profiles, job_memory, expect_profiles, tlabel, ise2j):
        # read expect
        job_embedding = self.get_profile_embeddings(job_profiles, False)
        expect_preference = self.read(expect_memory, job_embedding, tlabel)
        # read job
        expect_embedding = self.get_profile_embeddings(expect_profiles, True)
        job_preference = self.read(job_memory, expect_embedding, tlabel)

        # interact
        e2j_interact = expect_preference * job_embedding
        j2e_interact = job_preference * expect_embedding

        feature = torch.cat([e2j_interact, job_embedding, j2e_interact, expect_embedding], dim=-1)

        if ise2j:
            e2j_scores = None
            for i in range(self.behavior_typenum):
                mask_label = torch.from_numpy(np.array([label == i for label in tlabel])).float().to(DEVICE)
                score = torch.squeeze(self.e2j_MTL_MLPs[i](feature), -1)
                if i == 0:
                    e2j_scores = score * mask_label
                else:
                    e2j_scores += score * mask_label
            return e2j_scores
        else:
            j2e_scores = None
            for i in range(self.behavior_typenum):
                mask_label = torch.from_numpy(np.array([label == i for label in tlabel])).float().to(DEVICE)
                score = torch.squeeze(self.j2e_MTL_MLPs[i](feature), -1)
                if i==0:
                    j2e_scores = score * mask_label
                else:
                    j2e_scores += score * mask_label
            return j2e_scores

