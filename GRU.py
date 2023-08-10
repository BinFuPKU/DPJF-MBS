import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WORD_EMBED_DIM = 32
USER_EMBED_DIM = 16

WORD_EMBED_DIM = 32
MAX_PROFILELEN = 10

class GRUNetwork(torch.nn.Module):
    def __init__(self, word_embeddings, model_name):
        super(GRUNetwork, self).__init__()

        # profile: word embeddings for look_up
        # embedding_matrix = [[0...0], [...], ...[]]
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.word_embeddings.weight.requires_grad = False

        # bilstm: int(USER_EMBED_DIM/2) * 2 = USER_EMBED_DIM
        self.words_gru = torch.nn.GRU(input_size=WORD_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=True, dropout=0.)
        self.term_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=True, dropout=0.)

        # gru for behavior sequence
        self.behavior_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=USER_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False, dropout=0.)

        self.match_MLP = torch.nn.Sequential(
            torch.nn.Linear(4 * USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(USER_EMBED_DIM, 1),
            torch.nn.Sigmoid())

    # profiles: [batch_size, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50), word idx
    # return: [batch_size, USER_EMBED_DIM]
    def get_profile_embeddings(self, profiles):
        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        shape = profiles.shape
        profiles_ = profiles.contiguous().view([-1, shape[-1]])
        # sort expects_sample_: large to small
        # sorted [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        lens = (profiles_ > 0).sum(dim=-1)
        lens_sort, ind_sort = lens.sort(dim=0, descending=True)
        profiles_sort = profiles_[ind_sort]
        # embeddings: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        profile_embed = self.word_embeddings(profiles_sort).float()
        # compress: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        profile_pack = pack_padded_sequence(profile_embed, lens_sort, batch_first=True)
        # [batch_size * MAX_PROFILELEN, hidden_dim]
        _, term_state = self.words_gru(profile_pack)
        # print(shape, term_state.shape)
        term_state = term_state.view([shape[0], shape[1], -1])
        # print(term_state.shape)

        # [batch_size * MAX_PROFILELEN, hidden_dim]
        _, profile_state = self.term_gru(term_state)
        return profile_state.view([shape[0], -1])

    # b_profiless: [batch, max_seq_len, sent, word] [1, 3, 20, 50], gpu tensor
    # b_seq_lens: [], list
    # b_seq_labels: [[], ...], 0,1,2,3
    # return: [batch, USER_EMBED_DIM]
    def process_seq(self, b_seq_profiless, b_seq_lens):
        # [batch, max_seq_len, sent, word] ->[batch_size, MAX_PROFILELEN, MAX_TERMLEN]
        shape = b_seq_profiless.shape
        b_seq_profiless_ = b_seq_profiless.view([-1, shape[-2], shape[-1]])
        # (batch, hidden_size) -> [batch, max_seq_len, embed]
        b_seq_embeds = self.get_profile_embeddings(b_seq_profiless_).view([shape[0], shape[1], -1])

        # sort expects_sample_: large to small
        # sorted [batch, max_seq_len, embed]
        lens = torch.from_numpy(np.array(b_seq_lens)).to(DEVICE)
        lens_sort, ind_sort = lens.sort(dim=0, descending=True)
        b_seq_embeds_ = b_seq_embeds[ind_sort, :, :]

        # compress: [batch, max_seq_len, embed] -> [batch, embed]
        seq_pack = pack_padded_sequence(b_seq_embeds_, lens_sort, batch_first=True)
        seq_output, seq_state = self.behavior_gru(seq_pack)

        return seq_state.squeeze(0)[ind_sort]


    # a_seq_profiless: [batch, max_seq_len, sent, word] [1, 3, 20, 50], gpu tensor
    # a_seq_lens: [], list
    # a_profiles: [batch, sent, word]
    def predict(self, a_seq_profiless, a_seq_lens, a_profiles, b_profiles):
        # read profile
        a_embeddings = self.get_profile_embeddings(a_profiles)
        b_embeddings = self.get_profile_embeddings(b_profiles)
        # state
        a_state = self.process_seq(a_seq_profiless, a_seq_lens)

        # interact
        interact = a_state * b_embeddings
        feature = torch.cat([a_state, interact, b_embeddings, a_embeddings], dim=-1)

        return self.match_MLP(feature).squeeze(-1)