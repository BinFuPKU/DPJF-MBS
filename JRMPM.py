
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_EMBED_DIM = 32
USER_EMBED_DIM = 16

MAX_PROFILELEN = 10
MAX_TERMLEN = 30


class JRMPM(torch.nn.Module):
    def __init__(self, word_embeddings):
        super(JRMPM, self).__init__()

        # profile: word embeddings for look_up
        # embedding_matrix = [[0...0], [...], ...[]]
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.word_embeddings.weight.requires_grad = False

        # BI-GRU: int(USER_EMBED_DIM/2) * 2 = USER_EMBED_DIM
        self.expect_words_gru = torch.nn.GRU(input_size=WORD_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.job_words_gru = torch.nn.GRU(input_size=WORD_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=True)

        # GRU: USER_EMBED_DIM
        self.expect_sent_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=USER_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False)
        self.job_sent_gru = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=USER_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False)

        # memory profiling
        self.expect_momery = torch.nn.Embedding(MAX_PROFILELEN, USER_EMBED_DIM)
        self.expect_momery.weight.requires_grad = True
        self.job_momery = torch.nn.Embedding(MAX_PROFILELEN, USER_EMBED_DIM)
        self.job_momery.weight.requires_grad = True


        # update pi: beta, gamma
        self.expect_update_pi = torch.nn.Sequential(
            torch.nn.Linear(MAX_PROFILELEN, MAX_PROFILELEN, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-2)
        )
        self.job_update_pi = torch.nn.Sequential(
            torch.nn.Linear( MAX_PROFILELEN,  MAX_PROFILELEN, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-2)
        )

        # update g:
        self.expect_g_update = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.job_g_update = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )

        # read phi: alpha
        self.expect_read_phi = torch.nn.Sequential(
            torch.nn.Linear(MAX_PROFILELEN, MAX_PROFILELEN, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-2)
        )
        self.job_read_phi = torch.nn.Sequential(
            torch.nn.Linear(MAX_PROFILELEN, MAX_PROFILELEN, bias=False),
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=-2)
        )
        # read g:
        self.expect_g_read = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.job_g_read = torch.nn.Sequential(
            torch.nn.Linear(3 * USER_EMBED_DIM, 1, bias=False),
            torch.nn.Sigmoid()
        )

        # match
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * MAX_PROFILELEN * USER_EMBED_DIM,  MAX_PROFILELEN * USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(MAX_PROFILELEN * USER_EMBED_DIM, 1),
            torch.nn.Sigmoid()
        )



    # profiles: [batch_size, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50), word idx
    def __words_BiGRU__(self, profiles, isexpect=True):
        # word level:
        shape = profiles.shape # [132, 20, 50]
        profiles_ = profiles.contiguous().view([-1, shape[-1]])
        # sort expects_sample_: large to small
        # sorted [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        lens = (profiles_ > 0).sum(dim=-1)
        lens_sort, ind_sort = lens.sort(dim=0, descending=True)
        profiles_sort = profiles_[ind_sort]
        # embeddings: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        profile_embed = self.word_embeddings(profiles_sort).float()
        profile_pack = pack_padded_sequence(profile_embed, lens_sort, batch_first=True)
        if isexpect:
            _, sent_hidden = self.expect_words_gru(profile_pack)
        else:
            _, sent_hidden = self.job_words_gru(profile_pack)
        # [2640, 2, 50]
        sent_hidden = sent_hidden.permute(1, 0, 2).contiguous().view([-1, USER_EMBED_DIM])
        sent_hidden = sent_hidden[ind_sort].view([shape[0], shape[1], -1])
        # [132, 20, 100]
        return sent_hidden

    # sents: [batch_size, MAX_PROFILELEN, dim]
    def __sents_GRU__(self, sent_hidden, isexpect=True):
        if isexpect:
            out, _ = self.expect_sent_gru(sent_hidden)
        else:
            out, _ = self.job_sent_gru(sent_hidden)
        return out

    def profile2sent(self, profiles, isexpect):
        return self.__sents_GRU__(self.__words_BiGRU__(profiles, isexpect), isexpect)

    # memory:  [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
    # a_sents: [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
    # b_sents: [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
    # col_mask: [batch]
    def update(self, memory, a_sents, b_sents, col_mask, isexpect=True):
        if isexpect:
            # [batch, n, n*]
            beta = self.expect_update_pi(torch.bmm(memory, a_sents.permute(0, 2, 1)))
            gamma = self.expect_update_pi(torch.bmm(memory, b_sents.permute(0, 2, 1)))
        else:
            beta = self.job_update_pi(torch.bmm(memory, a_sents.permute(0, 2, 1)))
            gamma = self.job_update_pi(torch.bmm(memory, b_sents.permute(0, 2, 1)))

        # [batch, n, n*] * [batch, n, dim] = [batch, n, dim]
        i_update = torch.bmm(beta, a_sents) + torch.bmm(gamma, b_sents)
        # [batch, n, dim]
        if isexpect:
            g_update = self.expect_g_update(torch.cat([memory, i_update, memory * i_update], dim=-1))
        else:
            g_update = self.job_g_update(torch.cat([memory, i_update, memory * i_update], dim=-1))
        # m_{k+1}
        # [batch, MAX_PROFILELEN, USER_EMBED_DIM]
        memory_update = g_update * memory + (1-g_update) * memory

        # mask
        shape = memory_update.shape
        memory_update_mask = (torch.unsqueeze(col_mask, 1) * memory_update.view([shape[0], -1])).view(shape)
        memory_noupdate_mask = (torch.unsqueeze(1.-col_mask, 1) * memory.contiguous().view([shape[0], -1])).view(shape)

        return memory_update_mask + memory_noupdate_mask

    # memory: [batch, n, dim] [1, 20, 100]
    # hidden_last: [batch, n, dim] [1, 20, 100]
    # a_sents: [batch, n, dim] [1, 20, 100]
    def read(self, memory, hidden_last, a_sents, isexpect=True):
        # [batch, n, n*]
        if isexpect:
            alpha = self.expect_read_phi(torch.bmm(memory, (hidden_last * a_sents).permute(0, 2, 1)))
        else:
            alpha = self.job_read_phi(torch.bmm(memory, (hidden_last * a_sents).permute(0, 2, 1)))

        # [batch, n, n*] * [batch, n, dim] = [batch, n, dim]
        i_read = torch.bmm(alpha, memory)
        # [batch, n, dim],
        if isexpect:
            g_read = self.expect_g_read(torch.cat([a_sents, i_read, a_sents * i_read], dim=-1))
        else:
            g_read = self.job_g_read(torch.cat([a_sents, i_read, a_sents * i_read], dim=-1))

        # [batch, n, dim]
        hidden = g_read * i_read + (1 - g_read) * hidden_last
        return hidden

    # a_profiles: [batch, sent, word] [1, 20, 50], tensor
    # b_profiless: [batch, max_seq_len, sent, word] [1, 3, 20, 50], tensor
    # b_seq_lens: [], list
    def process_seq(self, a_profiles, b_seq_profiless, b_seq_lens, isexpect=True):

        # [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
        batch_a_sents = self.profile2sent(a_profiles, isexpect)
        # [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
        batch_memory = batch_hidden = self.profile2sent(a_profiles, not isexpect)

        for i in range(max(b_seq_lens)):
            # [1,0,... ]
            col_mask = torch.from_numpy((np.array(b_seq_lens)-i>0)+0.).float().to(DEVICE)
            # [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
            batch_b_sents = self.profile2sent(b_seq_profiless[:, i, :, :], not isexpect)

            # batch_memory:  [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
            # batch_a_sents: [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
            # batch_b_sents: [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100]
            # batch_memory:  [batch, MAX_PROFILELEN, USER_EMBED_DIM] [1, 20, 100])
            batch_memory = self.update(batch_memory, batch_a_sents,
                                  batch_b_sents, col_mask, isexpect)
            batch_hidden = self.read(batch_memory, batch_hidden, batch_a_sents, isexpect)

        return batch_hidden


    # [100, 20, 100] [100, 20, 100]
    def predict(self, expect_hidden, job_hidden):

        expect_hidden_ = expect_hidden.reshape([expect_hidden.shape[0], -1])
        job_hidden_ = job_hidden.reshape([job_hidden.shape[0], -1])

        return self.MLP(torch.cat([expect_hidden_, job_hidden_], -1))





