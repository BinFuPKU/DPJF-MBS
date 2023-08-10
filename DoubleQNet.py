import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from copy import deepcopy

from src.Model.Profile2Vec import Profile2Vec

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WORD_EMBED_DIM = 32
USER_EMBED_DIM = 16

STATE_EMBED_DIM = 8

class StateRep(torch.nn.Module):
    def __init__(self, bprofile2vec, a_model_name):
        super(StateRep, self).__init__()
        self.a_model_name, self.bprofile2vec = a_model_name, bprofile2vec

        self.det_GRU = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=False)
        self.add_GRU = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=False)
        self.add__GRU = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=False)
        self.ac_GRU = torch.nn.GRU(input_size=USER_EMBED_DIM, hidden_size=int(USER_EMBED_DIM/2),
                                    num_layers=1, batch_first=True, bidirectional=False)

    # a_bseq_profiless: [batch, max_seq_len, sent, word]
    # a_bseq_lens: []
    def process_seq_full(self, gru, a_bseq_profiless, a_bseq_lens):
        # [batch, max_seq_len, sent, word] -> [batch_size, MAX_PROFILELEN, MAX_TERMLEN], [112, 6, 10, 30]
        shape = a_bseq_profiless.shape
        # print(1, shape): 全部是实的！
        a_bseq_profiless_ = a_bseq_profiless.contiguous().view([-1, shape[-2], shape[-1]])
        # [112, 6, 4], word_idx
        a_bseq_embeds = self.bprofile2vec(a_bseq_profiless_).view([shape[0], shape[1], -1])
        # sort for pack
        lens = a_bseq_lens
        lens_sort, ind_sort = lens.sort(dim=0, descending=True)
        a_bseq_embeds_ = a_bseq_embeds[ind_sort, :, :]
        # pack
        seq_pack = pack_padded_sequence(a_bseq_embeds_, lens_sort, batch_first=True)
        # gru
        seq_output, seq_state = gru(seq_pack)
        # rollback sort, [1, 112, 4], [batch, USER_EMBED_DIM]
        return seq_state.squeeze(0)[ind_sort, :]

    # a_bseq_profiless: [batch, max_seq_len, sent, word]
    # a_bseq_lens: []
    def process_seq(self, gru, a_bseq_profiless, a_bseq_lens):
        all_bool = torch.gt(a_bseq_lens, 0)
        idx_true = [i for i, x in enumerate(all_bool) if x == True]
        if len(idx_true) == all_bool.shape[0]:
            return self.process_seq_full(gru, a_bseq_profiless, a_bseq_lens)
        elif len(idx_true) == 0:
            return torch.from_numpy(np.zeros((all_bool.shape[0], STATE_EMBED_DIM))).float().to(DEVICE)
        else:
            a_bseq_profiless_ = a_bseq_profiless[idx_true]
            a_bseq_lens_ = a_bseq_lens[idx_true]
            seq_state_ = self.process_seq_full(gru, a_bseq_profiless_, a_bseq_lens_)
            seq_state = torch.from_numpy(np.zeros((all_bool.shape[0], STATE_EMBED_DIM))).float().to(DEVICE)
            for ind, idx in enumerate(idx_true):
                seq_state[idx, :] = seq_state_[ind, :]
            return seq_state

    # input: [(40, 4, 5, 10, 30), (40, 4)]
    # output: [batch, USER_EMBED_DIM]
    def forward(self, a_bseq_profiless, a_bseq_lenss):
        det_state = self.process_seq(self.det_GRU, a_bseq_profiless[:,0,:,:,:], a_bseq_lenss[:,0])
        add_state = self.process_seq(self.add_GRU, a_bseq_profiless[:,1,:,:,:], a_bseq_lenss[:,1])
        add__state = self.process_seq(self.add__GRU, a_bseq_profiless[:,2,:,:,:], a_bseq_lenss[:,2])
        ac_state = self.process_seq(self.ac_GRU, a_bseq_profiless[:,3,:,:,:], a_bseq_lenss[:,3])
        return torch.cat([det_state, add_state, add__state, ac_state], dim=-1)

    def parameters(self):
        return list(self.det_GRU.parameters()) + list(self.add_GRU.parameters()) + \
                 list(self.add__GRU.parameters()) + list(self.ac_GRU.parameters())

class UserAgent(torch.nn.Module):
    def __init__(self, a_profile2vec, b_profile2vec, a_staterep, a_model_name):
        super(UserAgent, self).__init__()
        self.a_model_name = a_model_name
        self.a_profile2vec, self.b_profile2vec = a_profile2vec, b_profile2vec
        self.a_staterep = a_staterep

    # [(40, 10, 30), (40, 4, 5, 10, 30), (40, 4)]
    def forward(self, a_profiles, a_bseq_profiless, a_bseq_lenss): # one
        a_embeddings = self.a_profile2vec(a_profiles)
        a_state = self.a_staterep(a_bseq_profiless, a_bseq_lenss)
        # return torch.cat([a_embeddings, b_embeddings, a_state], dim=-1)
        return a_embeddings, a_state

    def fold_dim2(self, data):
        shape = [x for x in data.shape]
        shape_ = deepcopy(shape)
        shape_.pop(1)
        shape_[0] = -1
        return shape[1], data.view(shape_)
    def unfold_dim2(self, data, dim2_len):
        shape = [x for x in data.shape]
        shape_ = deepcopy(shape)
        shape_.insert(1, dim2_len)
        shape_[0]=-1
        return data.view(shape_)
    # [(40, 4-, 10, 30), (40, 4-, 4, 5, 10, 30), (40, 4-, 4)]
    def forwards(self, a_action_bprofiles, a_action_state_bprofiles, a_action_state_lens): # many
        len1, a_action_bprofiles_ = self.fold_dim2(a_action_bprofiles)
        a_embeddings = self.a_profile2vec(a_action_bprofiles_)
        a_embeddings_ = self.unfold_dim2(a_embeddings, len1)

        if a_action_state_bprofiles is not None:
            len2, a_action_state_bprofiles_ = self.fold_dim2(a_action_state_bprofiles)
            _, a_action_state_lens_ = self.fold_dim2(a_action_state_lens)
            a_states = self.a_staterep(a_action_state_bprofiles_, a_action_state_lens_)
            a_states_ = self.unfold_dim2(a_states, len2)
        else:
            a_states_ = None
        return a_embeddings_, a_states_


class QNet(torch.nn.Module):
    def __init__(self, person_profile2vec, job_profile2vec, person_staterep, job_staterep, model_name):
        super(QNet, self).__init__()
        self.model_name = model_name
        self.person_agent = UserAgent(person_profile2vec, job_profile2vec, person_staterep, 'person_agent')
        self.job_agent = UserAgent(job_profile2vec, person_profile2vec, job_staterep, 'job_agent')

        # 预测函数：person端和job端共享

        self.match_MLP = torch.nn.Sequential(
            torch.nn.Linear(6 * USER_EMBED_DIM, 3 * USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(3 * USER_EMBED_DIM, USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(USER_EMBED_DIM, 1),
            # torch.nn.ReLU()
            # torch.nn.Sigmoid()
        )
        self.person_like_MLP = torch.nn.Sequential(
            torch.nn.Linear(4 * USER_EMBED_DIM, 2 * USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * USER_EMBED_DIM, 1),
            # torch.nn.ReLU()
            # torch.nn.Sigmoid()
        )
        self.job_like_MLP = torch.nn.Sequential(
            torch.nn.Linear(4 * USER_EMBED_DIM, 2 * USER_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * USER_EMBED_DIM, 1),
            # torch.nn.ReLU()
            # torch.nn.Sigmoid()
        )

    def parameters(self):
        return list(self.match_MLP.parameters()) + list(self.person_like_MLP.parameters()) + list(self.job_like_MLP.parameters())

    def predict_like_score(self, a_profiles, b_profiles, a_now_seq_profiless, a_now_seq_lens, is_e2j):
        if is_e2j:
            e_vector, e_state = self.person_agent.forward(a_profiles, a_now_seq_profiless, a_now_seq_lens)
            j_vector = self.person_agent.b_profile2vec(b_profiles)
            return self.person_like_MLP(torch.cat([e_vector, e_state, j_vector], dim=-1)).squeeze(-1)
        else:
            j_vector, j_state = self.job_agent.forward(a_profiles, a_now_seq_profiless, a_now_seq_lens)
            e_vector = self.job_agent.b_profile2vec(b_profiles)
            return self.job_like_MLP(torch.cat([j_vector, j_state, e_vector], dim=-1)).squeeze(-1)

    def predict_like_max_score(self, a_profiles, a_action_bprofiles,
                                       a_next_state_seq_profiless, a_next_seq_lens, is_e2j):
        if is_e2j:#e-to-js
            e_vector, e_state = self.person_agent.forward(a_profiles, a_next_state_seq_profiless, a_next_seq_lens)
            j_vectors, _ = self.job_agent.forwards(a_action_bprofiles, None, None)
            scores = self.person_like_MLP(torch.cat([self._expand_like_(e_vector, j_vectors), self._expand_like_(e_state, j_vectors),
                                               j_vectors], dim=-1)).squeeze(-1)
        else:# es-to-j
            j_vector, j_state = self.job_agent.forward(a_profiles, a_next_state_seq_profiless, a_next_seq_lens)
            e_vectors, _ = self.person_agent.forwards(a_action_bprofiles, None, None)
            scores = self.person_like_MLP(torch.cat([self._expand_like_(j_vector, e_vectors), self._expand_like_(j_state, e_vectors),
                                               e_vectors], dim=-1)).squeeze(-1)
        indices = torch.max(scores, dim=1)[1]
        next_max_b_profiles = torch.cat([torch.unsqueeze(a_action_bprofiles[rind, cind, :, :], 0) for rind, cind in enumerate(indices)], 0)
        return next_max_b_profiles

    def predict_match_score(self, e_profiles, j_profiles,
                                e_now_seq_profiless, e_now_seq_lens, j_now_seq_profiless, j_now_seq_lens): # a2b===e2j
        e_vector, e_state = self.person_agent.forward(e_profiles, e_now_seq_profiless, e_now_seq_lens)
        j_vector, j_state = self.job_agent.forward(j_profiles, j_now_seq_profiless, j_now_seq_lens)
        return self.match_MLP(torch.cat([e_vector, e_state, j_vector, j_state], dim=-1)).squeeze(-1)

    # 1:n
    # [(40, 4-, 10, 30), (40, 4-, 4, 5, 10, 30), (40, 4-, 4)]
    def predict_match_max_score(self, a_profiles, a_action_bprofiles,
                                       a_next_state_seq_profiless, a_next_seq_lens,
                                       a_action_state_bprofiles, a_action_state_lens, is_e2j):
        if is_e2j:# e-to-js
            e_vector, e_state = self.person_agent.forward(a_profiles, a_next_state_seq_profiless, a_next_seq_lens)
            j_vectors, j_states = self.job_agent.forwards(a_action_bprofiles, a_action_state_bprofiles, a_action_state_lens)
            scores = self.match_MLP(torch.cat([self._expand_like_(e_vector, j_vectors), self._expand_like_(e_state, j_states),
                                               j_vectors, j_states], dim=-1)).squeeze(-1)

        else:# es-to-j
            j_vector, j_state = self.job_agent.forward(a_profiles, a_next_state_seq_profiless, a_next_seq_lens)
            e_vectors, e_states = self.person_agent.forwards(a_action_bprofiles, a_action_state_bprofiles, a_action_state_lens)
            scores = self.match_MLP(torch.cat([e_vectors, e_states, self._expand_like_(j_vector, e_vectors),
                                               self._expand_like_(j_state, e_states)], dim=-1)).squeeze(-1)
        indices = torch.max(scores, dim=1)[1]
        next_max_b_profiles = torch.cat([torch.unsqueeze(a_action_bprofiles[rind,cind,:,:], 0) for rind, cind in enumerate(indices)], 0)
        next_max_b_state_seq_profiles = torch.cat([torch.unsqueeze(a_action_state_bprofiles[rind,cind,:,:,:,:], 0) for rind, cind in enumerate(indices)], 0)
        next_max_b_state_lens = torch.cat([torch.unsqueeze(a_action_state_lens[rind,cind,:], 0) for rind, cind in enumerate(indices)], 0)

        return next_max_b_profiles, next_max_b_state_seq_profiles, next_max_b_state_lens

    def _expand_like_(self, a, b):
        a = torch.unsqueeze(a, 1)
        shape = a.shape
        return a.expand(shape[0], b.shape[1], shape[2])

class DoubleQNetMain(torch.nn.Module):
    def __init__(self, word_embeddings):
        super(DoubleQNetMain, self).__init__()
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.word_embeddings.weight.requires_grad = False

        self.person_profile2vec = Profile2Vec(self.word_embeddings, 'person_profile2vec')
        self.job_profile2vec = Profile2Vec(self.word_embeddings, 'job_profile2vec')

        self.person_staterep = StateRep(self.job_profile2vec, 'person_staterep')
        self.job_staterep = StateRep(self.person_profile2vec, 'job_staterep')

        # person_profile2vec, job_profile2vec, person_staterep, job_staterep, model_name
        self.main_policy = QNet(self.person_profile2vec, self.job_profile2vec, self.person_staterep, self.job_staterep, 'main_policy')
        self.target_policy = QNet(self.person_profile2vec, self.job_profile2vec, self.person_staterep, self.job_staterep, 'target_policy')

    def like_forward(self, a_profiles, a_now_seq_profiless, a_now_seq_lens, b_profiles,
                            a_next_state_seq_profiless, a_next_seq_lens,  # a=e/j
                            a_action_bprofiles, is_e2j):
        # current
        current_reward = self.main_policy.predict_like_score(a_profiles,b_profiles, a_now_seq_profiless, a_now_seq_lens, is_e2j)
        # next step
        b_profiles_maxscore = self.target_policy.predict_like_max_score(a_profiles, a_action_bprofiles,
                                       a_next_state_seq_profiless, a_next_seq_lens, is_e2j)
        next_reward = self.main_policy.predict_like_score(a_profiles,b_profiles_maxscore, a_now_seq_profiless, a_now_seq_lens, is_e2j)
        return current_reward, next_reward

    def match_forward(self, e_profiles, e_now_seq_profiless, e_now_seq_lens,
                      j_profiles, j_now_seq_profiless, j_now_seq_lens,
                      a_next_state_seq_profiless, a_next_seq_lens,  # a=e/j
                      a_action_bprofiles, a_action_state_bprofiles, a_action_state_lens,
                        is_e2j):

        # current: 固定e-j match
        current_reward = self.main_policy.predict_match_score(e_profiles, j_profiles,
                                e_now_seq_profiless, e_now_seq_lens, j_now_seq_profiless, j_now_seq_lens)
        # next step
        if is_e2j: # a=e, b=j
            next_max_j_profiles, next_max_j_state_seq_profiles, next_max_j_state_lens = \
                self.target_policy.predict_match_max_score(e_profiles, a_action_bprofiles,
                                                     a_next_state_seq_profiless, a_next_seq_lens,
                                                     a_action_state_bprofiles, a_action_state_lens, is_e2j)
            next_reward = self.predict_match(e_profiles, next_max_j_profiles,
                                             a_next_state_seq_profiless, a_next_seq_lens,
                                             next_max_j_state_seq_profiles, next_max_j_state_lens)
        else: # a=j, b=e
            next_max_e_profiles, next_max_e_state_seq_profiles, next_max_e_state_lens = \
                self.target_policy.predict_match_max_score(j_profiles, a_action_bprofiles,
                                                     a_next_state_seq_profiless, a_next_seq_lens,
                                                     a_action_state_bprofiles, a_action_state_lens, is_e2j)
            next_reward = self.predict_match(next_max_e_profiles, j_profiles,
                                             next_max_e_state_seq_profiles, next_max_e_state_lens,
                                             a_next_state_seq_profiless, a_next_seq_lens)
        return current_reward, next_reward

    def predict_like(self, a_profiles, b_profiles, a_bseq_profiless, a_bseq_lenss, is_e2j):
        # current_reward
        return self.main_policy.predict_like_score(a_profiles, b_profiles, a_bseq_profiless, a_bseq_lenss, is_e2j)

    def predict_match(self, e_profiles, j_profiles, e_now_seq_profiless, e_now_seq_lens, j_now_seq_profiless, j_now_seq_lens):
        # current_reward
        return self.main_policy.predict_match_score(e_profiles, j_profiles, e_now_seq_profiless, e_now_seq_lens, j_now_seq_profiless, j_now_seq_lens)

    def parameters(self):
        return list(self.person_profile2vec.parameters()) + list(self.job_profile2vec.parameters()) + \
               list(self.person_staterep.parameters()) + list(self.job_staterep.parameters()) + \
               list(self.main_policy.parameters())

    def update_target_QNet(self):
        self.target_policy.load_state_dict(self.main_policy.state_dict())