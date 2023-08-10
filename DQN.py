import torch
from torch.nn.utils.rnn import pack_padded_sequence
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ITEM_EMBED_DIM = 32
STATE_GRU_DIM = 32

class StateRep(torch.nn.Module):
    def __init__(self, ):
        super(StateRep, self).__init__()
        self.GRU = torch.nn.GRU(input_size=ITEM_EMBED_DIM, hidden_size=STATE_GRU_DIM,
                                    num_layers=1, batch_first=True, bidirectional=False)
    def forward(self, item_seqs, item_seq_len):
        # sort for pack
        lens_sort, ind_sort = item_seq_len.sort(dim=0, descending=True)
        item_seqs_ = item_seqs[ind_sort, :, :]
        # pack
        seq_pack = pack_padded_sequence(item_seqs_, lens_sort, batch_first=True)
        # gru
        seq_output, seq_state = self.GRU(seq_pack)
        return seq_state.squeeze(0)[ind_sort, :]
    def parameters(self, ):
        return list(self.GRU.parameters())


class DQN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(DQN, self).__init__()

        self.DQN_MLP = torch.nn.Sequential()
        for ind, layer_size in enumerate(layer_sizes):
            in_size = STATE_GRU_DIM + ITEM_EMBED_DIM if ind==0 else layer_sizes[ind-1]
            self.DQN_MLP.add_module('linear-'+str(ind), torch.nn.Linear(in_size, layer_sizes[ind]))
            # self.DQN_MLP.add_module('relu-'+str(ind), torch.nn.ReLU())
            self.DQN_MLP.add_module('tanh-'+str(ind), torch.nn.Tanh())
            # self.DQN_MLP.add_module('sigmoid-'+str(ind), torch.nn.Sigmoid())
        self.DQN_MLP.add_module('linear', torch.nn.Linear(layer_sizes[-1], 1))
        print('DQN_MLP=', self.DQN_MLP)
    def forward(self, state_embed, item_embed):
        return self.DQN_MLP(torch.cat([state_embed, item_embed], dim=-1)).squeeze(-1)
    # item_masks [0,1]
    def forward_max(self, state_embed, item_embeds, item_masks):
        state_embeds = torch.unsqueeze(state_embed, 1)
        shape = state_embeds.shape
        state_embeds = state_embeds.expand(shape[0], item_embeds.shape[1], shape[2])
        scores = torch.sigmoid(self.DQN_MLP(torch.cat([state_embeds, item_embeds], dim=-1)).squeeze(-1))
        scores = scores * item_masks
        scores_max, indices_max = torch.max(scores, dim=1)
        return scores_max
    def parameters(self):
        return list(self.DQN_MLP.parameters())

class DQN_REC(torch.nn.Module):
    def __init__(self, isize, discounts):
        super(DQN_REC, self).__init__()
        self.isize = isize
        self.discounts = discounts

        self.item_embeds = torch.nn.Embedding(isize+1, ITEM_EMBED_DIM, padding_idx=0)
        torch.nn.init.normal_(self.item_embeds.weight, std=0.01)
        self.item_embeds.weight.requires_grad = True

        self.dqn = DQN([32, 16])
        self.staterep = StateRep()

        self.item_idx = torch.LongTensor(torch.arange(0, isize+1)).to(DEVICE)

    def forward(self, states, states_len, actions, rewards_kstep,
                next_states, next_states_len, next_action_mask):
        states_ = self.item_embeds(states).float()
        state_embed = self.staterep(states_, states_len)
        action_embed = self.item_embeds(actions)
        current_rewards = self.dqn(state_embed, action_embed)

        next_states_ = self.item_embeds(next_states).float()
        next_state_embed = self.staterep(next_states_, next_states_len)
        item_embed = self.item_embeds(self.item_idx).unsqueeze(dim=0).repeat((next_state_embed.shape[0], 1, 1))
        # detach or not
        next_rewards = self.dqn.forward_max(next_state_embed, item_embed, next_action_mask) # .detach()
        return current_rewards, sum(self.discounts[:-1]) + self.discounts[-1] * next_rewards

    def predict(self, states, states_len, action_mask, topN):
        states_ = self.item_embeds(states).float()
        state_embed = self.staterep(states_, states_len)

        state_embeds = torch.unsqueeze(state_embed, 1)
        shape = state_embeds.shape
        state_embeds = state_embeds.expand(shape[0], self.isize+1, shape[2])

        item_embed = self.item_embeds(self.item_idx).unsqueeze(dim=0).repeat((state_embeds.shape[0], 1, 1))
        scores = torch.sigmoid(self.dqn(state_embeds, item_embed))
        scores = scores * action_mask
        _, inds = torch.topk(scores, topN)
        return inds

    def parameters(self, ):
        return list(self.item_embeds.parameters()) + self.dqn.parameters() + self.staterep.parameters()