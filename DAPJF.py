import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

WORD_EMBED_DIM = 32
WORD_LSTM_EMBED_DIM = 16
TERM_LSTM_EMBED_DIM = 8

MAX_PROFILELEN = 10
MAX_TERMLEN = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DAPJF(torch.nn.Module):
    def __init__(self, word_embeddings):
        super(DAPJF, self).__init__()

        # embedding_matrix = [[0...0], [...], ...[]]
        self.Word_Embeds = torch.nn.Embedding.from_pretrained(word_embeddings, padding_idx=0)
        self.Word_Embeds.weight.requires_grad = False

        # 双向lstm
        self.Expect_Word_BiLSTM = torch.nn.LSTM(input_size=WORD_EMBED_DIM, hidden_size=WORD_LSTM_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.Job_Word_BiLSTM = torch.nn.LSTM(input_size=WORD_EMBED_DIM, hidden_size=WORD_LSTM_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=True)
        # 双向lstm
        self.Expect_Term_BiLSTM = torch.nn.LSTM(input_size=WORD_LSTM_EMBED_DIM*2, hidden_size=TERM_LSTM_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.Job_Term_BiLSTM = torch.nn.LSTM(input_size=WORD_LSTM_EMBED_DIM*2, hidden_size=TERM_LSTM_EMBED_DIM,
                                    num_layers=1, batch_first=True, bidirectional=True)


        # word-level attention-layer
        self.Expect_Word_Attn_Layer = torch.nn.Sequential(
            torch.nn.Linear(WORD_LSTM_EMBED_DIM * 2, WORD_LSTM_EMBED_DIM),
            torch.nn.Tanh(), # LeakyReLU
            torch.nn.Linear(WORD_LSTM_EMBED_DIM, 1, bias=False),
        )
        self.Job_Word_Attn_Layer = torch.nn.Sequential(
            torch.nn.Linear(WORD_LSTM_EMBED_DIM * 2, WORD_LSTM_EMBED_DIM),
            torch.nn.Tanh(), # LeakyReLU
            torch.nn.Linear(WORD_LSTM_EMBED_DIM, 1, bias=False),
        )
        # term-level attention-layer
        self.Expect_Term_Attn_Layer = torch.nn.Sequential(
            torch.nn.Linear(TERM_LSTM_EMBED_DIM* 2, TERM_LSTM_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(TERM_LSTM_EMBED_DIM, 1, bias=False),
        )
        self.Job_Term_Attn_Layer = torch.nn.Sequential(
            torch.nn.Linear(TERM_LSTM_EMBED_DIM* 2, TERM_LSTM_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(TERM_LSTM_EMBED_DIM, 1, bias=False),
        )

        self.A = Variable(torch.FloatTensor(torch.rand(50, 2*TERM_LSTM_EMBED_DIM)).to(DEVICE), requires_grad = True)
        self.B_D1 = Variable(torch.FloatTensor(torch.rand(50, 2*TERM_LSTM_EMBED_DIM)).to(DEVICE), requires_grad = True)
        self.B_D2 = Variable(torch.FloatTensor(torch.rand(50, 2*TERM_LSTM_EMBED_DIM)).to(DEVICE), requires_grad = True)

        # match mlp
        self.Match_MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * 2 * TERM_LSTM_EMBED_DIM + 3 * MAX_PROFILELEN, 2 * TERM_LSTM_EMBED_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * TERM_LSTM_EMBED_DIM, 1),
            torch.nn.Sigmoid()
        )

        self.ConvNet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=MAX_PROFILELEN, out_channels=MAX_PROFILELEN, kernel_size=5),
            # torch.nn.Conv2d(MAX_PROFILELEN, MAX_PROFILELEN, kernel_size=(4,4)),
            # BatchNorm1d只处理第二个维度
            torch.nn.BatchNorm1d(MAX_PROFILELEN),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2)
        )

    # [batch_size *2, MAX_PROFILELEN, MAX_TERMLEN] = (40, 15, 50)
    # term: padding same, word: padding 0
    # expects_sample, jobs_sample are in same format
    def forward(self, expects, jobs, domain=1):
        # word level:
        # [batch_size, MAX_PROFILELEN, MAX_TERMLEN] (40, 15, 50) ->
        # [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        shape = expects.shape
        expects_, jobs_ = expects.view([-1, shape[-1]]), jobs.view([-1, shape[-1]])

        # sort expects_: large to small
        # sorted [batch_size * MAX_PROFILELEN, MAX_TERMLEN](40 * 15, 50)
        expects_lens = (expects_ > 0).sum(dim=-1)
        expects_lens_sort, expects_ind_sort = expects_lens.sort(dim=0, descending=True)
        expects_sort = expects_[expects_ind_sort, :]
        # sort jobs_sample_: large to small
        jobs_lens = (jobs_ > 0).sum(dim=-1)
        jobs_lens_sort, jobs_ind_sort = jobs_lens.sort(dim=0, descending=True)
        jobs_sort = jobs_[jobs_ind_sort, :]

        # embeddings: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        expects_wordembed_sort = self.Word_Embeds(expects_sort).float()
        jobs_wordembed_sort = self.Word_Embeds(jobs_sort).float()

        # compress: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, EMBED_DIM]
        expects_wordpack_sort = pack_padded_sequence(expects_wordembed_sort, expects_lens_sort, batch_first=True)
        jobs_wordpack_sort = pack_padded_sequence(jobs_wordembed_sort, jobs_lens_sort, batch_first=True)

        # dynamic: to execute the LSTM over only the valid timesteps
        # output_sequence: [batch_size * MAX_PROFILELEN, time_step, hidden_size]
        # h: [num_layers*num_directions, batch_size * MAX_PROFILELEN, hidden_size]
        # c: [num_layers*num_directions, batch_size * MAX_PROFILELEN, hidden_size]
        expects_wordoutput_pack_sort, _ = self.Expect_Word_BiLSTM(expects_wordpack_sort)
        jobs_wordoutput_pack_sort, _ = self.Job_Word_BiLSTM(jobs_wordpack_sort)

        # output: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2]
        expects_wordoutput_sort, _ = torch.nn.utils.rnn.pad_packed_sequence(expects_wordoutput_pack_sort, batch_first=True)
        jobs_wordoutput_sort, _ = torch.nn.utils.rnn.pad_packed_sequence(jobs_wordoutput_pack_sort, batch_first=True)

        # rollback-sort
        # print(shape, expects_wordoutput_sort.shape, jobs_wordoutput_sort.shape)
        # print(expects_ind_sort.shape)
        expects_wordoutput = expects_wordoutput_sort[expects_ind_sort, :, :]
        jobs_wordoutput = jobs_wordoutput_sort[jobs_ind_sort, :, :]

        # attention: [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2]
        expects_wordattn = F.softmax(self.Expect_Word_Attn_Layer(expects_wordoutput), dim=-2)
        jobs_wordattn = F.softmax(self.Job_Word_Attn_Layer(jobs_wordoutput), dim=-2)

        # [batch_size * MAX_PROFILELEN, MAX_TERMLEN, words_lstm_hidden_dims * 2] ->
        # [batch_size * MAX_PROFILELEN, words_lstm_hidden_dims * 2]
        expects_terms = (expects_wordattn * expects_wordoutput).sum(-2)\
            .view([shape[0], MAX_PROFILELEN, 2 * WORD_LSTM_EMBED_DIM])
        jobs_terms = (jobs_wordattn * jobs_wordoutput).sum(-2)\
            .view([shape[0], MAX_PROFILELEN, 2 * WORD_LSTM_EMBED_DIM])

        # term level:
        # [batch_size * MAX_PROFILELEN, words_lstm_hidden_dims * 2] ->
        # output_sequence: [batch_size, MAX_PROFILELEN, terms_lstm_hidden_dims * 2]
        expects_termoutput, _ = self.Expect_Term_BiLSTM(expects_terms)
        jobs_termoutput, _ = self.Job_Term_BiLSTM(jobs_terms)

        # attention: [batch_size, MAX_PROFILELEN, terms_lstm_hidden_dims * 2]
        expect_termattn = F.softmax(self.Expect_Term_Attn_Layer(expects_termoutput), dim=-2)
        job_termattn = F.softmax(self.Job_Term_Attn_Layer(jobs_termoutput), dim=-2)

        # [batch_size, MAX_PROFILELEN, terms_lstm_hidden_dims*2] -> [batch_size, terms_lstm_hidden_dims*2]
        expects_embed = (expect_termattn * expects_termoutput).sum(-2)
        jobs_embed = (job_termattn * jobs_termoutput).sum(-2)

        # conv
        if domain == 1:
            W = torch.mm(torch.transpose(self.A, 1, 0), self.B_D1)
        else:
            W = torch.mm(torch.transpose(self.A, 1, 0), self.B_D2)

        # [batch_size, MAX_PROFILELEN, terms_lstm_hidden_dims * 2]
        expects_termoutput_1 = torch.mm(expects_termoutput.contiguous().view([shape[0]*shape[1], -1]), W).view([shape[0], shape[1], -1])
        # [batch_size, MAX_PROFILELEN, MAX_PROFILELEN]
        M = torch.bmm(expects_termoutput_1, jobs_termoutput.permute([0, 2, 1]))
        interaction = self.ConvNet(M).view([shape[0], -1])

        # profile-level interaction
        return self.Match_MLP(torch.cat([jobs_embed, expects_embed, interaction], dim=-1)).squeeze(-1)

    # def parameters(self):
    #     return list(self.Expect_Word_BiLSTM.parameters()) + list(self.Expect_Term_BiLSTM.parameters()) + \
    #            list(self.Expect_Word_Attn_Layer.parameters()) + list(self.Expect_Term_Attn_Layer.parameters()) + \
    #            list(self.Job_Word_BiLSTM.parameters()) + list(self.Job_Term_BiLSTM.parameters()) + \
    #            list(self.Job_Word_Attn_Layer.parameters()) + list(self.Job_Term_Attn_Layer.parameters())
