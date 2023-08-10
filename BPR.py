
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BPRMF(torch.nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPRMF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = torch.nn.Embedding(user_num, factor_num)
        torch.nn.init.normal_(self.embed_user.weight, std=0.01)
        self.embed_user.weight.requires_grad = True
        self.embed_item = torch.nn.Embedding(item_num, factor_num)
        torch.nn.init.normal_(self.embed_item.weight, std=0.01)
        self.embed_item.weight.requires_grad = True

        self.item_idx = torch.LongTensor(torch.arange(0, item_num)).to(DEVICE)

    def forward(self, user, items):
        user_embed = self.embed_user(user)
        item_embeds = self.embed_item(items)
        scores = torch.bmm(user_embed.unsqueeze(dim=1), item_embeds.permute((0,2,1))).squeeze(dim=1)
        return scores