from core.field_meta import *
from exp1_functional import compute_all_kline_indicators, compute_future_extremes
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
import torch
from core.modules import SettledPositionalEncoding
from transformers import get_cosine_schedule_with_warmup
from core.utils import TorchTrainingVisualizer
from tqdm import tqdm


# schema: time open close high low volume
TRAIN_CSV = "/media/xavier/Samsumg/codes/blankly/scripts/train.csv"
TEST_CSV = "/media/xavier/Samsumg/codes/blankly/scripts/test.csv"
DIM=32
BATCH=2
LEN=120


train_df = pd.read_csv(TRAIN_CSV,index_col=0)
test_df = pd.read_csv(TEST_CSV,index_col=0)
train_feature = pd.merge(compute_all_kline_indicators(train_df), train_df, 'inner', 'time')
test_feature = pd.merge(compute_all_kline_indicators(test_df), test_df, 'inner', 'time')
train_target = compute_future_extremes(train_df)[['target_up_3pct_5','target_down_3pct_5','time']]
test_target = compute_future_extremes(test_df)[['target_up_3pct_5','target_down_3pct_5','time']]
train_target['target'] = (train_target['target_up_3pct_5'] & ~train_target['target_down_3pct_5']).astype(int)
test_target['target'] = (test_target['target_up_3pct_5'] & ~test_target['target_down_3pct_5']).astype(int)

full_train = pd.merge(train_feature, train_target, 'inner', 'time')
full_train.dropna(axis=1, how='any',inplace=True)
full_test = pd.merge(test_feature, test_target, 'inner', 'time')
full_test.dropna(axis=1, how='any',inplace=True)

target_col = 'target'
fea_col = []
for col in full_train.columns:
    if col not in ['time', target_col] and 'target' not in col:
        fea_col.append(col)

print(f'fea_col: {fea_col}')

fm, pi = get_meta_process_info_from_dataframe(full_train[fea_col])

preprcess_func = get_preprocess_function(fm, pi)

class MySet(Dataset):

    def __init__(self, df) -> None:
        super().__init__()
        self.datas = df
    
    def __getitem__(self, index):
        part = self.datas.iloc[index:index+LEN,:]
        dis_idx = []
        dis_mask = []
        con_value = []
        con_idx = []
        con_mask = []
        for i in range(LEN):
            x = preprcess_func(part.iloc[i,:].to_dict())
            dis_idx.append(x['discrete_index'])
            dis_mask.append(x['discrete_mask'])
            con_value.append(x['continue_value'])
            con_idx.append(x['continue_index'])
            con_mask.append(x['continue_mask'])
        return torch.stack(dis_idx), torch.stack(dis_mask),torch.stack(con_value),torch.stack(con_idx),torch.stack(con_mask), part[target_col].iloc[LEN-1]

    def __len__(self):
        return self.datas.shape[0]-LEN+1


class WholeTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.ebd_module = get_embedding_module(fm, pi, DIM, 10)
        self.compress_model = BertEncoder(BertConfig(vocab_size=10, hidden_size=DIM,num_hidden_layers=4,num_attention_heads=8,intermediate_size=512))
        self.register_parameter("compress_token", torch.nn.Parameter(torch.randn([1, DIM])))
        self.register_buffer("compress_mask", torch.ones([1, 1]))
        self.seq_model = BertEncoder(BertConfig(vocab_size=10, hidden_size=DIM,num_hidden_layers=4,num_attention_heads=8,intermediate_size=512))
        self.seq_embed = SettledPositionalEncoding(DIM, LEN)
        self.register_parameter("clf_token", torch.nn.Parameter(torch.randn([1, DIM])))
        self.fc = torch.nn.Linear(DIM, 1)
    
    def forward(self, di,dm,cv,ci,cm, batch=BATCH):
        emb = self.ebd_module(
            di.view(batch*LEN, -1).cuda(),
            dm.view(batch*LEN, -1).cuda(),
            cv.view(batch*LEN, -1).cuda(),
            ci.view(batch*LEN, -1).cuda(),
            cm.view(batch*LEN, -1).cuda()
        )
        mask = torch.cat([self.compress_mask.repeat(batch*LEN, 1), emb[1]], dim=1)
        full_mask = (1.0 - mask[:, None, None, :]) * torch.finfo(torch.float).min
        compressed = self.compress_model(
            hidden_states=torch.cat([self.compress_token.repeat(batch*LEN, 1, 1), emb[0]], dim=1),
            attention_mask=full_mask
        ).last_hidden_state[:, 0, :]
        seq_embed = compressed.view(batch, LEN, -1)
        positioned_seq_embed = self.seq_embed(seq_embed)
        last_layer = self.seq_model(
            hidden_states=torch.cat([positioned_seq_embed, self.clf_token.repeat(batch, 1, 1)], dim=1)
        ).last_hidden_state[:, -1, :]
        return self.fc(last_layer)


def train():
    ms = MySet(full_train)
    dl = DataLoader(ms, BATCH, True,drop_last=True)
    ttv = TorchTrainingVisualizer("run/", "seq_train")

    model = WholeTransformer()
    model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5,betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss()

    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=2*len(ms)//BATCH,
        num_training_steps=50*len(ms)//BATCH
    )

    for e in tqdm(range(50)):
        for di,dm,cv,ci,cm,gt in dl:
            opt.zero_grad()
            # [batch, 1]
            out = model(di,dm,cv,ci,cm)

            loss = bce(out, gt.float().cuda().view(-1,1))
            float_loss = loss.detach().cpu().float()
            ttv.log_metrics({"loss": float_loss,"lr": scheduler.get_last_lr()[0]})
            loss.backward()
            opt.step()
            scheduler.step()
        torch.save(model.state_dict(), "run/seq.pth")
    ttv.close()

def test():
    ms = MySet(full_test)
    dl = DataLoader(ms, BATCH, False, drop_last=True)

    model = WholeTransformer()
    model.cuda()
    model.load_state_dict(torch.load("run/seq.pth"))
    model.eval()

    y_gt = []
    y_pred = []
    with torch.no_grad():
        for di,dm,cv,ci,cm,gt in tqdm(dl):
            y_gt.append(gt.numpy())
            out = torch.sigmoid(model(di,dm,cv,ci,cm))
            f = out.detach().cpu().numpy()[:,0]
            y_pred.append(f)
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(np.concatenate(y_gt), np.concatenate(y_pred)))

if __name__ == "__main__":
    train()