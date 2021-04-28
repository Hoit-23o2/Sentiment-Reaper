from collections import defaultdict
from os import sep
from pandas.core.frame import DataFrame
from torch.nn.modules.loss import CrossEntropyLoss
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from configs import train_data, test_data

# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

# 配置参数
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ["negative", "positive"]

# 读取数据
review_data_df = pd.read_csv(train_data)
review_data_df = shuffle(review_data_df)

target_review_data_df = pd.read_csv(test_data)

for review_id in review_data_df.index:
    sentiment = review_data_df.loc[review_id, "label"]
    if sentiment == "positive":
        review_data_df.loc[review_id, "label"] = int(1)
    else :
        review_data_df.loc[review_id, "label"] = int(0)

PRE_TRAINED_MODEL_NAME = "./data/chinese_L-12_H-768_A-12"

# 加载预训练模型
BertConfig.from_json_file("./data/chinese_L-12_H-768_A-12/config.json")
tokenizer = BertTokenizer.from_pretrained("./data/chinese_L-12_H-768_A-12")

# 小测一下
# sample_txt = '牛逼啊，这是我见过断肢血腥类道具最假的片子，剧情也烂得一批'
# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')

# # 标记每个句子的结束
# print((tokenizer.sep_token, tokenizer.sep_token_id))
# # 我们必须把这个cls_token加到每一个句子之前，从而使BERT知道我们正在分类
# print((tokenizer.cls_token, tokenizer.cls_token_id))

# sample_encoding = tokenizer.encode_plus(
#     sample_txt,
#     max_length = 32,
#     add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
#     return_token_type_ids = False,
#     padding = 'max_length',
#     return_attention_mask = True,
#     return_tensors = 'pt',     # 返回PyTorch tensors
# )

# print(sample_encoding.keys())
# print(len(sample_encoding['input_ids'][0]))
# print(sample_encoding['input_ids'][0])

# tokens = tokenizer.convert_ids_to_tokens(sample_encoding['input_ids'][0])
# print(tokens)


# 开始正式Work
token_lens = []

for record in review_data_df.text:
    tokens = tokenizer.encode(record, max_length = 512, truncation = True)
    token_lens.append(len(tokens))

# sns.distplot(token_lens)
# plt.xlim([0, 256]);
# plt.xlabel('Token count');
# plt.show()

# >>>> max_length取60左右就够了
MAX_LEN = 60

class MovieReviewDataSet(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        if len(self.targets) != 0:
            target = self.targets[item]
        else:
            target = []
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'review_text': review,
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# 划分训练集
review_data_df_train, review_data_df_test = train_test_split(
    review_data_df,
    test_size = 0.1,
)

# 划分验证集、测试集
review_data_df_val, review_data_df_test = train_test_split(
    review_data_df_test,
    test_size = 0.5
)

review_data_df_val = pd.read_csv("./data/test1_data_manual_verify.txt", sep="\t")

print((review_data_df_train.shape, review_data_df_val.shape, review_data_df_test.shape))

def create_data_loader(df : DataFrame, tokenizer, max_len, batch_size):
    ds = MovieReviewDataSet(
        reviews = df.text.to_numpy(),
        targets = df.label.to_numpy() if "label" in df.columns else [],
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size = batch_size,            # 读入时千万不要shuffle
    )

BATCH_SIZE = 16

train_data_loader = create_data_loader(review_data_df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(review_data_df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(review_data_df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# 读取数据
data = next(iter(train_data_loader))
print(data.keys())

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# 建立BERT模型
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

# 测试一下
# o = bert_model(
#     input_ids = sample_encoding['input_ids'],
#     attention_mask = sample_encoding['attention_mask'], 
#     output_attentions=True, 
#     output_hidden_states=True
# )
# last_hidden_state = o.last_hidden_state
# pooler_output = o.pooler_output

# print(last_hidden_state.shape)
# print(pooler_output.shape)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.out  = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        o = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            output_attentions=True, 
            output_hidden_states=True
        )
        pooler_output = o.pooler_output
        output = self.drop(pooler_output)
        return self.out(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)

# 训练了
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape)      # 一个batch的形状
print(attention_mask.shape) # 一个batch的形状

softmax = nn.Softmax(1)
output = softmax(model(input_ids, attention_mask))
print(output)

EPOCHS = 10
optimizer = AdamW(model.parameters(), lr = 1e-5, correct_bias = False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
    model : nn.Module,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_example
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        # 预测结果
        outputs = model(
            input_ids, 
            attention_mask
        )

        _, preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # 反向传播
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_example, np.mean(losses)


def eval_model(
    model : nn.Module,
    data_loader,
    loss_fn,
    device,
    n_example
):
    model = model.eval()
    
    pres = []
    losses = []
    correct_predictions = 0

    # 验证
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review_text"]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)
            # 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)

            pres.extend(preds)
            losses.append(loss.item())
    pres = torch.stack(pres).cpu()
    return correct_predictions.double() / n_example, np.mean(losses), pres

# 正式训练

# history = defaultdict(list)
# best_acc = 0

# for epoch in range(EPOCHS):
#     print(f'轮次 {epoch + 1}/{EPOCHS}')
#     print('-' * 10)
#     # 训练
#     train_acc, train_loss = train_epoch(
#         model,
#         train_data_loader,
#         loss_fn,
#         optimizer,
#         device,
#         scheduler,
#         len(review_data_df_train)
#     )

#     print(f'训练误差： {train_loss} 精确度： {train_acc}')
#     # 验证
#     val_acc, val_loss = eval_model(
#         model,
#         val_data_loader,
#         loss_fn,
#         device,
#         len(review_data_df_val)
#     )
#     print(f'验证误差： {val_loss} 精确度： {val_acc}')
#     print("")

#     history["train_acc"].append(train_acc)
#     history["train_loss"].append(train_loss)
#     history['val_acc'].append(val_acc)
#     history['val_loss'].append(val_loss)

#     if val_acc > best_acc:
#         torch.save(model.state_dict(), './modelCKPT/bert_best_model_state.bin')
#         best_acc = val_acc


# plt.plot(history['train_acc'], label='训练精确度')
# plt.plot(history['val_acc'], label='验证精确度')
# plt.title('训练历史记录')
# plt.ylabel('精确度')
# plt.xlabel('轮次')
# plt.legend()
# plt.ylim([0, 1]);  
# plt.savefig("./data/50epochs.png")

model.load_state_dict(torch.load('./modelCKPT/bert_best_model_state.bin'))
model = model.to(device)

print(review_data_df_val)
test_acc, _, pres = eval_model(
  model,
  val_data_loader,
  loss_fn,
  device,
  len(review_data_df_val)
)

print(pres)

print(test_acc.item())


final_test_data_loader = create_data_loader(target_review_data_df, tokenizer, MAX_LEN, BATCH_SIZE)
softmax = nn.Softmax(1)

# def get_predictions(model : nn.Module, data_loader):
#     model = model.eval()

#     predictions = []
#     prediction_probs = []

#     with torch.no_grad():
#         for data in data_loader:
#             input_ids = data["input_ids"].to(device)
#             attention_mask = data["attention_mask"].to(device)
#             outputs = model(
#                 input_ids = input_ids,
#                 attention_mask = attention_mask
#             )
            
#             # 横向查看 是哪一列最大
#             _, preds = torch.max(outputs, dim = 1)
            
#             predictions.extend(preds)
#             prediction_probs.extend(outputs)

#     predictions = torch.stack(predictions).cpu()
#     prediction_probs = torch.stack(prediction_probs).cpu()
#     return predictions, prediction_probs

def get_predictions(
    model : nn.Module,
    data_loader
):
    model.eval()

    predictions = []
    prediction_probs = []
    # 验证
    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # outputs = softmax(outputs)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(outputs)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return predictions, prediction_probs

y_pred, y_pred_probs = get_predictions(
    model,
    val_data_loader
)

print(y_pred)
print(y_pred_probs)

y_pred_probs_np = y_pred_probs.numpy()
y_pred_probs_df = pd.DataFrame(y_pred_probs_np)
y_pred_probs_df.to_csv("./data/test_results2.csv", header=False, sep="\t", index=False)



