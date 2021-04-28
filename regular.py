import pandas as pd
import re

from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from configs import fake_test_res, fake_test_data, fake_test_ref, test_ref, test_res, test_data

test_res_df = pd.read_csv(test_res, sep="\t", names=["negative", "positive"])
test_data_df = pd.read_csv(test_data)
test_ref_df = pd.read_csv(test_ref, sep="\t")
print(test_ref_df)
def eval(res_df : DataFrame, ref_df : DataFrame):
    correct_cnt = 0
    total_cnt = 0
    for res_id in res_df.index:
        total_cnt += 1
        ref_res = ref_df.loc[res_id, 'label']
        final_res = res_df.loc[res_id, 'final_out']
        if str(ref_res) == str(final_res):
            correct_cnt += 1
    print("正确率: {percentage}%".format(percentage = (correct_cnt * 100 / total_cnt)))

for res_id in test_res_df.index:
    test_res_df.loc[res_id, 'text'] = test_data_df.loc[res_id, "text"]

res = []
for res_id in test_res_df.index:
    content = test_res_df.loc[res_id, "text"]

    positive_score = test_res_df.loc[res_id, "positive"]
    negative_score = test_res_df.loc[res_id, "negative"]

    if positive_score >= negative_score:
        test_res_df.loc[res_id, "final_out"] = "1"
        res.append(1)
    else:
        test_res_df.loc[res_id, "final_out"] = "0"
        res.append(0)

print("矫正前:")
eval(test_res_df, test_ref_df)
print(res)

# 基于规则的预测矫正
for res_id in test_res_df.index:
    content = test_res_df.loc[res_id, "text"]

    if re.search(re.compile(r'烂+片?'), content):
        test_res_df.loc[res_id, 'final_out'] = "0"

    elif re.search(re.compile(r'睡+着?'), content):
        test_res_df.loc[res_id, 'final_out'] = "0"
    
    elif re.search(re.compile(r'无聊+'), content):
        test_res_df.loc[res_id, 'final_out'] = "0"

    elif re.search(re.compile(r'不好看+'), content):
        test_res_df.loc[res_id, 'final_out'] = "0"

    elif re.search(re.compile(r'真?不错+'), content):
        test_res_df.loc[res_id, 'final_out'] = "1"
    
    
print("矫正后:")
eval(test_res_df, test_ref_df)
test_res_df.final_out.to_csv("./answer.txt", index=False, header=False)