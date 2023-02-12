import json
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print("===============simu_ablation_tune_results=============")

input_file = open ('logs/simu1/tune/result_ivc_tune_20.json')
json_array = json.load(input_file)
store_list = []
real_list=[]
for item in json_array:
    store_list.append(item)
for i in range(1,len(store_list),2):
    real_list.append(store_list[i])
real_list=pd.DataFrame(real_list)
real_list["mean"]= 0
for i in range(len(real_list)):
    real_list["mean"][i]=pd.DataFrame(real_list["Vcnet_disentangled"][i]).mean()
real_list.to_csv("tune_result_simu_20.csv")
print(real_list.sort_values(by=["mean"]))
print("length",real_list.shape[0])


print("===============news_tune_results=============")

input_file = open ('logs/news/tune/result_ivc_tune_20.json')
json_array = json.load(input_file)
store_list = []
real_list=[]
for item in json_array:
    store_list.append(item)
for i in range(0,len(store_list),2):
    real_list.append(store_list[i])
real_list=pd.DataFrame(real_list)
print("============",real_list)

real_list["mean"]= 0
for i in range(len(real_list)):
    real_list["mean"][i]=pd.DataFrame(real_list["Vcnet_disentangled"][i]).mean()
real_list.to_csv("tune_result_news_20.csv")
print("length",real_list.shape[0])

print(real_list.sort_values(by=["mean"]))

print("===============ihdp_tune_results=============")
input_file = open ('logs/ihdp/tune/result_ivc_tune_20.json')
json_array = json.load(input_file)
store_list = []
real_list=[]
for item in json_array:
    store_list.append(item)
for i in range(1,len(store_list),2):
    real_list.append(store_list[i])
real_list=pd.DataFrame(real_list)
real_list["mean"]= 0
for i in range(len(real_list)):
    real_list["mean"][i]=pd.DataFrame(real_list["Vcnet_disentangled"][i]).mean()
real_list.to_csv("tune_result_ihdp_20.csv")
print("length",real_list.shape[0])

print(real_list.sort_values(by=["mean"]))
