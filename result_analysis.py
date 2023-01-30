import json
import pandas as pd

with open("logs/simu1/eval/result_ivc.json", "r") as f:
    simu_our = pd.DataFrame(json.load(f))
print(simu_our)

with open("logs/simu1/eval/result.json", "r") as f:
    simu_baseline =  pd.DataFrame(json.load(f))
print(simu_baseline)

simu_all=pd.concat([simu_baseline,simu_our],axis=1)
print(simu_all)

print(simu_all.mean(axis=0))

print("=========news=============")
with open("logs/news/eval/result_ivc.json", "r") as f:
    news_our = pd.DataFrame(json.load(f))
print(news_our)

with open("logs/news/eval/result.json", "r") as f:
    news_baseline =  pd.DataFrame(json.load(f))
print(news_baseline)

news_all=pd.concat([news_baseline,news_our],axis=1)
print(news_all)

print(news_all.mean(axis=0))

print("=========ihdp=============")
with open("logs/ihdp/eval/result_ivc.json", "r") as f:
    ihdp_our = pd.DataFrame(json.load(f))
print(ihdp_our)

with open("logs/ihdp/eval/result.json", "r") as f:
    ihdp_baseline =  pd.DataFrame(json.load(f))
print(ihdp_baseline)

ihdp_all=pd.concat([ihdp_baseline,ihdp_our],axis=1)
print(ihdp_all)

print(ihdp_all.mean(axis=0))
# result_our=json.loads("logs/simu1/eval/result_ivc.json")
# result_baseline= json.loads("logs/simu1/eval/result.json")

# print(result_our)
# print(result_baseline)