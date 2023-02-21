import json
import pandas as pd
pd.set_option('display.max_columns', None)

# with open("logs/simu1/eval/result_ivc_50.json", "r") as f:
#     simu_our = pd.DataFrame(json.load(f))
# print(simu_our)
#
# with open("logs/simu1/eval/result_50.json", "r") as f:
#     simu_baseline =  pd.DataFrame(json.load(f))
# print(simu_baseline)
#
# with open("logs/simu1/eval/result_50_transee.json", "r") as f:
#     transee =  pd.DataFrame(json.load(f))
# print(transee)
#
# simu_all=pd.concat([simu_baseline,transee,simu_our],axis=1).T
# print(simu_all)
# simu_mean=simu_all.mean(axis=1)
# simu_all=pd.concat([simu_all,simu_mean,simu_all.std(axis=1)],axis=1)
# print(simu_mean)
# print(simu_all.round(4).to_latex(caption="Comparison between baselines on Simulation Dataset "))
#
# print("=========news=============")
# with open("logs/news/eval/result_ivc_50.json", "r") as f:
#     news_our = pd.DataFrame(json.load(f))
# print(news_our)
#
# with open("logs/news/eval/result_50.json", "r") as f:
#     news_baseline =  pd.DataFrame(json.load(f))
# print(news_baseline)
#
# with open("logs/news/eval/result_50_transee.json", "r") as f:
#     news_transee =  pd.DataFrame(json.load(f))
# print(news_transee)
#
# news_all=pd.concat([news_baseline,news_transee,news_our],axis=1).T
# print(news_all)
# news_all=pd.concat([news_all,news_all.mean(axis=1),news_all.std(axis=1)],axis=1)
# print(news_all.round(4).to_latex(caption="Comparison between baselines on News Dataset "))
#
# print(news_all)
#
# print("=========ihdp=============")
# with open("logs/ihdp/eval/result_ivc_50.json", "r") as f:
#     ihdp_our = pd.DataFrame(json.load(f))
# print(ihdp_our)
#
# with open("logs/ihdp/eval/result_50.json", "r") as f:
#     ihdp_baseline =  pd.DataFrame(json.load(f))
# print(ihdp_baseline)
#
# with open("logs/ihdp/eval/result_50_transee.json", "r") as f:
#     ihdp_transee =  pd.DataFrame(json.load(f))
# print(ihdp_transee)
#
# ihdp_all=pd.concat([ihdp_baseline,ihdp_transee,ihdp_our],axis=1).T
# print(ihdp_all)
# ihdp_all=pd.concat([ihdp_all,ihdp_all.mean(axis=1),ihdp_all.std(axis=1)],axis=1)
# print(ihdp_all.round(4).to_latex(caption="Comparison between baselines on IHDP Dataset "))
#
# print(ihdp_all)
# result_our=json.loads("logs/simu1/eval/result_ivc.json")
# result_baseline= json.loads("logs/simu1/eval/result.json")
# print("===============simu_ablation_beta=============")
# with open("logs/simu1/eval/result_ivc_50_no_beta.json", "r") as f:
#     simu_nobeta = pd.DataFrame(json.load(f))
# print(simu_nobeta.mean())
# print("===============news_ablation_beta=============")
# with open("logs/news/eval/result_ivc_50_no_beta.json", "r") as f:
#     news_nobeta = pd.DataFrame(json.load(f))
# print(news_nobeta.mean())
# print("===============ihdp_ablation_beta=============")
# with open("logs/ihdp/eval/result_ivc_50_no_beta.json", "r") as f:
#     ihdp_nobeta = pd.DataFrame(json.load(f))
# print(ihdp_nobeta.mean())
# #
#
# print("===============simu_ablation_gamma=============")
# with open("logs/simu1/eval/result_ivc_50_no_gamma.json", "r") as f:
#     simu_nogamma = pd.DataFrame(json.load(f))
# print(simu_nogamma)
# print(simu_nogamma.mean())
# print("===============news_ablation_gamma=============")
# with open("logs/news/eval/result_ivc_50_no_gamma.json", "r") as f:
#     news_nogamma = pd.DataFrame(json.load(f))
# print(news_nogamma.mean())
print("===============ihdp_ablation_gamma=============")
with open("logs/ihdp/eval/result_ivc_50_no_gamma.json", "r") as f:
    ihdp_nogamma = pd.DataFrame(json.load(f))
print(ihdp_nogamma)

print(ihdp_nogamma.mean())

# print("===============simu_ablation_reweight=============")
# with open("logs/simu1/eval/result_ivc_50_no_reweight.json", "r") as f:
#     simu_noreweight = pd.DataFrame(json.load(f))
# print(simu_noreweight.mean())
# print("===============news_ablation_reweight=============")
# with open("logs/news/eval/result_ivc_50_no_reweight.json", "r") as f:
#     news_noreweight = pd.DataFrame(json.load(f))
# print(news_noreweight.mean())
print("===============ihdp_ablation_reweight=============")
with open("logs/ihdp/eval/result_ivc_50_no_reweight.json", "r") as f:
    ihdp_noreweight = pd.DataFrame(json.load(f))
print(ihdp_noreweight.mean())
print(ihdp_noreweight.std())


print("===============ihdp_0.3_reweight=============")
with open("logs/ihdp/eval/result_ivc_50_03reweight.json", "r") as f:
    ihdp_03reweight = pd.DataFrame(json.load(f))
print(ihdp_03reweight.mean())
print(ihdp_03reweight.std())

print("===============ihdp_0.7_reweight=============")
with open("logs/ihdp/eval/result_ivc_50_07_reweight.json", "r") as f:
    ihdp_07reweight = pd.DataFrame(json.load(f))
print(ihdp_07reweight.mean())
print(ihdp_07reweight.std())

print("===============ihdp_1.5reweight=============")
with open("logs/ihdp/eval/result_ivc_50_15reweight.json", "r") as f:
    ihdp_2noreweight = pd.DataFrame(json.load(f))
print(ihdp_2noreweight.mean())
print(ihdp_2noreweight.std())

print("===============ihdp_2reweight=============")
with open("logs/ihdp/eval/result_ivc_50_2reweight.json", "r") as f:
    ihdp_2noreweight = pd.DataFrame(json.load(f))
print(ihdp_2noreweight.mean())
print(ihdp_2noreweight.std())
