library(dprep)
crd=read.csv("/Users/francisco/Desktop/Fall_2018/Data_Mining/HW/HW2/datasets/disc_crd.csv")
inc = inconsist(crd)
inc
delta = 0.01 # Delta sugerido por el Prof.
finco(crd,inc+delta)
# 7,8,11 son los mejores
# FINCO seleciono PAY_2, PAY_3, PAY_6 como los mejores 3 features
