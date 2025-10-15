import pandas as pd
df = pd.read_csv("results/gsm8k_main_test.csv")
bad = df[df.gold!=df.pred].sample(20)
print(bad[["gold","pred"]])
