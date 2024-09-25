import pandas as pd
df = pd.read_csv('static\labels.csv', index_col='LabelNo')

print(df)
print(df.size)
print("\n Labels w rgb vals\n")
for i in range(32):
    print(df.iloc[i]['r'])