import re
import pandas as pd

path = 'result/submission(cat).csv'
path1 = 'result/sample_submission.csv'
result = pd.read_csv(path)
answer = pd.read_csv(path1)
print(result.shape)
print(result)
label = list(result['label'])
print(label)