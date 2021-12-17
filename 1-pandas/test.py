import pandas as pd
import numpy as np

'''
DataFrame Series
'''
# df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
# print(df.iloc[1, :])
# print(df.loc[:, ['a', 'b']])

# print(type(df.iloc[1, :]))
# print(type(df.loc[:, ['a', 'b']]))

'''
read_csv
'''
# df = pd.read_csv("./res/report.csv", delimiter=";")
# df = df.loc[:, ['reportTitle','reporter', 'reportTime', 'reportLocation']]

# with open('report.sql', mode='w', encoding="utf-8") as f:
#     for i in range(20):
#         report = df.iloc[i].values
#         sql = "insert into report(title, speaker, time, location) values('{}', '{}', '{}', '{}');\n"\
#             .format(report[0], report[1], report[2], report[3])
#         f.write(sql)

'''
read_json
'''
df = pd.read_json('./res/site.json')
print(df)