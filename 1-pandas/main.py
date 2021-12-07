import pandas as pd
import os

df = pd.read_csv("./pandas/report.csv", delimiter=";")
df = df.loc[:, ['reportTitle','reporter', 'reportTime', 'reportLocation']]

with open('report.sql', mode='w', encoding="utf-8") as f:
    for i in range(20):
        report = df.iloc[i].values
        sql = "insert into report(title, speaker, time, location) values('{}', '{}', '{}', '{}');\n"\
            .format(report[0], report[1], report[2], report[3])
        f.write(sql)