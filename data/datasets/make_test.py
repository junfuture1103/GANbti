import pandas as pd
import random

# FILE_NAME = 'test2.csv'

# df = pd.read_csv(FILE_NAME)

ID_list = []

Q1_list = []
Q2_list = []
Q3_list = []
Q4_list = []
Q5_list = []
MBTI_list = []

for i in range(0,3000,1):
    ID_list.append(i)
    Q1_list.append(random.randrange(1,6))
    Q2_list.append(random.randrange(1,6))
    Q3_list.append(random.randrange(1,6))
    Q4_list.append(random.randrange(1,6))
    Q5_list.append(random.randrange(1,6))
    MBTI_list.append(random.randrange(1,17))

df = pd.DataFrame({'ID':ID_list, 'Q1':Q1_list, 'Q2':Q2_list,'Q3':Q3_list,'Q4':Q4_list,'Q5':Q5_list,'MBTI':MBTI_list})

# pd.concat([df, df2], axis = 0)
print(df)
df.to_csv("./test_mbti.csv", mode='w', index=False)
print(df)