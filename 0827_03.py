# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import pandas as pd

df = pd.read_csv("penguins.csv")


# 지시사항을 참고하여 코드를 작성하세요.
def total(df, s):
    df = pd.read_csv('penguins.csv')
    print(df)
    s = 30
    answer = []

    # 데이터가 없는 행은 제외하는 코드
    pendqueen = df.dropna()
    print(pendqueen)

    # 'sex' 가 'MALE'이고, 'bill_length_mm'가 s보다 큰 펭귄을 구하는 코드를 작성하세요
    poungqueen = pendqueen[(pendqueen['sex'] == 'MALE') & (pendqueen['bill_length_mm'] >=s )]
    print(poungqueen)

    # 부리의 길이가 n보다 큰 펭귄의 수를 answer 리스트의 원소로 추가하세요.
    answer = poungqueen[()]
    print(answer)

    # 부리의 깊이의 평균을 answer 리스트의 원소로 추가하세요.



    return answer


# 값을 확인하기 위한 코드입니다. 값을 변경해가며 테스트해 보세요!
print(total(df, 30))