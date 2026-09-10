# 이 셀은 그대로 실행하세요.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

_have = {f.name for f in fm.fontManager.ttflist}
for _f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
    if _f in _have:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 30)

DATA = Path('C:/Users/user/PycharmProjects/PythonProject3/secom')
SEED = 42          # 결과를 재현하려면 항상 이 값을 쓰세요

def check(name, cond, hint=''):
    if cond:
        print('[통과] ' + name)
    else:
        print('[실패] ' + name + (('  ->  ' + str(hint)) if hint else ''))

print('폰트:', plt.rcParams['font.family'][0], '| 데이터 폴더:', DATA.exists())




print('='*24+'1번'+'='*24)

# TODO 1-1: 데이터 불러오기
#   지난주와 같은 secom_equipment.csv, signal_metadata.csv 입니다
#   timestamp 는 parse_dates 로 읽으세요
df = pd.read_csv(DATA / 'secom_equipment.csv', encoding='utf-8-sig', parse_dates=['timestamp'])
meta = pd.read_csv(DATA / 'signal_metadata.csv', encoding='utf-8-sig')

# TODO 1-2: 전처리를 함수로 만들기
#   지난주 미션 2~4 를 순서대로 담으면 됩니다.
#     1) SIG_ 로 시작하는 컬럼 모으기
#     2) 결측 비율이 50% 를 넘는 신호 골라내기
#     3) 값이 항상 같은 신호 골라내기
#     4) 2)와 3)을 set 으로 합쳐 빼고, 남은 결측은 중앙값으로 채우기
def preprocess(df):
    """신호 컬럼을 정리해 (X, keep_cols) 를 돌려준다"""
    sig_cols = [sig for sig in meta['signal_id'] if sig.startswith('SIG_')]
    miss = meta.set_index('signal_id')['missing_rate']
    high_missing = miss[miss > 0.5].index.tolist()
    const_cols = meta.loc[meta['n_unique'] == 1, 'signal_id'].tolist()
    drop_cols = list(set(high_missing) | set(const_cols))
    keep_cols = [col for col in sig_cols if col not in drop_cols]
    X = df[keep_cols].fillna(df[keep_cols].median())
    return X, keep_cols

# TODO 1-3: 함수를 써서 X, keep_cols, y 만들기 (y 는 df['label'])
X, keep_cols = preprocess(df)
y = df['label']
print (X.shape)
print (X.isna().sum().sum())
print (y.sum())
check('X 크기 1567 x 446', X is not None and X.shape == (1567, 446), None if X is None else X.shape)
check('결측 없음', X is not None and int(X.isna().sum().sum()) == 0)
check('y 불량 104건', y is not None and int(y.sum()) == 104)




print('='*24+'2번'+'='*24)

# TODO 2-1: 층화 분할 (test_size=0.3, stratify=y, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=SEED
)

# TODO 2-2: 양쪽 크기와 불량 비율 출력
#   불량 비율은 y_train.mean() * 100 으로 구합니다
print(f"학습용 데이터 크기: {len(X_train)}개 (불량 비율: {y_train.mean() * 100:.2f}%)")
print(f"평가용 데이터 크기: {len(X_test)}개 (불량 비율: {y_test.mean() * 100:.2f}%)")



check('학습용 1096장', X_train is not None and len(X_train) == 1096, None if X_train is None else len(X_train))
check('평가용 471장', X_test is not None and len(X_test) == 471)
check('학습용 불량 73건', y_train is not None and int(y_train.sum()) == 73, 'stratify=y, random_state=SEED 확인')
check('평가용 불량 31건', y_test is not None and int(y_test.sum()) == 31)



print('='*24+'3번'+'='*24)

# TODO 3-1: 평가용 전체를 0(양품)이라고 예측
pred_baseline = np.zeros(len(y_test), dtype=int)

# TODO 3-2: 정확도 계산해서 출력
#   힌트: accuracy_score(y_test, pred_baseline)
acc_baseline = accuracy_score(y_test, pred_baseline)


# TODO 3-3: 이 모델이 실제로 잡아낸 불량은 몇 건입니까?
#   평가용의 불량 31건 중 몇 건을 맞혔는지 세어보세요.
#   힌트: (pred_baseline == 1) & (y_test == 1) 이 True 인 개수
print( ((pred_baseline == 1) & (y_test == 1)).sum() )


check('베이스라인 예측 생성', pred_baseline is not None and len(pred_baseline) == 471)
check('정확도 93.42%', acc_baseline is not None and abs(acc_baseline - 0.9342) < 0.001, acc_baseline)
check('잡아낸 불량 0건', pred_baseline is not None and int(((pred_baseline == 1) & (y_test == 1)).sum()) == 0)



print('='*24+'4번'+'='*24)

# TODO 4-1: 표준화 (학습용으로만 fit)
#   scaler 를 X_train 으로 fit 한 뒤, 학습용과 평가용 둘 다 transform 하세요
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)



# TODO 4-2: 로지스틱 회귀 학습 (max_iter=3000, random_state=SEED)
#   표준화한 X_train_s 로 학습해야 합니다
logit = LogisticRegression(max_iter=3000, random_state=SEED)
logit.fit(X_train_s, y_train)

# TODO 4-3: 평가용 예측하고 정확도 출력
pred_logit = logit.predict(X_test_s)

check('표준화 완료', X_train_s is not None and abs(X_train_s.mean()) < 0.01, '학습용 평균이 0에 가까워야 함')
check('모델 학습', logit is not None and hasattr(logit, 'coef_'))
check('예측 471건', pred_logit is not None and len(pred_logit) == 471)
check('베이스라인보다 불량을 더 잡음',
      pred_logit is not None and int(((pred_logit == 1) & (y_test == 1)).sum()) > 0,
      '정확도는 낮아도 불량은 잡기 시작합니다')


print('='*24+'5번'+'='*24)

# TODO 5-1: 두 모델의 혼동행렬 출력
#  힌트: confusion_matrix(y_test, 예측).ravel() -> tn, fp, fn, tp
tn, fp, fn, tp = confusion_matrix(y_test, pred_baseline).ravel()
print(tn,fp,fn,tp)

# TODO 5-2: 평가 결과를 한 줄짜리 딕셔너리로 돌려주는 함수 만들기
#   미션 6에서 여섯 번 더 쓰게 되니 지금 함수로 만들어 둡니다.
#   키 이름은 아래 그대로 쓰세요. 자가진단과 미션 6이 이 이름을 찾습니다.
#   힌트: precision_score, recall_score, f1_score 에 zero_division=0 을 넣으세요
def evaluate(name, pred):
    """{'모델', '정확도', '정밀도', '재현율', 'F1'} 을 돌려준다"""

    return {
        '모델': name,
        '정확도': accuracy_score(y_test, pred),
        '정밀도': precision_score(y_test, pred, zero_division=0),
        '재현율': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0)
    }


# TODO 5-3: 베이스라인과 로지스틱을 이 순서로 평가해 표 만들기
#   scores.iloc[0] 이 베이스라인, scores.iloc[1] 이 로지스틱이어야 합니다
scores = pd.DataFrame([
    evaluate('베이스라인', pred_baseline),
    evaluate('로지스틱', pred_logit)
])
print (scores)



# TODO 5-4: 표를 보고 답하세요 (주석으로)
#   베이스라인의 재현율이 0인 이유는?
#   답: 맨 처음 베이스 라인의 학습을 전부 양품처리로 박아버렸기 때문에 실전에서 불량을 단 하나도 잡지 못했기 때문에 재현율이 없습니다.


check('scores 표 생성', scores is not None and len(scores) == 2)
check('베이스라인 재현율 0', scores is not None and abs(scores.iloc[0]['재현율']) < 1e-9)
check('로지스틱 재현율 > 0.15', scores is not None and scores.iloc[1]['재현율'] > 0.15, scores.iloc[1]['재현율'] if scores is not None else None)
check('로지스틱 정확도가 더 낮음', scores is not None and scores.iloc[1]['정확도'] < scores.iloc[0]['정확도'],
      '정확도가 낮은데 더 좋은 모델입니다')






print('='*24+'6번'+'='*24)

# TODO 6-1: 여섯 조합을 학습하고 evaluate() 로 평가
#   힌트: for cw in [None, 'balanced']: 로 반복하면 편합니다
#   모델 이름은 '로지스틱-기본', '로지스틱-balanced' 처럼 붙이세요.
#   결정트리와 랜덤포레스트도 같은 규칙으로 지으면 여섯 개가 됩니다.
#   (자가진단이 '결정트리-balanced' 라는 이름을 찾습니다)
#   로지스틱만 표준화한 X_train_s / X_test_s 를 씁니다
results_list = []

for cw in [None, 'balanced']:


    logit_model = LogisticRegression(
        max_iter=3000,
        class_weight=cw,
        random_state=SEED
    )
    logit_model.fit(X_train_s, y_train)
    pred = logit_model.predict(X_test_s)

    name = '로지스틱-기본' if cw is None else '로지스틱-balanced'
    results_list.append(evaluate(name, pred))


    tree_model = DecisionTreeClassifier(
        max_depth=4,
        class_weight=cw,
        random_state=SEED
    )
    tree_model.fit(X_train, y_train)
    pred = tree_model.predict(X_test)

    name = '결정트리-기본' if cw is None else '결정트리-balanced'
    results_list.append(evaluate(name, pred))


    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight=cw,
        random_state=SEED
    )
    rf_model.fit(X_train, y_train)
    pred = rf_model.predict(X_test)

    name = '랜덤포레스트-기본' if cw is None else '랜덤포레스트-balanced'
    results_list.append(evaluate(name, pred))

print(results_list)

results = pd.DataFrame(results_list)
print(results)



# TODO 6-2: 결과를 표로 만들어 재현율 내림차순으로 정렬
#   정렬한 뒤 인덱스를 다시 매기세요 (reset_index)
compare = results.sort_values('재현율', ascending=False).reset_index(drop=True)
print(compare)

# TODO 6-3: 정밀도와 재현율을 막대그래프로 나란히 비교
compare.set_index('모델')[['정밀도', '재현율']].plot(
    kind='bar',
    figsize=(10, 5)
)

plt.ylabel('점수')
plt.title('모델별 정밀도와 재현율 비교')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# TODO 6-4: 어느 모델을 공정팀에 주겠습니까? 이유와 함께 주석으로
#   답: 결정트리를 주도록 하겠습니다. 아무리 정밀도가 높아도 재현율이 떨어지면 불량이 통과하는 상황이 발생할거 같고,
#   불량을 통과 시킬바엔 차라리 검사를 여러번 진행해서 정밀도를 보완하는 방식을 사용할거 같습니다.


check('여섯 조합 학습', compare is not None and len(compare) == 6, None if compare is None else len(compare))
check('최고 재현율 0.4 이상', compare is not None and compare.iloc[0]['재현율'] > 0.4,
      compare.iloc[0]['재현율'] if compare is not None else None)
check('재현율 1위는 결정트리-balanced', compare is not None and compare.iloc[0]['모델'] == '결정트리-balanced',
      None if compare is None else compare.iloc[0]['모델'])
check('재현율이 오르면 정확도는 내려감',
      compare is not None and compare.iloc[0]['정확도'] < compare['정확도'].max(),
      '트레이드오프가 보여야 합니다')




print('='*24+'7번'+'='*24)
# TODO 7-1: 랜덤포레스트 학습
#   n_estimators=200, max_depth=6, class_weight='balanced', random_state=SEED
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight='balanced',
    random_state=SEED
)
rf.fit(X_train, y_train)

# TODO 7-2: 중요도 Top 10 (힌트: pd.Series(rf.feature_importances_, index=keep_cols))
#   importance 는 446개 전부, top10_model 은 큰 순으로 10개
importance = pd.Series(rf.feature_importances_,index=keep_cols)

top10_model = importance.sort_values(ascending=False).head(10)


# TODO 7-3: 지난주 효과크기 Top 10 과 겹치는 신호 찾기
#   힌트: 두 목록을 set 으로 만들어 교집합
week1_top10 = ['SIG_060', 'SIG_104', 'SIG_511', 'SIG_349', 'SIG_432',
               'SIG_435', 'SIG_431', 'SIG_022', 'SIG_436', 'SIG_029']
overlap = list(
    set(top10_model.index) & set(week1_top10)
)
print('겹치는 신호:', overlap)


# TODO 7-4: 중요도 Top 10 을 가로 막대그래프로 (barh)
#   지난주와 겹치는 신호만 색을 다르게 하면 한눈에 보입니다
colors = [
    'tomato' if signal in overlap else 'steelblue'
    for signal in top10_model.index
]

top10_model.sort_values().plot(kind='barh',figsize=(8, 5),color=colors)

plt.xlabel('Feature Importance')
plt.title('랜덤포레스트 중요도 Top 10')
plt.tight_layout()
plt.show()

check('랜덤포레스트 학습', rf is not None and hasattr(rf, 'feature_importances_'))
check('중요도 446개', importance is not None and len(importance) == 446)
check('Top 10 추출', top10_model is not None and len(top10_model) == 10)
check('겹치는 신호 3개', overlap is not None and len(overlap) == 3, None if overlap is None else overlap)
check('SIG_060 은 양쪽 모두', overlap is not None and 'SIG_060' in overlap)