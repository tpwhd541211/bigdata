# 이 셀은 그대로 실행하세요.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, r2_score)

_have = {f.name for f in fm.fontManager.ttflist}
for _f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
    if _f in _have:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 30)

DATA = Path('C:/Users/user/PycharmProjects/PythonProject3/etch_fdc')   # 폴더를 옮겼다면 이 줄만 고치세요
SEED = 42          # 결과를 재현하려면 항상 이 값을 쓰세요

def check(name, cond, hint=''):
    if cond:
        print('[통과] ' + name)
    else:
        print('[실패] ' + name + (('  ->  ' + str(hint)) if hint else ''))

print('폰트:', plt.rcParams['font.family'][0], '| 데이터 폴더:', DATA.exists())






print('='*24+'1번'+'='*24)

# TODO 1-1: 데이터 불러오기 (지난주와 같은 traces.csv, wafer_info.csv)
#   wafer_info 의 timestamp 는 parse_dates 로
tr = pd.read_csv(DATA / 'traces.csv', encoding='utf-8-sig')
wi = pd.read_csv(DATA / 'wafer_info.csv', encoding='utf-8-sig', parse_dates=['timestamp'])

SENSORS = ['rf_forward_W', 'rf_reflected_W', 'chamber_pressure_mTorr', 'ar_flow_sccm',
           'cf4_flow_sccm', 'esc_temp_C', 'he_backside_Torr', 'endpoint_intensity']

# TODO 1-2: 특징 추출 함수
#   지난주 미션 5, 6 을 순서대로 담으면 됩니다.
#     1) MainEtch 구간만 남기기
#     2) wafer_id 로 묶어 SENSORS 의 mean/std/min/max (32개)
#     3) 컬럼 이름 평탄화
#     4) pressure_slope 컬럼 추가 (33개)
#     5) reset_index() 해서 wafer_id 를 컬럼으로
def make_features(tr):
    main = tr[tr['step'] == 'MainEtch']

    feat = main.groupby('wafer_id')[SENSORS].agg(
        ['mean', 'std', 'min', 'max']
    )
    feat.columns = ['_'.join(c) for c in feat.columns]

    slope = main.groupby('wafer_id').apply(
        lambda g: np.polyfit(
            g['t_sec'],
            g['chamber_pressure_mTorr'],
            1
        )[0]
    )

    feat['pressure_slope'] = slope
    feat = feat.reset_index()
    return feat

feat = make_features(tr)

print(feat.shape)
print(feat.head())


# TODO 1-3: 특징을 만들고 wafer_info 와 합치기 (wafer_id 기준, 400행)
feat = make_features(tr)
data = feat.merge(wi, on='wafer_id', how='left')
print(data.shape)
print(data.head())


check('data 400행', data is not None and len(data) == 400, None if data is None else len(data))
check('특징 33개', data is not None and len([c for c in data.columns if c not in ('wafer_id','timestamp','chamber','recipe','fault_type','label')]) == 33)
check('pressure_slope 있음', data is not None and 'pressure_slope' in data.columns)







print('='*24+'2번'+'='*24)

# TODO 2-1: X, y 만들기
#   FEATURES 는 data 의 컬럼 중 아래 여섯 개를 뺀 나머지 33개입니다
#     wafer_id, timestamp, chamber, recipe, fault_type, label
#   y 는 data['label']
DROP_COLS = ['wafer_id', 'timestamp', 'chamber', 'recipe', 'fault_type', 'label']

FEATURES = [c for c in data.columns if c not in DROP_COLS]
X = data[FEATURES]
y = data['label']
print('특징 수:', len(FEATURES))
print('X shape:', X.shape)
print('y shape:', y.shape)

# TODO 2-2: 층화 분할 (test_size=0.3, stratify=y, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=SEED
)


# TODO 2-3: 양쪽 크기와 이상 비율 출력
print('X_train:', X_train.shape)
print('X_test :', X_test.shape)

print('train 이상 비율:', y_train.mean())
print('test 이상 비율 :', y_test.mean())



check('학습 280장', X_train is not None and len(X_train) == 280, None if X_train is None else len(X_train))
check('평가 120장', X_test is not None and len(X_test) == 120)
check('평가 이상 25장', y_test is not None and int(y_test.sum()) == 25, None if y_test is None else int(y_test.sum()))







print('='*24+'3번'+'='*24)

# TODO 3-1: 랜덤포레스트 학습 (n_estimators=200, random_state=SEED)
rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf.fit(X_train, y_train)

# TODO 3-2: 평가용 예측
pred = rf.predict(X_test)


# TODO 3-3: 혼동행렬과 정확도/정밀도/재현율 출력
#   힌트: confusion_matrix(y_test, pred).ravel() -> tn, fp, fn, tp
#   정밀도와 재현율 중 어느 쪽이 낮은지 보세요

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

print('혼동행렬:')
print(confusion_matrix(y_test, pred))

print('정확도 :', accuracy_score(y_test, pred))
print('정밀도 :', precision_score(y_test, pred))
print('재현율 :', recall_score(y_test, pred))

print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)


check('모델 학습', rf is not None and hasattr(rf, 'feature_importances_'))
check('예측 120건', pred is not None and len(pred) == 120)
check('정밀도 0.9 이상', pred is not None and precision_score(y_test, pred, zero_division=0) > 0.9)
check('재현율 0.6~0.9', pred is not None and 0.6 < recall_score(y_test, pred, zero_division=0) < 0.9,
      None if pred is None else round(recall_score(y_test, pred, zero_division=0), 3))







print('='*24+'4번'+'='*24)

# TODO 4-1: 놓친 웨이퍼(실제 이상인데 정상이라고 예측) 골라내기
#   힌트: (y_test == 1) & (pred == 0)
#   y_test 는 Series, pred 는 배열이라 y_test.values 로 맞춰야 할 수 있습니다
missed = (y_test.values == 1) & (pred == 0)

# TODO 4-2: 놓친 웨이퍼들의 fault_type 세어보기
#   힌트: data.loc[X_test.index[missed], 'fault_type']
missed_types = data.loc[X_test.index[missed], 'fault_type'].value_counts()

print(missed_types)

# TODO 4-3: 유형별로 몇 장 중 몇 장을 놓쳤는지 비율까지 출력
#   평가용 전체의 유형별 장수도 같이 세야 분모가 나옵니다
test_fault_types = data.loc[X_test.index, 'fault_type']

total_types = test_fault_types.value_counts()

result = pd.DataFrame({
    '전체': total_types,
    '놓침': missed_types
}).fillna(0)

result['놓침률'] = result['놓침'] / result['전체']
print(result)


check('놓친 웨이퍼 계산', missed is not None and int(missed.sum()) > 0)
check('GAS_LEAK 을 가장 많이 놓침',
      missed_types is not None and missed_types.idxmax() == 'GAS_LEAK',
      None if missed_types is None else dict(missed_types))







print('='*24+'5번'+'='*24)


# TODO 5-1: 목표를 fault_type 으로 바꿔 다시 분할
#   ym 은 data['fault_type']. 분할 조건은 미션 2와 같게 (test_size=0.3,
#   stratify=ym, random_state=SEED) 두어야 같은 웨이퍼가 평가용에 들어갑니다
ym = data['fault_type']

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X, ym,
    test_size=0.3,
    stratify=ym,
    random_state=SEED
)

# TODO 5-2: 랜덤포레스트 학습과 예측 (n_estimators=200, random_state=SEED)
rf4 = RandomForestClassifier(
    n_estimators=200,
    random_state=SEED
)

rf4.fit(Xm_train, ym_train)
pred4 = rf4.predict(Xm_test)


# TODO 5-3: 정확도와 classification_report 출력
#   힌트: classification_report(ym_test, pred4, zero_division=0)
from sklearn.metrics import accuracy_score, classification_report

print('정확도:', accuracy_score(ym_test, pred4))
print(classification_report(ym_test, pred4, zero_division=0))



check('4분류 학습', rf4 is not None and len(rf4.classes_) == 4)
check('정확도 0.90 이상', pred4 is not None and accuracy_score(ym_test, pred4) > 0.90,
      None if pred4 is None else round(accuracy_score(ym_test, pred4), 4))
check('예측 120건', pred4 is not None and len(pred4) == 120)







print('='*24+'6번'+'='*24)

# TODO 6-1: 혼동행렬 계산 (labels 순서를 고정하세요)
#   힌트: confusion_matrix(ym_test, pred4, labels=LABELS)
LABELS = ['NORMAL', 'RF_UNSTABLE', 'PRESSURE_DRIFT', 'GAS_LEAK']
cm = confusion_matrix(ym_test, pred4, labels=LABELS)

# TODO 6-2: imshow 로 그리고 각 칸에 숫자 넣기
#   힌트: ax.imshow(cm) 로 색을 칠하고, 이중 for 문에서 ax.text(j, i, cm[i, j]) 로
#         숫자를 얹습니다. 축 눈금 이름은 LABELS 로

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(len(LABELS)))
ax.set_yticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=30)
ax.set_yticklabels(LABELS)

ax.set_xlabel('예측')
ax.set_ylabel('실제')
ax.set_title('Confusion Matrix')

for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()



# TODO 6-3: 어느 유형이 어느 유형으로 잘못 갔습니까? (주석으로)
#   답:4건의 GAS_LEAK가 NORMAL로 흘러갔고, 2건의 NORMAL값이 GAS_LEAK로 흘러갔습니다.



check('혼동행렬 4x4', cm is not None and cm.shape == (4, 4))
check('RF_UNSTABLE 전부 맞힘', cm is not None and cm[1, 1] == cm[1].sum())
check('PRESSURE_DRIFT 전부 맞힘', cm is not None and cm[2, 2] == cm[2].sum())
check('GAS_LEAK 은 절반 이하', cm is not None and cm[3, 3] <= cm[3].sum() * 0.7,
      '가장 미묘한 이상이라 잘 안 잡힙니다')







print('='*24+'7번'+'='*24)

# TODO 7-1: 중요도 상위 10개
#   힌트: pd.Series(rf4.feature_importances_, index=FEATURES) 를 큰 순으로 정렬
importance = pd.Series(rf4.feature_importances_,index=FEATURES).sort_values(ascending=False)
top10 = importance.head(10)
print(top10)

# TODO 7-2: 가로 막대그래프 (barh)
plt.figure(figsize=(8, 6))

top10.sort_values().plot(kind='barh')

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()

# TODO 7-3: pressure_slope 는 몇 위입니까?
#   힌트: 정렬된 importance 의 인덱스 목록에서 위치를 찾고 1을 더하면 등수입니다
rank = importance.index.tolist().index('pressure_slope') + 1
print(rank)


check('중요도 33개', importance is not None and len(importance) == 33)
check('pressure_slope 3위 안', rank is not None and rank <= 3, rank)
check('상위 10개 추출', top10 is not None and len(top10) == 10)