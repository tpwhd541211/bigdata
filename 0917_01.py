# 이 셀은 그대로 실행하세요.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

_have = {f.name for f in fm.fontManager.ttflist}
for _f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
    if _f in _have:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 30)

DATA = Path('C:/Users/user/PycharmProjects/PythonProject3/etch_fdc')   # 폴더를 옮겼다면 이 줄만 고치세요

def check(name, cond, hint=''):
    if cond:
        print('[통과] ' + name)
    else:
        print('[실패] ' + name + (('  ->  ' + str(hint)) if hint else ''))

print('폰트:', plt.rcParams['font.family'][0], '| 데이터 폴더:', DATA.exists())




print('='*24+'1번'+'='*24)

# TODO 1-1: 두 파일 불러오기
#   traces.csv 는 tr 로, wafer_info.csv 는 wi 로
#   wafer_info 의 timestamp 는 parse_dates 로 읽으세요
tr = pd.read_csv(DATA / 'traces.csv', encoding='utf-8-sig')
wi = pd.read_csv(DATA / 'wafer_info.csv', encoding='utf-8-sig', parse_dates=['timestamp'])
print(tr)
print(wi)
# TODO 1-2: 각각의 크기, 웨이퍼 수, 웨이퍼 한 장당 행 수를 출력
#   힌트: tr.wafer_id.nunique() 로 웨이퍼 수를 세고, 전체 행 수를 그것으로 나누세요
print(len(tr))
print(tr.wafer_id.nunique())
print (len(tr) / tr.wafer_id.nunique())


# TODO 1-3: 이상 유형(fault_type)별 웨이퍼 수
#   힌트: value_counts(). 자가진단이 fault_counts.get('NORMAL') 을 봅니다
fault_counts = wi['fault_type'].value_counts()
print(fault_counts)

check('traces 24000행 11열', tr is not None and tr.shape == (24000, 11), None if tr is None else tr.shape)
check('웨이퍼 400장', wi is not None and len(wi) == 400)
check('장당 60행', tr is not None and len(tr) // tr.wafer_id.nunique() == 60)
check('정상 316장', fault_counts is not None and fault_counts.get('NORMAL') == 316, None if fault_counts is None else dict(fault_counts))




print('='*24+'2번'+'='*24)

# TODO 2-1: WF0001 의 데이터만 뽑기 (60행이어야 합니다)
one = tr[tr['wafer_id'] == 'WF0001']
print(one)

# TODO 2-2: 4행 2열 subplot 에 센서 8개를 각각 그리기
#   x축은 t_sec, 각 칸에 센서 이름을 제목으로
#   힌트: plt.subplots(4, 2, sharex=True) 가 돌려주는 axes 는 4x2 배열입니다.
#         axes.ravel() 로 8칸짜리 한 줄로 편 다음 SENSORS 와 zip 하세요
SENSORS = ['rf_forward_W', 'rf_reflected_W', 'chamber_pressure_mTorr', 'ar_flow_sccm',
           'cf4_flow_sccm', 'esc_temp_C', 'he_backside_Torr', 'endpoint_intensity']

fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)

for ax, sensor in zip(axes.ravel(), SENSORS):
    ax.plot(one['t_sec'], one[sensor])
    ax.set_title(sensor)

plt.tight_layout()
#plt.show()


check('WF0001 60행', one is not None and len(one) == 60, None if one is None else len(one))
check('센서 8개 지정', len(SENSORS) == 8)






print('='*24+'3번'+'='*24)

# TODO 3-1: 스텝별 지속 시간 (웨이퍼 한 장 기준)
#   one 을 step 으로 묶어 행 수를 세면 됩니다. 1행이 1초입니다.
#   공정 순서를 지키려면 groupby 에 sort=False 를 주세요
step_len = one.groupby('step',sort=False).size()
print(step_len)


# TODO 3-2: 스텝별 센서 평균 (전체 웨이퍼)
#   전체 tr 을 step 으로 묶고 SENSORS 컬럼의 평균을 내면 3행짜리 표가 됩니다
by_step = tr.groupby('step')[SENSORS].mean()
print(by_step)

check('Stabilize 10초', step_len is not None and step_len.get('Stabilize') == 10)
check('MainEtch 40초', step_len is not None and step_len.get('MainEtch') == 40)
check('Overetch 10초', step_len is not None and step_len.get('Overetch') == 10)
check('스텝별 평균 3행', by_step is not None and len(by_step) == 3)





print('='*24+'4번'+'='*24)

# TODO 4-1: 유형별로 웨이퍼 하나씩 고르기
#   힌트: wi[wi.fault_type == 'RF_UNSTABLE'].wafer_id.iloc[0]
#   RF_UNSTABLE, PRESSURE_DRIFT, GAS_LEAK 세 개를 {유형: wafer_id} 딕셔너리로
target_types = ['RF_UNSTABLE', 'PRESSURE_DRIFT', 'GAS_LEAK']
picks = {f_type: wi[wi.fault_type == f_type].wafer_id.iloc[0] for f_type in target_types}
print(picks)


# TODO 4-2: 유형마다 알맞은 센서를 골라 정상과 겹쳐 그리기
#   1행 3열 subplot 을 쓰고, MainEtch 구간만 그려도 좋습니다
#   센서는 이상 이름에서 짐작할 수 있습니다. RF 가 불안하면 어느 신호를 봐야 할까요.
#   정상 웨이퍼와 반드시 같은 축에 겹쳐 그리세요. 따로 그리면 y축 범위가 달라
#   차이가 있어도 안 보입니다

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

sensor_map = {
    'RF_UNSTABLE': 'rf_forward_W',
    'PRESSURE_DRIFT': 'chamber_pressure_mTorr',
    'GAS_LEAK': 'ar_flow_sccm'
}

normal_id = wi[wi['fault_type'] == 'NORMAL']['wafer_id'].iloc[0]

for ax, f_type in zip(axes, target_types):
    sensor = sensor_map[f_type]

    fault_data = tr[
        (tr['wafer_id'] == picks[f_type]) &
        (tr['step'] == 'MainEtch')
    ]

    normal_data = tr[
        (tr['wafer_id'] == normal_id) &
        (tr['step'] == 'MainEtch')
    ]

    ax.plot(normal_data['t_sec'], normal_data[sensor],
            label='NORMAL', color='blue')
    ax.plot(fault_data['t_sec'], fault_data[sensor],
            label=f_type, color='red')

    ax.set_title(f'{f_type} - {sensor}')
    ax.set_xlabel('t_sec')
    ax.set_ylabel(sensor)
    ax.legend()

plt.tight_layout()
plt.show()

check('세 유형 선택', picks is not None and len(picks) == 3)
check('선택한 웨이퍼가 실제 그 유형인가',
      picks is not None and all(wi.set_index('wafer_id').fault_type[v] == k for k, v in picks.items()))







print('='*24+'5번'+'='*24)

# TODO 5-1: MainEtch 구간만 뽑기 (16000행이어야 합니다)
main = None

# TODO 5-2: 웨이퍼별로 센서 8개의 mean/std/min/max 구하기
#   힌트: main.groupby('wafer_id')[SENSORS].agg(['mean','std','min','max'])
feat = None

# TODO 5-3: 컬럼 이름을 rf_forward_W_mean 처럼 한 층으로 펴기
#   힌트: feat.columns = ['_'.join(c) for c in feat.columns]


check('MainEtch 16000행', main is not None and len(main) == 16000, None if main is None else len(main))
check('특징 400행 32열', feat is not None and feat.shape == (400, 32), None if feat is None else feat.shape)
check('컬럼명 평탄화', feat is not None and 'rf_reflected_W_max' in feat.columns)


# TODO 6-1: 웨이퍼별 압력 기울기
#   힌트: main.groupby('wafer_id').apply(lambda g: np.polyfit(g.t_sec, g.chamber_pressure_mTorr, 1)[0])
slope = None

# TODO 6-2: slope를 feat 에  pressure_slope 컬럼으로 붙이고 wafer_info 와 합치기
#   feat 의 인덱스가 wafer_id 이므로 reset_index() 한 뒤 wi 와 merge 하세요
#   결과는 400행이고 fault_type 컬럼이 들어 있어야 합니다
data = None

# TODO 6-3: 이상 유형별 기울기 평균 출력
#   PRESSURE_DRIFT 만 튀는지 확인하세요


check('기울기 400개', slope is not None and len(slope) == 400)
check('data 결합', data is not None and 'fault_type' in data.columns and len(data) == 400)
check('DRIFT 기울기가 가장 큼',
      data is not None and data.groupby('fault_type').pressure_slope.mean().idxmax() == 'PRESSURE_DRIFT')


# TODO 7-1: 세 특징을 이상 유형별 박스플롯으로 (1행 3열)
#   ORDER 순서로 그리면 표와 그림을 나란히 읽기 좋습니다
PICK = ['rf_reflected_W_max', 'pressure_slope', 'cf4_flow_sccm_mean']
ORDER = ['NORMAL', 'RF_UNSTABLE', 'PRESSURE_DRIFT', 'GAS_LEAK']


# TODO 7-2: 유형별 평균을 표로
#   fault_type 으로 묶어 PICK 세 컬럼의 평균을 내세요 (4행 3열)
summary = None

# TODO 7-3: GAS_LEAK 은 정상과 얼마나 겹칩니까? (주석으로)
#   답:


check('요약표 4행', summary is not None and len(summary) == 4)
check('RF 반사전력 최대는 RF_UNSTABLE 이 1위',
      summary is not None and summary.rf_reflected_W_max.idxmax() == 'RF_UNSTABLE')
check('CF4 평균은 GAS_LEAK 이 최소',
      summary is not None and summary.cf4_flow_sccm_mean.idxmin() == 'GAS_LEAK')