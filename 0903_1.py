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

DATA = Path('C:/Users/user/PycharmProjects/PythonProject3/secom')   # 폴더를 옮겼다면 이 줄만 고치세요

def check(name, cond, hint=''):
    if cond:
        print('[통과] ' + name)
    else:
        print('[실패] ' + name + (('  ->  ' + str(hint)) if hint else ''))

print('폰트:', plt.rcParams['font.family'][0], '| 데이터 폴더:', DATA.exists())





print('='*24+'1번'+'='*24)

# TODO 1-1: 두 파일 불러오기
#   힌트: pd.read_csv(DATA / '파일명', encoding='utf-8-sig', parse_dates=['timestamp'])
df = pd.read_csv(DATA / 'secom_equipment.csv', encoding='utf-8-sig', parse_dates=['timestamp'])
meta = pd.read_csv(DATA / 'signal_metadata.csv', encoding='utf-8-sig')

# TODO 1-2: 두 데이터의 크기 출력
print(df.shape)
print(meta.shape)

# TODO 1-3: 불량 건수와 비율(%)
n_fail = df[df['label'] == True].shape[0]
print(n_fail)
n_amount = df['wafer_id'].count()
print(n_amount)
fail_rate = round(( n_fail / n_amount ) * 100, 2)
print(f"{fail_rate}%")


check('데이터 로드', df is not None and meta is not None, '두 파일을 읽어오세요')
check('df 크기 1567 x 594', df is not None and df.shape == (1567, 594), None if df is None else df.shape)
check('meta 590행', meta is not None and len(meta) == 590)
check('불량 104건', n_fail == 104, n_fail)
check('불량률 6.64%', fail_rate is not None and abs(fail_rate - 6.64) < 0.05, fail_rate)







print('='*24+'2번'+'='*24)

# TODO 2-1: SIG_ 로 시작하는 컬럼 이름만 모으기
sig_cols = [sig for sig in meta['signal_id'] if sig.startswith('SIG_')]
print(sig_cols)

# TODO 2-2: 신호별 결측 비율 (힌트: .isna().mean())
miss = meta.set_index('signal_id')['missing_rate']
print(miss)

# TODO 2-3: 결측이 많은 상위 10개 출력
print(miss.sort_values(ascending=False).head(10))

# TODO 2-4: 결측 50%를 넘는 신호 이름 리스트
high_missing = miss[miss > 0.5].index.tolist()
print(high_missing)

check('신호 590개', sig_cols is not None and len(sig_cols) == 590, None if sig_cols is None else len(sig_cols))
check('결측 비율 계산', miss is not None and abs(miss.mean() - 0.0454) < 0.001)
check('50% 초과 28개', high_missing is not None and len(high_missing) == 28, None if high_missing is None else len(high_missing))





print('='*24+'3번'+'='*24)

# TODO 3-1: 값이 항상 같은 신호 찾기 (힌트: .nunique())
const_cols = meta.loc[meta['n_unique'] == 1, 'signal_id'].tolist()

# TODO 3-2: 제거 목록 합치기 (중복 없이)
drop_cols = list(set(high_missing) | set(const_cols))

# TODO 3-3: 남는 신호
keep_cols = [col for col in sig_cols if col not in drop_cols]

check('상수 신호 116개', const_cols is not None and len(const_cols) == 116, None if const_cols is None else len(const_cols))
check('제거 144개', drop_cols is not None and len(drop_cols) == 144, '결측 28 + 상수 116')
check('사용 446개', keep_cols is not None and len(keep_cols) == 446, None if keep_cols is None else len(keep_cols))





print('='*24+'4번'+'='*24)


# TODO 4-1: 중앙값으로 채우기
#   힌트: X = df[keep_cols].fillna( ... .median())
X = df[keep_cols].fillna(df[keep_cols].median())

# TODO 4-2: 남은 결측 개수와 X 크기 출력
print( meta.set_index('signal_id')['missing_rate'].isna().sum() , X.shape)


check('X 생성', X is not None)
check('X 크기 1567 x 446', X is not None and X.shape == (1567, 446), None if X is None else X.shape)
check('결측 0개', X is not None and int(X.isna().sum().sum()) == 0)







print('='*24+'5번'+'='*24)
# TODO 5-1: 남은 신호의 메타데이터만 추출 (힌트: .isin())

meta_keep = meta[meta['signal_id'].isin(keep_cols)]
print(meta_keep)

# TODO 5-2: 모듈별 개수 세기 (힌트: value_counts)
by_module = meta_keep['module_kr'].value_counts()
print(by_module)

# TODO 5-3: 막대그래프. 제목과 축 이름을 꼭 넣으세요
plt.figure(figsize=(8, 5))
by_module.plot(kind='bar')
plt.title('모듈별 신호 개수')
plt.xlabel('모듈')
plt.ylabel('신호 개수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




check('meta_keep 446행', meta_keep is not None and len(meta_keep) == 446, None if meta_keep is None else len(meta_keep))
check('모듈 8종', by_module is not None and len(by_module) == 8)









print('='*24+'6번'+'='*24)
# TODO 6-1: 불량 마스크
is_fail = df['pass_fail'] == 'FAIL'
print(is_fail)

# TODO 6-2: 그룹별 평균
mean_fail = None
mean_pass = None

# TODO 6-3: 효과크기 (절댓값, 큰 순서로 정렬)
effect = None

# TODO 6-4: Top 10 을 meta 와 합쳐 표로 출력
top10 = None


# TODO 6-5: 1등 신호의 양품 vs 불량 박스플롯


check('불량 104건', is_fail is not None and int(is_fail.sum()) == 104)
check('효과크기 계산', effect is not None and len(effect.dropna()) > 400)
check('1위 SIG_060', top10 is not None and top10.index[0] == 'SIG_060', None if top10 is None else top10.index[0])
check('2위 SIG_104', top10 is not None and top10.index[1] == 'SIG_104')
check('1위 효과크기 0.627', top10 is not None and abs(top10.iloc[0] - 0.6265) < 0.01)









print('='*24+'7번'+'='*24)

# TODO 7-1: 월 단위 컬럼 만들기
#   힌트: df['timestamp'].dt.to_period('M')
#   (아래에 직접 작성하세요)


# TODO 7-2: 월별 처리량 / 불량수 / 불량률
monthly = None

# TODO 7-3: 월별 불량률 선그래프


check('month 컬럼', df is not None and 'month' in df.columns and df['month'].notna().all())
check('4개월 집계', monthly is not None and len(monthly) == 4, '2008년 7~10월')
check('7월이 가장 높음', monthly is not None and monthly['불량률'].idxmax() == monthly.index[0])



print('='*50)