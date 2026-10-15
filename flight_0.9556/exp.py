import numpy as np, pandas as pd, time, sys
from sklearn.model_selection import KFold, cross_validate
from fe import *
tr=pd.read_csv('data/train.csv'); X=tr.drop(columns='price'); y=tr.price
kf=KFold(5,shuffle=True,random_state=42)
sc={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'}
base=dict(n_estimators=100,random_state=42,n_jobs=-1)
exps=[
 ('E0 raw baseline: airline MISSING, dur median, flight raw, no FE', dict(airline_fill='missing',dur_fill='median',miss_flag=False,flight_mode='raw',extra=False)),
 ('E1 + flight split (prefix,num)', dict(airline_fill='missing',dur_fill='median',miss_flag=False,flight_mode='split',extra=False)),
 ('E2 + airline from flight prefix', dict(airline_fill='prefix',dur_fill='median',miss_flag=False,flight_mode='split',extra=False)),
 ('E3 + duration group-median (route,stops)', dict(airline_fill='prefix',dur_fill='group',miss_flag=False,flight_mode='split',extra=False)),
 ('E4 + missing-duration flag', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='split',extra=False)),
 ('E5 + extra FE (route, ordinal times, class*days)', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='split',extra=True)),
 ('E6 E4 + id included', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='split',extra=False,use_id=True)),
 ('E7 E4 with flight raw + split', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='both',extra=False)),
 ('E8 E4 with one-hot encoding', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='split',extra=False,onehot=True)),
 ('E9 E4 log1p target', dict(airline_fill='prefix',dur_fill='group',miss_flag=True,flight_mode='split',extra=False,log=True)),
]
rows=[]
for name,kw in exps:
    t=time.time(); oh=kw.pop('onehot',False); lg=kw.pop('log',False)
    p=make_pipe(base,onehot=oh,log=lg,**kw)
    r=cross_validate(p,X,y,cv=kf,scoring=sc)
    rows.append(dict(exp=name,rmse=-r['test_rmse'].mean(),rmse_sd=r['test_rmse'].std(),mae=-r['test_mae'].mean(),r2=r['test_r2'].mean(),sec=time.time()-t))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('exp_table.csv',index=False)
