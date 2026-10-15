import numpy as np, pandas as pd, json
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score
from fe import *
tr=pd.read_csv('data/train.csv'); X=tr.drop(columns='price'); y=tr.price
kf=KFold(5,shuffle=True,random_state=42)
best=dict(bootstrap=True,max_depth=None,max_features=0.8,max_samples=0.85,min_samples_leaf=1,min_samples_split=10)
FB_KW=dict(airline_fill='prefix',dur_fill='group',miss_flag=False,flight_mode='split',extra=False)
p=make_pipe(dict(n_estimators=150,random_state=42,n_jobs=-1,**best),log=True,**FB_KW)
r=cross_validate(p,X,y,cv=kf,scoring={'r2':'r2'},return_train_score=True)
print('train R2',r['train_r2'].round(4),r['train_r2'].mean()); print('val R2',r['test_r2'].round(4),r['test_r2'].mean(),r['test_r2'].std())
oof=cross_val_predict(p,X,y,cv=kf)
tr['oof']=oof; tr['err']=oof-y
print('overall oof R2',r2_score(y,oof))
for c,g in tr.groupby('flight_class'): print(c,'n',len(g),'R2',round(r2_score(g.price,g.oof),4),'RMSE',round(np.sqrt((g.err**2).mean())),'share of SSE',round((g.err**2).sum()/(tr.err**2).sum(),3))
# R2 without top 1% price rows (test-set variance sensitivity)
q=tr.price.quantile(.99); m=tr.price<=q; print('R2 excl top1% prices',r2_score(tr.price[m],tr.oof[m]))
print('top-20 abs errors share of SSE',(tr.err.abs().nlargest(20)**2).sum()/(tr.err**2).sum())
print(tr.assign(ae=tr.err.abs()).groupby('stops').ae.mean().round(0).to_dict())
# duplicates (features excl id) – do duplicate rows share price?
cols=[c for c in tr.columns if c not in ('id','price','oof','err')]
d=tr[tr.duplicated(cols,keep=False)].groupby(cols,dropna=False).price.agg(['count','std','mean'])
print('dup groups',len(d),'rows',int(d['count'].sum()),'mean within-group price std',d['std'].mean(),'mean price',d['mean'].mean())
print('price std by (route,class) vs days_left corr:',tr[['days_left','price']].corr().iloc[0,1])
tr['route']=tr.source_city+'>'+tr.destination_city
print(tr.groupby('route').price.mean().describe())
print('dur missing pct by class', tr.assign(m=tr.duration.isna()).groupby('flight_class').m.mean().round(3).to_dict())
print('dur vs price corr (non-missing):',tr[['duration','price']].corr().iloc[0,1])
