import numpy as np, pandas as pd, time, json, sys
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from fe import *
tr=pd.read_csv('data/train.csv'); X=tr.drop(columns='price'); y=tr.price.values
cols=[c for c in X.columns if c!='id']
grp=X.groupby(cols,dropna=False).ngroup().values       # identical-feature rows stay in the same fold
folds=list(GroupKFold(5,shuffle=True,random_state=42).split(X,y,grp))
BEST=dict(bootstrap=True,max_depth=None,max_features=0.8,max_samples=0.85,min_samples_leaf=1,min_samples_split=10)
def rf(n=150,**kw): return dict(n_estimators=n,random_state=42,n_jobs=-1,**{**BEST,**kw})
BASE_FB=dict(airline_fill='prefix',dur_fill='group',miss_flag=False,flight_mode='split',extra=False)
def build(fb=None,rfp=None,target='log',onehot=False):
    p=make_pipe(rfp or rf(),onehot=onehot,**{**BASE_FB,**(fb or {})})
    if target=='log': return TransformedTargetRegressor(p,func=np.log1p,inverse_func=np.expm1)
    if target=='sqrt': return TransformedTargetRegressor(p,func=np.sqrt,inverse_func=np.square)
    return p
class ClassSplit(BaseEstimator,RegressorMixin):
    def __init__(self,est): self.est=est
    def fit(self,X,y):
        self.m_={}
        for c in ['Business','Economy']:
            i=(X.flight_class==c).values; self.m_[c]=clone(self.est).fit(X[i],y[i])
        return self
    def predict(self,X):
        out=np.zeros(len(X))
        for c,m in self.m_.items():
            i=(X.flight_class==c).values
            if i.any(): out[i]=m.predict(X[i])
        return out
def evaluate(name,est):
    t=time.time(); tr_r2=[];va=[];rm=[];ma=[]
    for a,b in folds:
        m=clone(est).fit(X.iloc[a],y[a])
        tr_r2.append(r2_score(y[a],m.predict(X.iloc[a]))); p=m.predict(X.iloc[b])
        va.append(r2_score(y[b],p)); rm.append(mean_squared_error(y[b],p)**.5); ma.append(mean_absolute_error(y[b],p))
    return dict(exp=name,train_r2=np.mean(tr_r2),cv_r2=np.mean(va),cv_r2_sd=np.std(va),rmse=np.mean(rm),mae=np.mean(ma),sec=time.time()-t,_folds=va)
if __name__=='__main__':
    E=[('B0 baseline (current tuned RF, log, dur=route+stops median)',build()),
       ('X1 duration imputed from flight+route median',build({'dur_fill':'flight'})),
       ('X2 X1 + route categorical',build({'dur_fill':'flight','extras':('route',)})),
       ('X3 X1 + ordinal dep/arr times',build({'dur_fill':'flight','extras':('ord',)})),
       ('X4 X1 + flight prefix only (drop flight number)',build({'dur_fill':'flight','flight_mode':'prefix'})),
       ('X5 X1 + sqrt target',build({'dur_fill':'flight'},target='sqrt')),
       ('X6 X1 + raw target',build({'dur_fill':'flight'},target='raw')),
       ('X7 X1 + separate forests per class',ClassSplit(build({'dur_fill':'flight'}))),
       ('X8 X1 + one-hot categoricals',build({'dur_fill':'flight'},onehot=True)),
      ]
    rows=[];base=None
    for n,e in E:
        r=evaluate(n,e); 
        if base is None: base=np.array(r['_folds'])
        d=np.array(r['_folds'])-base
        r.update(dpair_mean=d.mean(),dpair_sd=d.std(ddof=1) if len(d)>1 else 0,folds_better=int((d>0).sum()))
        rows.append(r); print({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items() if k!='_folds'},flush=True)
    pd.DataFrame(rows).drop(columns='_folds').to_csv('exp2_table.csv',index=False)
