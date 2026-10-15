import numpy as np, pandas as pd, json
from fe import *
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); ss=pd.read_csv('data/sample_submission.csv')
X=tr.drop(columns='price'); y=tr.price
best=json.load(open('best.json')); best['max_samples']=None if best['max_samples']=='None' else best['max_samples']
FB_KW=dict(airline_fill='prefix',dur_fill='group',miss_flag=False,flight_mode='split',extra=False)
p=make_pipe(dict(n_estimators=500,random_state=42,n_jobs=-1,**best),log=True,**FB_KW).fit(X,y)
Xt=p.regressor_.named_steps['fb'].transform(te); Xtr=p.regressor_.named_steps['fb'].transform(X)
print('train',tr.shape,'test',te.shape,'| NaNs before (train/test):',int(X.isna().sum().sum()),int(te.isna().sum().sum()),'| after FE:',int(Xtr.isna().sum().sum()),int(Xt.isna().sum().sum()))
pred=p.predict(te); assert np.isfinite(pred).all()
sub=pd.DataFrame({ss.columns[0]:te.id.values, ss.columns[1]:pred})
assert list(sub.columns)==list(ss.columns) and len(sub)==len(ss) and (sub.id.values==ss.id.values).all() and sub.price.notna().all()
sub.to_csv('submission.csv',index=False)
print(pd.Series(pred).describe()[['min','max','mean','std']].round(1).to_dict()); print('checks passed'); print(pd.read_csv('submission.csv').head())
