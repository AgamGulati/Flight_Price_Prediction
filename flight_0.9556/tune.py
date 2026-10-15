import numpy as np, pandas as pd, time, json
from scipy.stats import randint
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from fe import *
tr=pd.read_csv('data/train.csv'); X=tr.drop(columns='price'); y=tr.price
kf=KFold(5,shuffle=True,random_state=42)
sc={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'}
FB_KW=dict(airline_fill='prefix',dur_fill='group',miss_flag=False,flight_mode='split',extra=False)
def report(name,p):
    r=cross_validate(p,X,y,cv=kf,scoring=sc,return_train_score=True)
    d=dict(model=name,rmse=-r['test_rmse'].mean(),rmse_sd=r['test_rmse'].std(),mae=-r['test_mae'].mean(),r2=r['test_r2'].mean(),train_rmse=-r['train_rmse'].mean())
    print(d,flush=True); return d
res=[]
res.append(report('baseline RF500 raw target',make_pipe(dict(n_estimators=500,random_state=42,n_jobs=-1),**FB_KW)))
res.append(report('baseline RF500 log target',make_pipe(dict(n_estimators=500,random_state=42,n_jobs=-1),log=True,**FB_KW)))
# search (parameters routed to the RF inside TransformedTargetRegressor)
est=make_pipe(dict(n_estimators=150,random_state=42,n_jobs=-1),log=True,**FB_KW)
space={'regressor__rf__max_depth':[None,16,24,32],
       'regressor__rf__max_features':[0.4,0.6,0.8,1.0],
       'regressor__rf__min_samples_split':randint(2,12),
       'regressor__rf__min_samples_leaf':randint(1,5),
       'regressor__rf__bootstrap':[True,False],
       'regressor__rf__max_samples':[None]}
# max_samples only valid if bootstrap=True -> use two grids
space=[dict(space,**{'regressor__rf__bootstrap':[True],'regressor__rf__max_samples':[None,0.7,0.85]}),
       dict(space,**{'regressor__rf__bootstrap':[False],'regressor__rf__max_samples':[None]})]
# score on the ORIGINAL price scale (RMSE) even though target is log-transformed internally
t=time.time()
rs=RandomizedSearchCV(est,space,n_iter=14,cv=kf,scoring='neg_root_mean_squared_error',random_state=42,n_jobs=1,refit=False,return_train_score=False)
rs.fit(X,y)
print('search sec',time.time()-t)
cv=pd.DataFrame(rs.cv_results_).sort_values('rank_test_score')
cv[['params','mean_test_score','std_test_score']].to_csv('search_results.csv',index=False)
print(cv[['mean_test_score','std_test_score','params']].head(6).to_string())
best={k.replace('regressor__rf__',''):v for k,v in rs.best_params_.items()}
json.dump(best,open('best.json','w'),default=str); print(best)
bp=dict(n_estimators=500,random_state=42,n_jobs=-1,**best)
res.append(report('tuned RF500 log target',make_pipe(bp,log=True,**FB_KW)))
pd.DataFrame(res).to_csv('comparison.csv',index=False)
