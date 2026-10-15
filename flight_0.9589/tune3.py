import pandas as pd, numpy as np, time
from sklearn.model_selection import RandomizedSearchCV
from exp2 import *
est=build({'dur_fill':'flight'},rfp=rf(150))
P='regressor__rf__'
space=[{P+'bootstrap':[True],P+'max_samples':[None,0.6,0.75,0.85],P+'max_features':[0.4,0.5,0.65,0.8,1.0],
        P+'min_samples_leaf':[1,2,3,5],P+'min_samples_split':[2,5,10,20],P+'max_depth':[None,24]},
       {P+'bootstrap':[False],P+'max_samples':[None],P+'max_features':[0.4,0.5,0.65,0.8],
        P+'min_samples_leaf':[1,2,3],P+'min_samples_split':[2,5,10,20],P+'max_depth':[None,24]}]
t=time.time()
rs=RandomizedSearchCV(est,space,n_iter=18,cv=folds,scoring='r2',random_state=42,n_jobs=1,refit=False,return_train_score=True)
rs.fit(X,y); print('sec',time.time()-t)
r=pd.DataFrame(rs.cv_results_).sort_values('rank_test_score')
r['params']=r.params.apply(lambda d:{k.replace(P,''):v for k,v in d.items()})
r[['rank_test_score','mean_train_score','mean_test_score','std_test_score','params']].to_csv('search3.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',200)
print(r[['mean_train_score','mean_test_score','std_test_score','params']].head(8).to_string())
