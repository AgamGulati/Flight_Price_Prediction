import time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

D='/mnt/user-data/uploads/'; O='/mnt/user-data/outputs/'
tr=pd.read_csv(D+'vehicle_train_10000.csv'); te=pd.read_csv(D+'vehicle_test.csv'); sub=pd.read_csv(D+'vehicle_sample_submission.csv')
T='vehicle_class'; print(tr.shape,te.shape,sub.shape)
print('missing train/test:',tr.isna().sum().sum(),te.isna().sum().sum(),'| dups:',tr.drop(columns='vehicle_id').duplicated().sum())
print(tr[T].value_counts().sort_index()); print((tr[T].value_counts(normalize=True)).round(3).to_dict())
feats=[c for c in tr.columns if c not in ('vehicle_id',T)]
# train/test drift check
from scipy.stats import ks_2samp
print('max KS train vs test:',max(ks_2samp(tr[c],te[c]).statistic for c in feats).round(3))
# id check: correlation of id with target
print('id vs class corr:',np.corrcoef(tr.vehicle_id,tr[T])[0,1].round(3))
X,y=tr[feats],tr[T]
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
sc=['accuracy','f1_macro','f1_weighted','balanced_accuracy']
imp=('imp',SimpleImputer(strategy='median'))
cands={
 'LogReg (untuned)':(Pipeline([imp,('sc',StandardScaler()),('m',LogisticRegression(max_iter=5000))]),None),
 'LogReg (tuned)':(Pipeline([imp,('sc',StandardScaler()),('m',LogisticRegression(max_iter=5000,solver='saga'))]),
   {'m__C':[.01,.1,1,10,100],'m__penalty':['l1','l2'],'m__class_weight':[None,'balanced']}),
 'GaussianNB (untuned)':(Pipeline([imp,('m',GaussianNB())]),None),
 'GaussianNB (tuned)':(Pipeline([imp,('m',GaussianNB())]),{'m__var_smoothing':np.logspace(-11,-5,7)}),
 'QDA Bayes (untuned)':(Pipeline([imp,('m',QuadraticDiscriminantAnalysis())]),None),
 'QDA Bayes (tuned)':(Pipeline([imp,('m',QuadraticDiscriminantAnalysis())]),{'m__reg_param':[0,.01,.05,.1,.3,.5]}),
 'Tree (untuned)':(Pipeline([imp,('m',DecisionTreeClassifier(random_state=42))]),None),
 'Tree (tuned)':(Pipeline([imp,('m',DecisionTreeClassifier(random_state=42))]),
   {'m__criterion':['gini','entropy'],'m__max_depth':[3,5,8,12,None],'m__min_samples_leaf':[1,5,20,50],'m__ccp_alpha':[0,1e-4,1e-3,5e-3]}),
}
rows=[];best={}
for n,(p,g) in cands.items():
    t=time.time()
    if g:
        gs=GridSearchCV(p,g,cv=cv,scoring='f1_macro',n_jobs=-1).fit(Xt,yt); est=gs.best_estimator_; bp=gs.best_params_
    else: est=p; bp={}
    r=cross_validate(est,Xt,yt,cv=cv,scoring=sc,n_jobs=-1)
    best[n]=est
    rows.append(dict(model=n,params=str(bp),acc=r['test_accuracy'].mean(),macroF1=r['test_f1_macro'].mean(),wF1=r['test_f1_weighted'].mean(),
      balAcc=r['test_balanced_accuracy'].mean(),std_macroF1=r['test_f1_macro'].std(),secs=round(time.time()-t,1)))
res=pd.DataFrame(rows).round(4).sort_values('macroF1',ascending=False)
pd.set_option('display.width',250,'display.max_colwidth',90); print(res.to_string(index=False))
res.to_csv(O+'model_comparison.csv',index=False)
# pick best (tie within 0.002 -> prefer simpler order: LogReg, NB, Tree, QDA)
top=res.macroF1.max(); ok=res[res.macroF1>=top-.002].model.tolist()
pref=[m for m in ['LogReg (tuned)','LogReg (untuned)','GaussianNB (tuned)','GaussianNB (untuned)','Tree (tuned)','QDA Bayes (tuned)','QDA Bayes (untuned)','Tree (untuned)'] if m in ok]
win=pref[0]; print('Selected:',win)
from sklearn.base import clone
m=clone(best[win]).fit(Xt,yt); pv=m.predict(Xv)
print('Hold-out 20% report'); print(classification_report(yv,pv,digits=4)); print(confusion_matrix(yv,pv))
final=clone(best[win]).fit(X,y)
pred=final.predict(te[feats])
out=pd.DataFrame({'vehicle_id':te.vehicle_id,'vehicle_class':pred})
out.to_csv(O+'vehicle_submission.csv',index=False)
chk=pd.read_csv(O+'vehicle_submission.csv')
print('rows',len(chk),'cols',list(chk.columns)==['vehicle_id','vehicle_class'],'ids match',(chk.vehicle_id.values==sub.vehicle_id.values).all(),
 'valid labels',set(chk.vehicle_class)<=set(y.unique())); print(chk.vehicle_class.value_counts().sort_index())
