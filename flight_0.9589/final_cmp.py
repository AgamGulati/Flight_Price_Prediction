from exp2 import *
C1=dict(min_samples_split=10,min_samples_leaf=1,max_samples=None,max_features=0.5,max_depth=None,bootstrap=True)
E=[('B0 baseline (original submitted model), 500 trees',build(rfp=rf(500))),
   ('F1 = B0 + flight-based duration imputation, 500 trees',build({'dur_fill':'flight'},rfp=rf(500))),
   ('F2 = F1 + search-best params (max_features=.5, no subsample), 500 trees',build({'dur_fill':'flight'},rfp=rf(500,**C1))),
   ('F3 = F1 + sqrt target, 500 trees',build({'dur_fill':'flight'},rfp=rf(500),target='sqrt'))]
rows=[];base=None
for n,e in E:
    r=evaluate(n,e); base=base if base is not None else np.array(r['_folds']); d=np.array(r['_folds'])-base
    r.update(dpair_mean=d.mean(),dpair_sd=d.std(ddof=1) if base is not None else 0,folds_better=int((d>0).sum()),fold_r2=[round(v,4) for v in r['_folds']])
    rows.append(r); print({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items() if k!='_folds'},flush=True)
pd.DataFrame(rows).drop(columns='_folds').to_csv('final_cmp.csv',index=False)
