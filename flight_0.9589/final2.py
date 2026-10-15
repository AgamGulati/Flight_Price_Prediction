import numpy as np, pandas as pd
from exp2 import *    # build(), rf(), BEST hyperparameters (unchanged from the original model)
te=pd.read_csv('data/test.csv'); ss=pd.read_csv('data/sample_submission.csv')
m=build({'dur_fill':'flight'},rfp=rf(500)).fit(X,y)           # fit on ALL labeled rows
pred=m.predict(te); assert np.isfinite(pred).all()
sub=pd.DataFrame({ss.columns[0]:te.id.values,ss.columns[1]:pred})
assert list(sub.columns)==list(ss.columns) and len(sub)==len(ss) and (sub.id.values==ss.id.values).all()
sub.to_csv('submission.csv',index=False)
fb=m.regressor_.named_steps['fb']; print('test duration NaN before/after imputation:',int(te.duration.isna().sum()),int(fb.transform(te).duration.isna().sum()))
print('pred stats',pd.Series(pred).describe()[['min','max','mean','std']].round(1).to_dict())
print('checks passed: rows',len(sub),'cols',list(sub.columns),'ids match, all finite'); print(pd.read_csv('submission.csv').head())
