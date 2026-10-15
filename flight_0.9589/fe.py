import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

CATS=['airline','source_city','departure_time','stops','arrival_time','destination_city','flight_class']
TOD={'Early_Morning':0,'Morning':1,'Afternoon':2,'Evening':3,'Night':4,'Late_Night':5}
STOPS={'zero':0,'one':1,'two_or_more':2}

class FB(BaseEstimator, TransformerMixin):
    """Leakage-safe feature builder; learns maps from the training fold only."""
    def __init__(self, use_id=False, flight_mode='split', airline_fill='prefix', dur_fill='group',
                 miss_flag=True, extra=True, extras=(), te_flight=False, te_smooth=10):
        self.use_id=use_id; self.flight_mode=flight_mode; self.airline_fill=airline_fill
        self.dur_fill=dur_fill; self.miss_flag=miss_flag; self.extra=extra; self.extras=extras; self.te_flight=te_flight; self.te_smooth=te_smooth
    def _te_fit(self, X, y):
        key=X.flight+'|'+X.flight_class; yy=pd.Series(np.asarray(y,float),index=X.index)
        cm=yy.groupby(X.flight_class).mean().to_dict(); st=yy.groupby(key).agg(['sum','count'])
        return cm,st
    def _te_apply(self, X, cm, st):
        key=X.flight+'|'+X.flight_class
        s=key.map(st['sum']).fillna(0.0); n=key.map(st['count']).fillna(0.0); prior=X.flight_class.map(cm)
        return (s+self.te_smooth*prior)/(n+self.te_smooth)
    def fit_transform(self, X, y=None, **kw):
        self.fit(X,y)
        out=self.transform(X)
        if self.te_flight:
            from sklearn.model_selection import KFold
            te=pd.Series(np.nan,index=X.index)
            for a,b in KFold(5,shuffle=True,random_state=0).split(X):
                cm,st=self._te_fit(X.iloc[a],np.asarray(y)[a]); te.iloc[b]=self._te_apply(X.iloc[b],cm,st).values
            out['te_flight']=te.values
        return out
    def fit(self, X, y=None):
        if self.te_flight: self.te_cm_,self.te_st_=self._te_fit(X,y)
        X=X.copy(); pre=X.flight.str.split().str[0]
        k=pd.DataFrame({'p':pre,'a':X.airline}).dropna()
        self.pmap_=k.groupby('p').a.agg(lambda s:s.value_counts().index[0]).to_dict()
        self.amode_=X.airline.mode()[0]
        d=X.assign(route=X.source_city+'>'+X.destination_city)
        self.dmed_=d.duration.median()
        self.g1_=d.groupby(['route','stops']).duration.median()
        self.g2_=d.groupby('stops').duration.median()
        self.fr_=d.groupby(['flight','route']).duration.median().to_dict()
        self.f_=d.groupby('flight').duration.median().to_dict()
        return self
    def transform(self, X):
        X=X.copy(); o=pd.DataFrame(index=X.index)
        pre=X.flight.str.split().str[0]; num=pd.to_numeric(X.flight.str.split().str[1],errors='coerce')
        al=X.airline.copy()
        if self.airline_fill=='prefix': al=al.fillna(pre.map(self.pmap_))
        if self.airline_fill in ('prefix','mode'): al=al.fillna(self.amode_)
        else: al=al.fillna('MISSING')
        o['airline']=al
        for c in CATS[1:]: o[c]=X[c]
        dur=X.duration.copy()
        if self.dur_fill=='group':
            route=X.source_city+'>'+X.destination_city
            g=pd.Series(list(zip(route,X.stops)),index=X.index).map(self.g1_.to_dict())
            dur=dur.fillna(g).fillna(X.stops.map(self.g2_))
        if self.dur_fill=='flight':
            route=X.source_city+'>'+X.destination_city
            a=pd.Series(list(zip(X.flight,route)),index=X.index).map(self.fr_)
            b=X.flight.map(self.f_)
            g=pd.Series(list(zip(route,X.stops)),index=X.index).map(self.g1_.to_dict())
            dur=dur.fillna(a).fillna(b).fillna(g).fillna(X.stops.map(self.g2_))
        dur=dur.fillna(self.dmed_)   # median fallback (also the 'median' strategy)
        o['duration']=dur; o['days_left']=X.days_left
        if self.miss_flag: o['dur_missing']=X.duration.isna().astype(int)
        if self.use_id: o['id']=X.id
        if self.flight_mode=='prefix': o['f_prefix']=pre
        if self.flight_mode in ('split','both'):
            o['f_prefix']=pre; o['f_num']=num.fillna(-1)
        if self.flight_mode in ('raw','both'): o['flight']=X.flight
        ex=set(self.extras)
        if 'route' in ex: o['route']=X.source_city+'>'+X.destination_city
        if 'ord' in ex: o['dep_o']=X.departure_time.map(TOD); o['arr_o']=X.arrival_time.map(TOD)
        if 'stops_n' in ex: o['stops_n']=X.stops.map(STOPS)
        if self.extra:
            o['route']=X.source_city+'>'+X.destination_city
            o['dep_o']=X.departure_time.map(TOD); o['arr_o']=X.arrival_time.map(TOD)
            o['stops_n']=X.stops.map(STOPS)
            o['class_dl']=(X.flight_class=='Business').astype(int)*100+X.days_left
        if self.te_flight: o['te_flight']=self._te_apply(X,self.te_cm_,self.te_st_).values
        return o

def make_pipe(rf_params=None, onehot=False, log=False, **fb):
    from sklearn.compose import TransformedTargetRegressor
    def selector(df):
        return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False) if onehot else \
          OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    ct=ColumnTransformer([('c',enc,selector)],remainder='passthrough')
    rf=RandomForestRegressor(**(rf_params or dict(n_estimators=200,random_state=42,n_jobs=-1)))
    p=Pipeline([('fb',FB(**fb)),('ct',ct),('rf',rf)])
    if log: return TransformedTargetRegressor(p,func=np.log1p,inverse_func=np.expm1)
    return p
