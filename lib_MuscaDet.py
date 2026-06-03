# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:59:06 2025

@author: ymsaintdrenan
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

def CSD_MuScaDet_v2(time,ETH,SZA,GHI,DHI,BNI,isValid=[],minDNI=10,Kmax=0.9,return_var=False,corrmin=0.995,c1=1,c2=4,filterMinNbCS=False):
    
    '''
    Emlb=E0*exp(-tau*mu**(-alpha))
    E0/Emlb=exp(tau*mu**(-alpha))
    log(E0/Emlb)=tau*mu**(-alpha)
    log(log(E0/Emlb))=log(tau) - alpha*log(mu)
    '''
    listNWindow=[120,60,30,15]
    
    isValid     = isValid & (GHI>0)
    isSuited0   = isValid & (GHI<ETH) & (BNI>minDNI) & (DHI<Kmax*GHI)

    GHI[(GHI>0)==False]=0
    data_csd = pd.DataFrame({'time':time,
                           'isValid':isValid,
                           'isSuited0':isSuited0,
                           'SZA':SZA,
                           'mu':np.cos(np.radians(SZA)),
                           'ETH':ETH,
                           'BNI':BNI,
                           'DHI':DHI,
                           'GHI':GHI}).set_index('time')
    dt_1min  = np.timedelta64(1,'m')
    new_time = np.arange(np.min(time[GHI>0]),np.max(time[GHI>0])+dt_1min,dt_1min)
    data_csd = data_csd.reindex(new_time,fill_value=0)
    
    isCS=data_csd['isSuited0'].values>0
    for NWindow in listNWindow:
        NWindow
        # check that more that 30% of the data are available left and right
        left_n    = ((data_csd[['isSuited0']]>0)*1).rolling(int(NWindow/2),center=False).sum()
        right_n   = (data_csd[['isSuited0']]>0)[::-1].rolling(int(NWindow/2),center=False).sum()[::-1]
        isSuited  = (data_csd['isSuited0'].values>0) & \
                  (left_n['isSuited0'].values>0.3*NWindow/2) &\
                  (right_n['isSuited0'].values>0.3*NWindow/2)
        
        # Preparation of the variables for fitting the MLB
        x1,x2,y   = np.nan*np.ones(data_csd.index.shape),np.nan*np.zeros(data_csd.index.shape),np.nan*np.zeros(data_csd.index.shape)
        x1[isSuited] = 1
        x2[isSuited] = -np.log(data_csd['mu'].values[isSuited])
        y[isSuited]  = np.log(np.log(data_csd['ETH'].values[isSuited]/data_csd['GHI'].values[isSuited]))
        
        # Fit of the MLB model
        a1,a2,cnt,isRegOK,det_XXt=RollingRegression2(x1,x2,y,NWindow=NWindow)
        idxCoefReg= \
            isSuited &\
            (np.isnan(a2)==False) & (np.isinf(a2)==False) & (np.abs(a2)<500) &\
            (np.isnan(a1)==False) & (np.isinf(a1)==False) & (np.abs(a1)<500)
        tau_,alpha_=np.nan*np.ones(data_csd.index.shape),np.nan*np.zeros(data_csd.index.shape)
        tau_[idxCoefReg]   = np.exp(a1[idxCoefReg])
        alpha_[idxCoefReg] = a2[idxCoefReg]
        data_csd['cnt_'+str(NWindow)]   = cnt
        data_csd['tau_'+str(NWindow)]   = tau_
        data_csd['alpha_'+str(NWindow)] = alpha_
        
        # Calculation of the covariance terms
        dT2=int(NWindow/2-1)
        sumX  = np.zeros(data_csd.index.shape)
        sumY  = np.zeros(data_csd.index.shape)
        sumXY = np.zeros(data_csd.index.shape)
        sumX2 = np.zeros(data_csd.index.shape)
        sumY2 = np.zeros(data_csd.index.shape)
        cntV  = np.zeros(data_csd.index.shape)
        
        tt = ' ' if NWindow < 100 else '' 
        for di in tqdm(np.arange(-dT2, dT2 + 1), desc=f"Window size={tt} {NWindow} min"):
                
            y_    = np.roll(data_csd['GHI'].values,di)
            e0_   = np.roll(data_csd['ETH'].values,di)
            mu_   = np.roll(data_csd['mu'].values,di)
            ix_   = np.roll(data_csd['isValid'].values,di)
            x_    = np.zeros(data_csd.index.shape)
            
            idxVal = idxCoefReg & ix_ #& (np.abs(alpha_)<1000)
            x_[idxVal]  = e0_[idxVal]*np.exp(-tau_[idxVal]*(mu_[idxVal]**(-alpha_[idxVal])))
            y_[idxVal==False]=0
            
            cntV+=idxVal*1
            sumX+=x_
            sumY+=y_
            sumXY+=x_*y_
            sumX2+=x_*x_
            sumY2+=y_*y_
        
        # Calculation of the correlation from the covariance terms
        
        idx_rxy = (cntV>0)
        n_xy    = cntV[idx_rxy]
        E_xy    = sumXY[idx_rxy]/n_xy
        E_x     = sumX[idx_rxy]/n_xy
        E_x2    = sumX2[idx_rxy]/n_xy
        E_y     = sumY[idx_rxy]/n_xy
        E_y2    = sumY2[idx_rxy]/n_xy
        cov_xy  = E_xy-E_x*E_y
        V_x     = E_x2-E_x**2
        V_y     = E_y2-E_y**2
        
        cov_xy=np.abs(cov_xy)
        
        sig_X   = np.maximum(1E-10,V_x)**0.5
        sig_Y   = np.maximum(1E-10,V_y)**0.5
        corr_xy = cov_xy/(sig_X*sig_Y)
        
        data_csd['cov_xy_'+str(NWindow)]=np.ones(data_csd.index.shape)*np.nan
        data_csd.loc[idx_rxy,'cov_xy_'+str(NWindow)]=cov_xy
        data_csd['sig_X_'+str(NWindow)]=np.ones(data_csd.index.shape)*np.nan
        data_csd.loc[idx_rxy,'sig_X_'+str(NWindow)]=sig_X
        data_csd['sig_Y_'+str(NWindow)]=np.ones(data_csd.index.shape)*np.nan
        data_csd.loc[idx_rxy,'sig_Y_'+str(NWindow)]=sig_Y
        data_csd['corr_xy_'+str(NWindow)]=np.ones(data_csd.index.shape)*np.nan
        data_csd.loc[idx_rxy,'corr_xy_'+str(NWindow)]=corr_xy
        
        covxy  = data_csd['cov_xy_'+str(NWindow)]
        vx     = data_csd['sig_X_'+str(NWindow)]**2
        vy     = data_csd['sig_Y_'+str(NWindow)]**2
        tol    = c1+c2*data_csd['mu'].values
        limcov = corrmin*((vx)**0.5)*((np.maximum(0,vy-tol**2))**0.5)
        isCS   = isCS & (covxy>limcov)
    
    # Verification that at least 30% of the values within the longest time window corresponds to clearsky
    data_csd['isCS'] = isCS
    count_nCS        = ((data_csd[['isCS']]>0)*1).rolling(int(NWindow),center=True).sum()
    if filterMinNbCS:
        isCS             = isCS & (count_nCS['isCS'].values>0.3*NWindow)
    data_csd['isCS'] = isCS
    
    sync_isCS=np.zeros(time.shape)
    ta=((time-np.datetime64('1970-01-01 00:00'))/np.timedelta64(1,'m')).astype("int64")
    tb=((data_csd.index-np.datetime64('1970-01-01 00:00'))/np.timedelta64(1,'m')).astype("int64")
    [c,ia,ib]=np.intersect1d(ta,tb,return_indices=True)
    sync_isCS[ia]=isCS.values[ib]
    
    if return_var:
        return sync_isCS,data_csd
    else:
        return sync_isCS


    
    
def RollingRegression2(x1,x2,y,NWindow=180,minNbVal=2,min_det_XXt=0):
    
    idxOK=(np.isnan(y)==False) & (np.isnan(x1)==False) & (np.isnan(x1)==False)
    
    y[idxOK==False]=0
    x1[idxOK==False]=0
    x2[idxOK==False]=0
    
    df=pd.DataFrame({'x1':x1,'x2':x2,'y':y})
    df['count']=1*(idxOK)
    # XY
    df['x1y']=df['x1']*df['y']
    df['x2y']=df['x2']*df['y']

    # Cx
    df['x1x2']=df['x1']*df['x2']
    
    # X^2
    df['x1_2']=df['x1']*df['x1']
    df['x2_2']=df['x2']*df['x2']

    df2=df.rolling(NWindow,center=True).sum()
    
    nT=len(df2.index)
    XXt=np.ones((nT,2,2))
    
    XXt[:,0,0]=df2['x1_2']
    XXt[:,0,1]=df2['x1x2']
    
    XXt[:,1,0]=df2['x1x2']
    XXt[:,1,1]=df2['x2_2']
    
    
    idxCount=(df2['count'].values>minNbVal)
    det_XXt=np.zeros(df2.index.shape)
    det_XXt[idxCount]=np.linalg.det(XXt[idxCount,:,:])
    df2['det_XXt']=det_XXt
    
    idx_det=(det_XXt>min_det_XXt)
    inv_XXt=np.linalg.inv(XXt[idx_det,:,:])
    
    a1,a2,isOK=np.nan*np.ones(nT),np.nan*np.ones(nT),np.zeros(nT)
    df3=df2[idx_det]
    a1[idx_det]=inv_XXt[:,0,0]*df3['x1y']+inv_XXt[:,1,0]*df3['x2y']
    a2[idx_det]=inv_XXt[:,0,1]*df3['x1y']+inv_XXt[:,1,1]*df3['x2y']
    cnt=df2['count'].values
    isOK[idx_det]=1
    
    return a1,a2,cnt,isOK,det_XXt