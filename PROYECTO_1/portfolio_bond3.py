#SELECCION DE ACTIVOS
def get_historical_closes(ticker, start_date, end_date):
    import pandas_datareader.data as web
    p = web.DataReader(ticker, "yahoo", start_date, end_date).sort_index('major_axis')
    d = p.to_frame()['Adj Close'].reset_index()
    d.rename(columns={'minor': 'Ticker', 'Adj Close': 'Close'}, inplace=True)
    pivoted = d.pivot(index='Date', columns='Ticker')
    pivoted.columns = pivoted.columns.droplevel(0)
    return pivoted

def get_volume(ticker, start_date, end_date):
    import pandas_datareader.data as web
    p = web.DataReader(ticker, "yahoo", start_date, end_date).sort_index('major_axis')
    d = p.to_frame()['Volume'].reset_index()
    d.rename(columns={'minor': 'Ticker', 'Adj Close': 'Close'}, inplace=True)
    pivoted = d.pivot(index='Date', columns='Ticker')
    pivoted.columns = pivoted.columns.droplevel(0)
    return pivoted

def calc_daily_returns(closes):
    import numpy as np
    return np.log(closes/closes.shift(1))[1:]

def calc_annual_returns(daily_returns):
    import numpy as np
    grouped = np.exp(daily_returns.groupby(lambda date: date.year).sum())-1
    return grouped

def calc_volum_an(dataV):
    import numpy as np
    grouped = np.mean(dataV.groupby(lambda date: date.year).sum())
    return grouped


#OPTIMIZACIÓN
def sim_mont_portfolio(daily_returns,num_portfolios,risk_free):
    num_assets=len(daily_returns.T)
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    import statsmodels.api as sm
    huber = sm.robust.scale.Huber()
    #Mean and standar deviation returns
    returns_av, scale = huber(daily_returns)
    #returns_av = daily_returns.mean()
    covariance= skcov.ShrunkCovariance().fit(daily_returns).covariance_
    #Simulated weights
    weights = np.array(np.random.random(num_assets*num_portfolios)).reshape(num_portfolios,num_assets)
    weights = weights*np.matlib.repmat(1/weights.sum(axis=1),num_assets,1).T
    ret=252*weights.dot(returns_av).T
    sd = np.zeros(num_portfolios)
    for i in range(num_portfolios):
        sd[i]=np.sqrt(252*(((weights[i,:]).dot(covariance)).dot(weights[i,:].T))) 
    sharpe=np.divide((ret-risk_free),sd)    
    return pd.DataFrame(data=np.column_stack((ret,sd,sharpe,weights)),columns=(['Returns','SD','Sharpe']+list(daily_returns.columns)))

def optimal_portfolio(daily_returns,N,r):
    # Frontier points
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    import cvxopt as opt
    from cvxopt import blas, solvers
    import statsmodels.api as sm
    huber = sm.robust.scale.Huber()
    n = len(daily_returns.T)
    returns = np.asmatrix(daily_returns)
    mus = [(10**(5.0 * t/N- 1.0)-10**(-1)) for t in range(N)]   
    #cvxopt matrices
    S = opt.matrix(skcov.ShrunkCovariance().fit(returns).covariance_)
    returns_av, scale = huber(returns)
    pbar = opt.matrix(returns_av)    
    # Constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    # Risk and returns
    returns = [252*blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(252*blas.dot(x, S*x)) for x in portfolios]
    portfolios=[np.eye(n).dot(portfolios[i])[:,0] for i in range(N)]
    returns = np.asarray(returns)
    risks = np.asarray(risks)
    sharpe=np.divide((returns-r),risks) 
    portfolios = np.asarray(portfolios)
    return  pd.DataFrame(data=np.column_stack((returns,risks,sharpe,portfolios)),columns=(['Returns','SD','Sharpe']+list(daily_returns.columns)))

def optimal_portfolio_b(daily_returns,N,r,c0):
    # Frontier points
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    import cvxopt as opt
    from cvxopt import blas, solvers
    import statsmodels.api as sm
    cm = np.insert((np.insert(skcov.ShrunkCovariance().fit(daily_returns).covariance_,len(daily_returns.T),0,axis=0)),len(daily_returns.T),0,axis=1)
    huber = sm.robust.scale.Huber()
    mus = [(10**(5.0 * t/N- 1.0)-10**(-1)) for t in range(N)]
    n = len(daily_returns.T)+1
    #cvxopt matrices
    S = opt.matrix(cm)
    returns_av, scale = huber(daily_returns)
    pbar = opt.matrix(np.r_[returns_av,c0]) 
    daily_returns['BOND']=c0*np.ones(daily_returns.index.size)
    returns = np.asmatrix(daily_returns)
    # Constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    # Risk and returns
    returns = [252*blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(252*blas.dot(x, S*x)) for x in portfolios]
    portfolios=[np.eye(n).dot(portfolios[i])[:,0] for i in range(N)]
    returns = np.asarray(returns)
    risks = np.asarray(risks)
    sharpe=np.divide((returns-r),risks) 
    portfolios = np.asarray(portfolios)
    return  pd.DataFrame(data=np.column_stack((returns,risks,sharpe,portfolios)),columns=(['Returns','SD','Sharpe']+list(daily_returns.columns)))



#COBERTURAS
def call_payoff(ST, K):
    return max(0, ST-K)

def call_payoffs(STmin, STmax, K, step=1):
    import pandas as pd
    import numpy as np
    maturities = np.arange(STmin, STmax+step, step)
    payoffs = np.vectorize(call_payoff)(maturities, K)
    df = pd.DataFrame({'Strike': K, 'Payoff': payoffs}, index=maturities)
    df.index.name = 'Precio de maduración'
    return df

def put_payoff(ST, K):
    return max(0, K-ST)

def plot_pnl(pnl_df, okind, who):
    import matplotlib.pyplot as plt
    plt.ylim(pnl_df.Payoff.min() - 10, pnl_df.Payoff.max() + 10)
    plt.ylabel("Ganancia/pérdida")
    plt.xlabel("Precio de maduración")
    plt.title('Ganancia y pérdida de una opción {0} para el {1}, Prima={2}, Strike={3}'.format(okind, who, pnl_df.Prima.iloc[0],
    pnl_df.Strike.iloc[0]))
    plt.ylim(pnl_df.PnL.min()-3, pnl_df.PnL.max() + 3)
    plt.xlim(pnl_df.index[0], pnl_df.index[len(pnl_df.index)-1])
    plt.plot(pnl_df.index, pnl_df.PnL)
    plt.axhline(0, color='g');

#long=compra
#short=venta
#ventas,compras
def bear_call(ct_short, K_short, ct_long, K_long, STmin, STmax, step = 1):
    import pandas as pd
    import numpy as np
    maturities = np.arange(STmin, STmax+step, step)
    payoffs_cl = np.vectorize(call_payoff)(maturities, K_long)
    payoffs_cs = np.vectorize(call_payoff)(maturities, K_short)
    df = pd.DataFrame({'Strike': K_long, 'Payoff': payoffs_cs, 'Prima': ct_long-ct_short, 'PnL': payoffs_cl-ct_long+(ct_short-payoffs_cs)}, index=maturities)
    df.index.name = 'Precio de maduración'
    return df

def straddle(ct_long, K_long, pt_long, STmin, STmax, step = 1):
    import pandas as pd
    import numpy as np
    maturities = np.arange(STmin, STmax+step, step)
    payoffs_cl = np.vectorize(call_payoff)(maturities, K_long)
    payoffs_pl = np.vectorize(put_payoff)(maturities, K_long)
    df = pd.DataFrame({'Strike': K_long, 'Payoff': payoffs_cl+payoffs_pl, 'Prima': pt_long+ct_long, 'PnL': payoffs_cl-ct_long+(payoffs_pl-pt_long)}, index=maturities)
    df.index.name = 'Precio de maduración'
    return df

