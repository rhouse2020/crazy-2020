#
# extract trends in new COVID cases in each US state
# from NYTimes data posted on their github page
#

# -------------------------------------------------------
# load libraries

# numerical libraries
import numpy as np
import scipy as sp
import pystan

# pandas!
import pandas as pd

# for caching stan files
import pickle
from hashlib import md5

# dates
import datetime

# -------------------------------------------------------
# constants

nyt_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
stan_filename = 'kalman_smoother_local_level_matrix.stan'
stan_modelname = 'ks_lcl_lvl'


# -------------------------------------------------------
# functions

def compile_stan_file(file_name,model_name):
    """
    compiles all stan file,
    before compiling, looks to see if pkl exists
    """
    
    print('Compiling Stan file: '+file_name)
        
    # make a code hasl
    code_hash = md5(file_name.encode('ascii')).hexdigest()
    cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
                
    # can we load a cached copy?
    try:
        stan_model = pickle.load(open(cache_fn, 'rb'))
    except:
        print('\t compiling model...')
        stan_model = pystan.StanModel(file=file_name,model_name=model_name)
        with open(cache_fn, 'wb') as f:
            pickle.dump(stan_model, f)
    else:
        print('\t Using cached StanModel for '+file_name)

    return stan_model


def get_clean_nyt_data(nyt_url):
    """
    given URL for NYTimes COVID data, download,
    clean, and return nice list of states
    and dataframe with daily data
    """

    # download NYT data
    df = pd.read_csv(nyt_url)
    print('Data downloaded from NYT gihub.')

    # aggregate to state-day
    stt_day_df = df.groupby(['date','state']).sum()['cases'].reset_index()
    stt_day_df['date'] = pd.to_datetime(stt_day_df['date'])
    max_day = max(stt_day_df['date'])
    print('Latest observations from: ')
    print(max_day)

    # for computing daily changes
    stt_day_yesterday_df = stt_day_df.copy()
    stt_day_yesterday_df.rename(columns={'date':'yesterday','cases':'cases_yesterday'},inplace=True)

    # ensure all state-date combinations are in dataset
    stt_lst = np.sort(list(set(stt_day_df['state'])))
    date_idx = pd.date_range(start=datetime.date(2020,3,1),end=max_day,freq='D')
    stt_date_idx = pd.MultiIndex.from_product([stt_lst,date_idx],names=['state','date'])

    stt_date_combo_df = pd.DataFrame(np.zeros(len(stt_date_idx)),
                                      index=stt_date_idx,
                                      columns=['z']
                                     )

    stt_date_combo_df.reset_index(inplace=True)
    stt_date_combo_df = stt_date_combo_df[['state','date']].copy()
    stt_date_combo_df['yesterday'] = [stt_date_combo_df.loc[i,'date'] - datetime.timedelta(1) for i in stt_date_combo_df.index]

    # merge today
    stt_day_df = stt_date_combo_df.merge(stt_day_df,left_on=['state','date'],right_on=['state','date'],how='left').copy()
    stt_day_df.fillna(0,inplace=True)

    # merge yesterday
    stt_day_df = stt_day_df.merge(stt_day_yesterday_df,left_on=['state','yesterday'],right_on=['state','yesterday'],how='left').copy()
    stt_day_df.fillna(0,inplace=True)

    # compute difference
    stt_day_df['new_cases'] = stt_day_df['cases'] - stt_day_df['cases_yesterday']

    # some days have wacky data and we get negative values
    stt_day_df[stt_day_df['new_cases'] < 0] = np.nan

    # make a linear interpolation
    stt_day_df['new_cases'] = stt_day_df['new_cases'].interpolate()

    return stt_lst, stt_day_df


def make_stan_data(df,stt_name):
    """return dictionary for stan model with day of week effects
    """
    select_df = df[df['state'] == stt_name].copy()
    len_y = len(select_df)
    y_log = np.log(1 + np.array(select_df['new_cases']).reshape([len_y,1]))
    
    # define matrices for local level model with trend

    p = 1 # number of series
    s = 7 # number of seasons
    m = (s - 1) + 2 #+ 1 # number of state variables: trend (1), slope (1), seasons(6), intervention(1)
    r = 2 # number of variances: slope(1), season (1)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # T matrix
    # this defines integrated random walk trend
    # with slope
    # and day of week seasonality
    T = np.zeros((m,m))

    # transitions for slope and trend
    T[:2,:2] = np.eye(2)
    T[0,1] = 1

    # transitions for seasons
    T[2,2:m] = -1
    T[3:m,2:(m-1)] = np.eye(s-2)


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Z matrix
    Z = np.zeros((p,m))

    # trend 
    Z[0,0] = 1

    # season
    Z[0,2] = 1

    # Z_t matrix
    Z_repeat = [Z for _ in range(len_y)]
    Z_t = np.stack(Z_repeat,axis=0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # R matrix
    # this makes an integrated random walk
    R = np.zeros((m,r))
    R[1,0] = 1
    R[2,1] = 1

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # setup dictionary for Stan model
    stan_data = {
        'n':len_y,
        'p':p,
        'm':m,
        'r':r,
        'Z':Z_t,
        'T':T,            
        'R':R,
        'Y':y_log,
        'prior_only':0
    }
    
    return stan_data, list(select_df['date'])


def estimate_model(df,stt_lst,stan_model):
    """iterate through list of states, estimate model, and store results
    """
    
    results = {}
    
    print('Estimating trends for: ')
    for s in stt_lst:
        print('\t'+s)
        s_dict, dates = make_stan_data(df,s)
        
        # conduct MCMC using Stan
        fit = stan_model.sampling(data=s_dict,
                                  iter=300,
                                  control={'max_treedepth':12,'adapt_delta':0.90},
                                  chains=2
                                 )
        
        # extract samples
        model_results = fit.extract(permuted=True)
        
        # assemble dataframe of results
        trend_df = pd.DataFrame(np.mean(np.exp(model_results['a_hat'][:,:,0]),axis=0) - 1,columns=['trend'])
        trend_df['state'] = s
        trend_df['date'] = dates
        
        # store in dictionary
        results[s] = trend_df
    
    # stack results
    all_trends = pd.concat([results[s] for s in stt_lst])[['state','date','trend']]
    
    return all_trends

# -------------------------------------------------------
# run code

# compile stan file
bsts = compile_stan_file(stan_filename,stan_modelname)

# get NYTimes data
stt_lst, stt_day_df  = get_clean_nyt_data(nyt_url)

# compute trends for all 50 states
trends = estimate_model(stt_day_df,stt_lst,bsts)

# export data
trends.to_csv('covid_state_localleveltrend_today_test.csv')