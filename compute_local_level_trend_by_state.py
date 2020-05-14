# -------------------------------------------- #
# -------------------------------------------- #
#              IMPORT PACKAGES                 #
# -------------------------------------------- #
# -------------------------------------------- #

# numerical libraries
import numpy as np
import scipy as sp
import pystan

# pandas!
import pandas as pd


# -------------------------------------------- #
# -------------------------------------------- #
#                 CLEAN DATA                   #
# -------------------------------------------- #
# -------------------------------------------- #

# download NYT data
nyt_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
df = pd.read_csv(nyt_url)

# aggregate to state-day
stt_day_df = df.groupby(['date','state']).sum()['cases'].reset_index()
stt_day_df['date'] = pd.to_datetime(stt_day_df['date'])

# for computing daily changes
stt_day_yesterday_df = stt_day_df.copy()
stt_day_yesterday_df.rename(columns={'date':'yesterday','cases':'cases_yesterday'},inplace=True)

# all state-date combinations are in dataset
stt_lst = sort(list(set(stt_day_df['state'])))
date_idx = pd.date_range(start=datetime.date(2020,3,1),end=datetime.date(2020,5,13),freq='D')
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


# -------------------------------------------- #
# -------------------------------------------- #
#              HELPER FUNCTIONS                #
# -------------------------------------------- #
# -------------------------------------------- #

def make_stan_data(df,stt_name,s=7):
    """return dictionary for stan model
    """
    select_df = df[df['state'] == stt_name].copy()
    len_y = len(select_df)
    y = np.log(1 + np.array(select_df['new_cases']))
    
    return {'T':len_y,'S':s,'Y':y}, list(select_df['date'])

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
                                  iter=800,
                                  control={'max_treedepth':13,'adapt_delta':0.98},
                                  chains=2
                                 )
        
        # extract samples
        model_results = fit.extract(permuted=True)
        
        # assemble dataframe of results
        trend_df = pd.DataFrame(np.mean(model_results['mu_exp'],axis=0) - 1,columns=['trend'])
        trend_df['state'] = s
        trend_df['date'] = dates
        
        # store in dictionary
        results[s] = trend_df
    
    # stack results
    all_trends = pd.concat([results[s] for s in stt_lst])[['state','date','trend']]
    
    return all_trends


# -------------------------------------------- #
# -------------------------------------------- #
#               ESTIMATE MODEL                 #
# -------------------------------------------- #
# -------------------------------------------- #

# compile Stan model
bsts = pystan.StanModel(file='lcl_lvl_irw_state.stan')

# compute trends for all 50 states
trends = estimate_model(stt_day_df,stt_lst,bsts)

# export data
trends.to_csv('covid_state_localleveltrend_today.csv')