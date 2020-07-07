library(tidyverse)
library(rstan)

va_data <- read_csv(filename ,
                    col_types = cols(`Report Date` = col_date(format = "%m/%d/%Y"),
                                     'Total Cases' = col_integer(),
                                     'Hospitalizations' = col_integer(),
                                     'Deaths' = col_integer()))

# Using Locality instead of health districts to pull out just a single county
va_locality <- va_data %>% 
    select(date=`Report Date`, Locality, tot_cases=`Total Cases`, hosp=Hospitalizations, deaths=Deaths) %>% 
    arrange(date)

va_locality <- va_locality %>% group_by(Locality) %>% 
    mutate(new_cases=tot_cases -lag(tot_cases), # get the count of daily new cases
           new_hosp = hosp - lag(hosp,1),       # get the count of daily new hospitalizations
           new_dead = deaths - lag(deaths,1)) %>% 
    filter(!(is.na(new_cases))) # removing that first date where we don't know the "new" number

fairfax <- va_locality %>% filter(Locality == "Fairfax")
# These next 3 lines are irritating that they are necessary
fairfax$new_cases[fairfax$new_cases < 0] <- 0 # b/c a couple of them are negative
fairfax$new_hosp[fairfax$new_hosp < 0] <- 0 # b/c a couple of them are negative
fairfax$new_dead[fairfax$new_dead < 0] <- 0 # b/c a couple of them are negative

# Data for the model
data <- list(
    'T'=length(fairfax$new_cases), # number of cases
    'S'=7, # days in a week
    'K'=2,
    'INTR'=c(51,68),
    'Y'=log(1 + fairfax$new_cases)) # log(# cases + 1)

# Compiling the model
model_file <- stan_model('lcl_lvl_irw_dow.stan')

# Estimating the model
stan_fit <- sampling(model_file, data, iter = 800, chains = 2,
                     control=list(max_treedepth=13, adapt_delta=0.98))
# Extracting the results
model_results <- extract(stan_fit)

# since we did log(new_cases + 1), subtracting the 1 here
trend <- apply(model_results$mu_exp, 2, mean) - 1
trend <- data.frame('date'=fairfax$date,'trend'=trend)

ggplot(fairfax, aes(x=date, y=new_cases)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 cases for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trend$trend), color='blue')

################################
# Model without intervention dates for Hospitalizations and Deaths

model_nointr_file <- stan_model('lcl_lvl_irw_dow_no_intr.stan')

# Hospitalizations
data_hosp <-list(
    'T'=length(fairfax$new_hosp), # number of cases
    'S'=7, # days in a week
    'Y'=log(1 + fairfax$new_hosp)) # b/c sometimes it is negative

stan_fit_hosp <- sampling(model_nointr_file, data_hosp, iter = 800, chains = 2,
                          control=list(max_treedepth=13, adapt_delta=0.98))

model_results_hosp <- extract(stan_fit_hosp)

# since we did log(new_cases + 1), subtracting the 1 here
trend_hosp <- apply(model_results_hosp$mu_exp, 2, mean) - 1
trend_hosp <- data.frame('date'=fairfax$date,'trend_hosp'=trend_hosp)

ggplot(fairfax, aes(x=date, y=new_hosp)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 hospitalizations for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trend_hosp$trend_hosp), color='blue')

# Deaths
data_dead <-list(
    'T'=length(fairfax$new_dead), # number of cases
    'S'=7, # days in a week
    'Y'=log(1 + fairfax$new_dead)) # b/c sometimes it is negative

stan_fit_dead <- sampling(model_nointr_file, data_dead, iter = 800, chains = 2,
                          control=list(max_treedepth=13, adapt_delta=0.98))

model_results_dead <- extract(stan_fit_dead)

# since we did log(new_cases + 1), subtracting the 1 here
trend_dead <- apply(model_results_dead$mu_exp, 2, mean) - 1
trend_dead <- data.frame('date'=fairfax$date,'trend_dead'=trend_dead)

ggplot(fairfax, aes(x=date, y=new_dead)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 deaths for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trend_dead$trend_dead), color='blue')

all_trends <- left_join(trend, trend_hosp, by='date')
all_trends <- left_join(all_trends, trend_dead, by='date')

write.csv(all_trends, 'trends/trend_today_dow.csv', row.names = FALSE)
