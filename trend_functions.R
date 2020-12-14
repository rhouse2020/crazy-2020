library(tidyverse)
library(rstan)

# Function to read in the virginia data, and extract a single county
fairfax_data <- function(va_file) {
    # read in the virginia data
    va_data <- read_csv(va_file,
                        col_types = cols(`Report Date` = col_date(format = "%m/%d/%Y"),
                                         'Total Cases' = col_integer(),
                                         'Hospitalizations' = col_integer(),
                                         'Deaths' = col_integer()))
    # Using Locality to group by county/city
    va_locality <- va_data %>% 
        select(date=`Report Date`, Locality, tot_cases=`Total Cases`, hosp=Hospitalizations, deaths=Deaths) %>% 
        arrange(date)
    
    va_locality <- va_locality %>% group_by(Locality) %>% 
        mutate(new_cases=tot_cases -lag(tot_cases), # get the count of daily new cases
               new_hosp = hosp - lag(hosp,1),       # get the count of daily new hospitalizations
               new_dead = deaths - lag(deaths,1)) %>% 
        filter(!(is.na(new_cases))) # removing that first date where we don't know the "new" number
    
    # Extracting just fairfax county
    fairfax <- va_locality %>% filter(Locality == "Fairfax")
    # These next 3 lines are irritating, but necessary, b/c some dates have negative counts!
    fairfax$new_cases[fairfax$new_cases < 0] <- 0 
    fairfax$new_hosp[fairfax$new_hosp < 0] <- 0 
    fairfax$new_dead[fairfax$new_dead < 0] <- 0 
    return(fairfax)
}

# Functions to get data formatted for stan
stan_data_cases <- function(fairfax) {
    data_cases <- list(
        'T'=length(fairfax$new_cases), # number of cases
        'S'=7, # days in a week
        'K'=2,
        'INTR'=c(51,68),
        'Y'=log(1 + fairfax$new_cases)) # log(# cases + 1)
    return(data_cases)
}

stan_data_hosp <- function(fairfax){
    data_hosp <-list(
        'T'=length(fairfax$new_hosp), # number of hospitalizations
        'S'=7, # days in a week
        'Y'=log(1 + fairfax$new_hosp)) # b/c sometimes it is negative
}


stan_data_dead <- function(fairfax) {
    data_dead <-list(
        'T'=length(fairfax$new_dead), # number of deaths
        'S'=7, # days in a week
        'Y'=log(1 + fairfax$new_dead)) # b/c sometimes it is negative
}

# Function to run the model
run_the_model <- function(model_name, stan_data) {
    # Estimate the model
    stan_fit <- sampling(model_name, stan_data, iter = 800, chains = 2,
                         control=list(max_treedepth=13, adapt_delta=0.98))
    # Extract the results
    model_results <- extract(stan_fit)
    # since we did log(new_cases + 1), subtracting the 1 here
    trend <- apply(model_results$mu_exp, 2, mean) - 1
    trend <- data.frame('date'=fairfax$date,'trend'=trend)
    return(trend)
}

# Function that runs all the functions
get_trends <- function(va_file) {
    # Data setup
    fairfax <<- fairfax_data(va_file) # saving this to global for plotting
    stan_cases <- stan_data_cases(fairfax)
    stan_hosp <- stan_data_hosp(fairfax)
    stan_dead <- stan_data_dead(fairfax)
    
    # Model compilation
    model_file <- stan_model('lcl_lvl_irw_dow.stan') # this is used for new cases
    model_nointr_file <- stan_model('lcl_lvl_irw_dow_no_intr.stan') # this is used for hosp and dead
    
    # Running the model to get trends
    trend_cases <- run_the_model(model_file, stan_cases)
    trend_hosp <- run_the_model(model_nointr_file, stan_hosp)
    trend_dead <- run_the_model(model_nointr_file, stan_dead)
    
    all_trends <- left_join(trend_cases, trend_hosp, by='date')
    all_trends <- left_join(all_trends, trend_dead, by='date')
    colnames(all_trends) <- c('date','trend','trend_hosp','trend_dead')
    return(all_trends)
}