library(tidyverse)
library(rstan)

va_data <- read_csv(filename ,
                    col_types = cols(`Report Date` = col_date(format = "%m/%d/%Y"),
                                     'Total Cases' = col_integer(),
                                     'Hospitalizations' = col_integer(),
                                     'Deaths' = col_integer()))
# renaming columns
va_data <- va_data %>% 
    select(date=`Report Date`, Locality, district=`VDH Health District`, cases=`Total Cases`, Hospitalizations, Deaths) %>% 
    arrange(date)

# Using Locality instead of health districts to pull out just a single county
va_locality <- va_data %>% 
    select(date, Locality, tot_cases=cases, hosp=Hospitalizations, deaths=Deaths)

va_locality <- va_locality %>% group_by(Locality) %>% 
    mutate(new_cases=tot_cases -lag(tot_cases)) %>% # get the count of daily new cases
    filter(!(is.na(new_cases))) # removing that first date where we don't know the "new" number


fairfax <- va_locality %>% filter(Locality == "Fairfax")

# Data for the model
data <- list(
    'T'=length(fairfax$new_cases), # number of cases
    'S'=7, # days in a week
    'INTR'=51,
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

write.csv(trend, 'trends/trend_today_dow.csv', row.names = FALSE)

ggplot(fairfax, aes(x=date, y=new_cases)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 cases for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trend$trend), color='blue')
