library(tidyverse)
library(rstan)

va_data <- read_csv("va_daily.csv" ,
                    col_types = cols(`Report Date` = col_date(format = "%m/%d/%Y"),
                                     'Total Cases' = col_integer(),
                                     'Hospitalizations' = col_integer(),
                                     'Deaths' = col_integer()))
# renaming columns
va_data <- va_data %>% 
    select(date=`Report Date`, Locality, district=`VDH Health District`, cases=`Total Cases`, Hospitalizations, Deaths) %>% 
    arrange(date)

# Summing up the health districts
va_districts <- va_data %>% group_by(date, district) %>% 
    summarise(tot_cases=sum(cases), 
              hosp=sum(Hospitalizations), 
              deaths=sum(Deaths))

va_districts <- va_districts %>% group_by(district) %>%
    mutate(new_cases=tot_cases -lag(tot_cases)) %>% # get the count of daily new cases
    filter(!(is.na(new_cases))) # removing that first date where we don't know the "new" number

fairfax <- va_districts %>% filter(district == "Fairfax")

# Data for the model
data <- list(
    'T'=length(fairfax$new_cases), # number of cases
    'Y'=log(1 + fairfax$new_cases)) # log(# cases + 1)

# Compiling the model
model_file <- stan_model('local_lvl_irw.stan')

# Estimating the model
stan_fit <- sampling(model_file, data, iter = 500, chains = 2,
                     control=list(max_treedepth=13, adapt_delta=0.98))
# Extracting the results
model_results <- extract(stan_fit)

# since we did log(new_cases + 1), subtracting the 1 here
trend <- apply(model_results$Y_sim_exp, 2, mean) - 1

ggplot(fairfax, aes(x=date, y=new_cases)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 cases for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trend), color='blue')
