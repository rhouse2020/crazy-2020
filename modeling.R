library(tidyverse)
library(rstan)

va_data <- read.csv('va_daily.csv')
va_data$Report.Date <- as.Date(as.character(va_data$Report.Date), '%m/%d/%Y')
va_data <- va_data %>% arrange(Report.Date)

va_districts <- va_data %>% group_by(Report.Date, VDH.Health.District) %>% 
    summarise(Total.Cases=sum(Total.Cases), 
              Hospitalizations=sum(Hospitalizations), 
              Deaths=sum(Deaths))
va_districts <- va_districts %>% group_by(VDH.Health.District) %>%
    mutate(day_change=Total.Cases -lag(Total.Cases), day_inc=day_change>lag(day_change))

fairfax <- va_districts %>% filter(VDH.Health.District == "Fairfax")

data <- list('T'=length(fairfax$Total.Cases), 'Y'=log(1 + fairfax$Total.Cases))

model_file <- stan_model('local_lvl_irw.stan')

stan_fit <- sampling(model_file, data, iter = 2000, chains = 4)
