# Run these 3 lines to get trend data

source('trend_functions.R') # change the path as needed

trends <- get_trends(filename) # filename is the daily virginia data file
write.csv(trends, 'trends/trend_today_dow.csv', row.names = FALSE) # save out the trend data

######################################

# If you want to look at the trend lines vs the data:

ggplot(fairfax, aes(x=date, y=new_cases)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 cases for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trends$trend), color='blue')

ggplot(fairfax, aes(x=date, y=new_hosp)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 hospitalizations for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trends$trend_hosp), color='blue')

ggplot(fairfax, aes(x=date, y=new_dead)) + geom_point() + 
    labs(title='Daily change in # of Covid-19 deaths for Fairfax') + xlab('Date') + ylab('Daily Delta') + 
    geom_line(aes(fairfax$date, y=trends$trend_dead), color='blue')
