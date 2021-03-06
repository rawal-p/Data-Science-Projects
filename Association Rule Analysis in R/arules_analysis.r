rm(list=ls())
setwd("C:\\Users\\rawal\\Desktop\\Masters\\Fall 2020\\STATS 780\\R Assignments\\week1\\Association Rule Analysis in R")
library(arules) 
source("std_lift.R")
x<-read.transactions("heart_fail_data.txt",format="single",cols=c(1,2))
x

# Interested in rules with consequent deaths (y5) 
app<-list(rhs=c("y5"),default = "lhs")
params<-list(support=0.008,confidence=0.45,minlen=2,maxlen=5)
fit<-apriori(x,parameter=params,appearance=app)

quality(fit)<-std_lift(fit,x)
inspect(sort(fit, by = "slift"))
