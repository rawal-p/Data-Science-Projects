#Pranav Rawal
#McMaster University - Electrical and Computer Engineering Department


library(ggplot2)
library(e1071)
library(dplyr)

set.seed(1)

#stars <- read.csv("hygdata_v3.csv", header=TRUE)
stars <- read.csv("outdata.csv", header=TRUE)

stars$dist[stars$dist == 1e05] <- NA
clean_stars <- stars[complete.cases(stars[, c("ci", "dist", "x", "y", "z", "mag")]), ]

#summary(clean_stars)
#summary(clean_stars$cat)
#summary(clean_stars$cat) / nrow(clean_stars) * 100
#summary(clean_stars$newcat)
#summary(clean_stars$newcat) / nrow(clean_stars) * 100

#ggplot(s, aes(x=ci, y=absmag, color=newcat)) + geom_point() + xlim(-0.5, 2.5) + ylim(18, -15) + labs(x="Color Index", y="Absolute Magnitude", title="H-R Diagram (Actual Classes)", color="Stellar Category")

stars_scaled <- clean_stars %>% dplyr::select(mag, absmag, dist, ci, x, y, z, lum, cat, newcat)
stars_scaled[, -c(9,10)] <- scale(stars_scaled[, -c(9, 10)])

train <- sample(nrow(clean_stars), 0.75 * nrow(clean_stars))

samp <- stars_scaled[train, -9]
st <- stars_scaled[train, -c(9, 10)]
labels <- factor(stars_scaled[train, 9])

nsamp <- stars_scaled[-train, -9]
nst <- stars_scaled[-train, -c(9, 10)]
nlabels <- factor(stars_scaled[-train, 9])

# random forests
library(randomForest)

# this took ran for two hours without completing, so I manually determined the values (default worked best)
#stars_rf <- tune.randomForest(newcat~., data = st, mtry=1:3, ntree=100*1:5, tunecontrol = tune.control(sampling = "cross", cross = 5))

st_rf <- randomForest(newcat~., data=samp, mtry=1, importance=TRUE, type="class")
st_rf_pred <- stats::predict(st_rf, nst, type="class")
table(nlabels, st_rf_pred)
adjustedRandIndex(nlabels, st_rf_pred)
#[1] 0.6279808

# k-means
library(cluster)

# from lecture notes, McNicholas (2019)
K <- 10
wss <- rep(0,K)
for (k in 1:K) {
    wss[k] <- sum(kmeans(st[, c(2:4)], k)$withinss)
}
plot(1:K, wss, typ="b", ylab="Total within cluster sum of squares", xlab="Number of clusters (k)")
dev.print(device=pdf, "elbow.pdf")

stars_k <- kmeans(st[, c(2,4)], 5)
ggplot(s, aes(x=ci, y=absmag, color=factor(stars_k$cluster))) + geom_point() + xlim(-0.5, 2.5) + ylim(18, -15) + labs(x="Color Index", y="Absolute Magnitude", title="H-R Diagram (K-means, k = 5)", color="Stellar Category")
#print(table(stars_k$cluster, s$newcat))
#dev.print(device=pdf, "k5.pdf")

stars_k <- kmeans(st[, c(2,4)], 9)
ggplot(st, aes(x=ci, y=absmag, color=factor(stars_k$cluster))) + geom_point() + xlim(-0.5, 2.5) + ylim(18, -15) + labs(x="Color Index", y="Absolute Magnitude", title="H-R Diagram (K-means, k = 9)", color="Stellar Category")
# 0.3377312
   #labels
       #A    B    C    D    F    G    K    M    O
  #1  615 2362    0    0   73    5    0    0   17
  #2    0    0    0  191    1    1  396  148    0
  #3    3    4    0    0   29 1560 3300   26    0
  #4   47   33    1    0  140 1989 2320    7    1
  #5  626   36    0   13 6831 1233   29    1    2
  #6   10    8    2   35  880 3435  899    1    1
  #7 3519 1275    0    3  920   11    3    0    1
  #8    2    6   30    1    5   71 1351  858    0
  #9    3    1    0    1    1   52 2468  420    0
#print(table(stars_k$cluster, labels))
#dev.print(device=pdf, "k7.pdf")

# neural networks
library(nnet)

#stars_cv <- tune.nnet(cat~., data = stars_scaled[, c(2,4, 9)], size= 1:30, decay=0:5, tunecontrol = tune.control(sampling = "cross", cross = 5))
#summary(stars_cv)

#- sampling method: 5-fold cross validation

#- best parameters:
 #size decay
   #11     1

#- best performance: 0.1892569

#- Detailed performance results:
  #size decay     error  dispersion
#1    9     0 0.1908752 0.004864396
#2   10     0 0.1908491 0.003797517
#3   11     0 0.1892569 0.004902203
#4   12     0 0.1900662 0.003908252
#plot(stars_cv)
#dev.print(device=pdf, "stars_nnet_perf.pdf")

cls <- class.ind(labels)
stars_nn <- nnet(st, cls, size=11, decay=0, softmax=TRUE)
stars_nn_predict <- stats::predict(stars_nn, nst, type="class")
table(nlabels, stars_nn_predict)
adjustedRandIndex(nlabels, stars_nn_predict)

#[1] 0.6365542
       #stars_nn_predict
#nlabels    A    B    D    F    G    K    M
      #A 1266  186    2  191    6    1    0
      #B  177 1034    0   15    7    0    1
      #C    0    0    0    0    0    0    7
      #D    3    2   43   10    3    1    2
      #F   92   12    4 2653  193   14    2
      #G    2    6    2  177 2086  408    5
      #K    1    1    0   11  393 3141  112
      #M    0    0    4    0    2  211  274
      #O    4    3    0    1    0    0    0

# gpcm modelling
library(mclust)

mod1 = Mclust(st)
cc <- clustCombi(mod1, st)
table(labels, cc$classification[[9]])

adjustedRandIndex(labels, cc$classification[[9]])
#[1] 0.0815

# generalized hyperbolic distribution
library(MixGHD)

#model = MGHD(st, G=9, method="modelBased")
model = MGHD(st, G=9)

#MixGHD::predict
table(model@map, labels)
   #labels
       #D    I   II  III   IV    V
  #1    1    7  103 2757  328  679
  #2  210    2    8  185  548 4765
  #3    0  116  296 3772  482 1091
  #4   29    8   50  877 1347 6666
  #5    3   32  208 4268  803 1897
  #6    0   16  151 2060  736 2088
  #7    1  204  247  752  177  343
adjustedRandIndex(model@map, labels)
#[1] 0.09810788


