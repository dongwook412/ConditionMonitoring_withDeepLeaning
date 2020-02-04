#install.packages("waveslim")
library(waveslim)
library(keras)
setwd("C:\\Users\\Yang\\Desktop\\data")

# 데이불 불러오기
PS <- as.matrix(read.table(file="PS1.txt", sep="\t")) # 압력, 100HZ
EPS <- as.matrix(read.table(file="EPS1.txt", sep="\t")) # 전력, 100HZ
FS <- as.matrix(read.table(file="FS1.txt", sep="\t")) # 유량, 10HZ
TS <- as.matrix(read.table(file="TS1.txt", sep="\t")) # 온도, 1HZ
VS <- as.matrix(read.table(file="VS1.txt", sep="\t")) # 진동, 1HZ
CE <- as.matrix(read.table(file="CE.txt", sep="\t")) # Cooling 효율, 1HZ
CP <- as.matrix(read.table(file="CP.txt", sep="\t")) # Cooling 파워, 1HZ
SE <- as.matrix(read.table(file="SE.txt", sep="\t")) # 효율성 지표, 1HZ
y <- as.matrix(read.table(file="profile.txt", sep="\t")) # 기계 상태(쿨러, 밸브, 펌프 누출, 누산기 상태)



# Noise 제거 함수 
noise_del <- function(x, level, thresholding){
  result <- NULL 
  for(i in 1:length(x[,1])){
    ## 노이즈 제거
    temp <- dwt(x[i,], wf="haar", n.levels=level) # 웨이블릿 변환
    sigma = sqrt(2*log(length(x)))*sd(temp$d1) ## universal threshold
    
    # Thresholding Rule, 인수인 Thresholding 값이 Hard, Soft에 따라 달라짐
    for(i in 1:level){
      temp[[i]][which(abs(temp[[i]])<sigma)] = 0
      
      if(thresholding == "Soft"){
        temp[[i]][which(abs(temp[[i]])>=sigma)] = temp[[i]][which(abs(temp[[i]])>=sigma)] - 
          (sign(temp[[i]][which(abs(temp[[i]])>=sigma)])*sigma)
      }
    }
    inv_temp <- idwt(temp)
    result <- rbind(result, inv_temp)
  }
  return(result)
}


# 2차원 데이터 -> 구간으로 나누어서 3차원 데이터로 변경  
split_data = function(x,num_data){ 
  t <- array(dim=c(dim(x)[[1]], dim(x)[[2]]/num_data, num_data))
  for (i in 1:length(x[,1])){
    w <- data.frame()
    for (j in 1:(length(x[1,])/num_data)){
      q <- c(x[i,(num_data*j-(num_data-1)):(num_data*j)])
      w <- rbind(w, q)
      w <- as.matrix(w)
      w <- as.array(w)
    }
    t[i,,] <- w
  }
  return(t)
}

# 여러 클래스를 가진 y중 하나에 대해서 특정 정수 라벨로 변경하고, one-hot encoding 진행한 후 리턴하는 함수 
y_selection <- function(y, select_number, dimension){
  y[,1] <- replace(y[,1], y[,1] == 3, 0)
  y[,1] <- replace(y[,1], y[,1] == 20, 1)
  y[,1] <- replace(y[,1], y[,1] == 100, 2)
  
  y[,2] <- replace(y[,2], y[,2] == 73, 0)
  y[,2] <- replace(y[,2], y[,2] == 80, 1)
  y[,2] <- replace(y[,2], y[,2] == 90, 2)
  y[,2] <- replace(y[,2], y[,2] == 100, 3)
  
  y[,4] <- replace(y[,4], y[,4] == 90, 0)
  y[,4] <- replace(y[,4], y[,4] == 100, 1)
  y[,4] <- replace(y[,4], y[,4] == 115, 2)
  y[,4] <- replace(y[,4], y[,4] == 130, 3)
  
  y_new <- y[,select_number]
  label <- to_one_hot(y_new, dimension)
  return(label)
}
to_one_hot <- function(labels, dimension){
  results <- matrix(0, nrow=length(labels), ncol=dimension)
  for (i in 1:length(labels)){
    results[i, labels[[i]]+1] <- 1
  }
  return(results)
}
 
## 데이터 함수 적용 및 split
x <- CP # 어떤 데이터 사용할지
level <- 2 #웨이블릿 함수 몇수준까지 할건지 
thresholding <- "Soft" # Thresholding rule 인수, "Hard", "Soft"
y_label <- c(1,2,3,4,5) # y값의 모든 클래스
y_dimension <- c(3,4,3,4,2) # 각 클래스에 대한 라벨의 종류 갯수
y_index <- 5 # y값 중 몇 번째 클래스를 불러올건지

result <- noise_del(x, level, thresholding)
label <- y_selection(y, y_label[y_index], y_dimension[y_index])


set.seed(1234) 
idx <- sample(x = c("train", "test"),
              size = nrow(result),
              replace = TRUE,
           prob = c(8,2))

x_train <- result[idx == "train",]
x_train <- split_data(x_train, 6)
x_test <- result[idx == "test", ]
x_test <- split_data(x_test, 6)

y_train <- label[idx == "train", ]
y_test <- label[idx == "test", ]


## 모델
model <- keras_model_sequential() %>%
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2,input_shape = list(dim(x_train)[[2]], dim(x_train)[[3]])) %>%
  layer_dense(units = y_dimension[y_index], activation="softmax")

model %>% compile(optimizer = optimizer_sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov = TRUE), 
                  loss = "categorical_crossentropy", metrics =c("accuracy"))
history <- model %>% fit(x_train, y_train, epochs=200, batch_size = 64, validation_data = list(x_test, y_test))



#par(mfcol=c(2,1), pty="m", mar=c(2,2,2,2))
#plot(1:length(x[1,]), x[1,], type="l")
#plot(1:length(x[1,]), result[1,], type="l")


## 특징 추출 부분을 사용해보았으나 detail부분이 모두 0인 경우가 많아서 제외하였음.
# x <- x_train
# func <- "haar"
# level <- 2
# 
# value = array(dim=c(dim(x)[[1]], dim(x)[[2]], 2))
# for(d in 1:1){ # 1:dim(x)[[1]]
#   for(i in 1:dim(x)[[2]]){
#     temp = dwt(x[d,i,], func, level)
#     x.axis = 1:level
#     y.axis = c()
#     for(j in 1:level){
#       y.axis[j] = log2(mean(temp[[j]]^2))
#     }
#     
#     state = TRUE # d1, d2가 0인 경우가 매우 많아서 이럴때는 y.axis가 -inf되므로 regression을 돌리지못하게돼서 추가함.
#     for(k in 1:level){
#       if(y.axis<0){
#         state = FALSE
#       }
#     }
#     if(state){
#       reg = lm(y.axis ~ x.axis)
#       value[d,i,] = as.array(as.matrix(coef(reg)))
#     }else{
#       value[d,i,] = as.array(matrix(0, nrow=1, ncol=2))
#     }
#   }  
# }

















# model <- keras_model_sequential() %>%
#   layer_dense(units = 128, activation = "relu", input_shape = c(2)) %>%
#   layer_dense(units = 64, activation = "relu") %>%
#   layer_dense(units = 32, activation = "relu") %>%
#   layer_dense(units = 32, activation = "relu") %>%
#   layer_dense(units=3, activation="softmax")
# model %>% compile(optimizer = optimizer_rmsprop(), loss = "categorical_crossentropy", metrics =c("accuracy"))
# history <- model %>% fit(x_train, y_train, epochs=20, batch_size = 64, validation_data = list(x_valid, y_valid))
