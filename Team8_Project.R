# set memory limits
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB

# Connect to a MariaDB version of a MySQL database
library(RMariaDB)
con <- dbConnect(RMariaDB::MariaDB(), host="datamine.rcac.purdue.edu", port=3306
                 , dbname="kag_promotion", user="kaggle_promotion", password="kaggle_pass")

# query
d <- dbGetQuery(con, "select * from train")
holdout <- dbGetQuery(con, "select * from test")
ss <- dbGetQuery(con, "select * from sample_submission")

# disconnect from db
dbDisconnect(con)

################################################################################
################################################################################
# Rename holdout
te = holdout
rm(holdout,con)


# kick index column and id for d
d$index = NULL
te$index = NULL
ss$index = NULL

d$employee_id = NULL


# which variable got NA
sapply(d, function(x) sum(is.na(x))) #  education and rating
sapply(te, function(x) sum(is.na(x))) #  education and rating


# fill NA
# create a new level called "unknown" 
d$education = ifelse(is.na(d$education), "unknown", d$education)
te$education = ifelse(is.na(te$education), "unknown", te$education)


#  3 is the median for previous_year_rating
median(d$previous_year_rating, na.rm = TRUE) 
d$previous_year_rating = ifelse(is.na(d$previous_year_rating), 
                                3, 
                                d$previous_year_rating)
te$previous_year_rating = ifelse(is.na(te$previous_year_rating), 
                                 3, 
                                 te$previous_year_rating)


# coerce data type
for (i in c(1:5,13)){
  d[,i] = as.factor(d[,i])
}

for (i in 6:12){
  d[,i] = as.numeric(d[,i])
}


str(d)

for (i in 2:6){
  te[,i] = as.factor(te[,i])
}

for (i in c(1,7:13)){
  te[,i] = as.numeric(te[,i])
}

str(te)

rm(i)

d = d[,c(13,1:12)]
names(d)[1] = "y"


# interesting about data
library(ggplot2)
ggplot(d, aes(x = previous_year_rating, y = avg_training_score)) + geom_point(aes(color = department))


################################################################################
################################################################################
# create dummies
library(caret)

dummies <- dummyVars(y ~ ., data = d)
ex <- data.frame(predict(dummies, newdata = d))
names(ex) <- gsub("\\.", "", names(ex))
d <- cbind(d$y, ex)
names(d)[1] <- "y"
rm(dummies, ex)

dummies2 = dummyVars("~.",data = te)
ex2 <- data.frame(predict(dummies2, newdata = te))
names(ex2) <- gsub("\\.", "", names(ex2))
te <- ex2
rm(dummies2, ex2)


# identify and remove correlated variable corr = 0.85
descrCor <-  cor(d[,2:ncol(d)])
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85)
summary(descrCor[upper.tri(descrCor)])

highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
filteredDescr <- d[,2:ncol(d)][,-highlyCorDescr]
descrCor2 <- cor(filteredDescr)
summary(descrCor2[upper.tri(descrCor2)])

d <- cbind(d$y, filteredDescr)
names(d)[1] <- "y"

rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)


# identify and remove variable with linear dependencies 
y <- d$y
d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
names(d)[1] <- "ones"

comboInfo <- findLinearCombos(d)

d <- d[, -comboInfo$remove]
d <- d[, c(2:ncol(d))]
d <- cbind(y, d)

rm(y, comboInfo)


# remove features with limited variation
nzv <- nearZeroVar(d, saveMetrics = TRUE)
head(nzv)
d <- d[, c(TRUE,!nzv$nzv[2:ncol(d)])]
rm(nzv)


# standardize numeric features using a min-max
preProcValues <- preProcess(d[,2:ncol(d)], method = c("range"))
d <- predict(preProcValues, d)

preProcValues2 = preProcess(te[,2:ncol(te)], method = c("range"))
te = predict(preProcValues2, te)

levels(d$y) = make.names(levels(factor(d$y)))

rm(preProcValues, preProcValues2)


# 80/20 train/test
inTrain <- createDataPartition(y = d$y,
                               p = .80,
                               list = F)

train <- d[inTrain,]
test <- d[-inTrain,]

rm(inTrain)


# 5-fold cross-validation design
ctrl <- trainControl(method="cv",
                     number=5,
                     classProbs = T,
                     summaryFunction = twoClassSummary,
                     allowParallel=T)


# define model using generalized linear model
model <- train(y ~ .,
            data = train,
            method = "glm",
            trControl = ctrl,
            family = "binomial",
            metric = "ROC")


# check model
library(e1071)
options(scipen=999, digits = 3)
defaultSummary(data=data.frame(obs=train$y, pred=predict(model, newdata=train))
               , model=model)

defaultSummary(data=data.frame(obs=test$y, pred=predict(model, newdata=test))
               , model=mode)


# predict
preds = predict(model, newdata = te)
results = data.frame(cbind(te[,1], preds))

rm(preds)


# compare with actual result
comp = data.frame(cbind(ss,results[,2]))
names(comp) = c("employee_id", "y", "y_hat")

comp$y_hat = ifelse(comp$y_hat == 1, 0, 1)
comp[,2] = as.numeric(comp[,2])

# number of wrong prediction
wrong = sum(comp$y_hat)
correctness = 1 - (sum(comp$y_hat) / nrow(ss))

summary(model)


################################################################################
################################################################################
# h2o
library(h2o)

h2o.init(nthreads=12, max_mem_size="64g")
d_h2o <- as.h2o(d)


# split d and assign x, y
y = "y"
x <- setdiff(names(d_h2o), y)
parts <- h2o.splitFrame(d_h2o, 0.8, seed=99)
train_h2o <- parts[[1]]
test_h2o <- parts[[2]]


# using deep learning model
model_dl = h2o.deeplearning(x, y, train_h2o)
h2o.performance(model_dl, train_h2o)
h2o.performance(model_dl, test_h2o)

#             Train         Test
#   MSE       0.0505        0.0516
#   RMSE      0.225         0.227

#   not over fit

# predict
te_h2o = as.h2o(te)
results_h2o <- h2o.predict(model_dl, te_h2o)

results_h2o = as.data.frame(results_h2o)
comp_h2o = data.frame(cbind(ss,results_h2o[,1]))
names(comp_h2o) = c("employee_id", "y", "y_hat")
comp_h2o$y_hat = ifelse(comp_h2o$y_hat == 'X0', 0, 1)

summary(model_dl)

# number of wrong prediction
wrong_h2o = sum(comp_h2o$y_hat)
correctness_h2o = 1 - (sum(comp_h2o$y_hat) / nrow(ss))


################################################################################
################################################################################
# shiny

options(digits = 2)
Compare = matrix(c(wrong,correctness,wrong_h2o,correctness_h2o), nrow = 2, ncol =2)
rownames(Compare) = c("wrong","Correct%")
colnames(Compare) = c("glm","dpm")




library(shiny)

ui = fluidPage(
  sidebarPanel(
    numericInput(inputId = 'ana',
                label = 'Do you work in department of analytics?',
                0, min = 0, max = 1),
    numericInput(inputId = 'opr',
                label = 'Do you work in department of operation?',
                0, min = 0, max = 1),
    numericInput(inputId = 'pro',
                label = 'Do you work in department of procurement?',
                0, min = 0, max = 1),
    numericInput(inputId = 'mar',
                label = 'Do you work in department of marketing?',
                1, min = 0, max = 1),
    numericInput(inputId = 'r2',
                label = 'Are you from region_2?',
                0, min = 0, max = 1),
    numericInput(inputId = 'r22',
                label = 'Are you from region_22?',
                0, min = 0, max = 1),
    numericInput(inputId = 'r7',
                label = 'Are you from region_7?',
                0, min = 0, max = 1),
    numericInput(inputId = 'mas',
                label = 'Do you have a master degree or above?',
                1, min = 0, max = 1),
    numericInput(inputId = 'not',
                label = 'how many times have you attended the training?',
                2),
    sliderInput(inputId = 'age',
                 label = 'How old are you',
                 35, min = 20, max = 60),
    sliderInput(inputId = 'rat',
                label = 'what is your previous year rating',
                3, min = 1, max = 5),
    numericInput(inputId = 'KPI',
                label = 'Have you reached 80% of your KPI?',
                1, min = 0, max = 1),
    sliderInput(inputId = 'sco',
                 label = 'What is your training score?',
                 65, min = 39, max = 99)
  ),
  mainPanel(
    verbatimTextOutput('Text_0'),
    verbatimTextOutput('Text_1'),
    verbatimTextOutput('Text_2'),
    verbatimTextOutput('Text_3')
  )
)

server = function(input, output){
  ana = reactive({
    input$ana
  })
  opr = reactive({
    input$opr
  })
  pro = reactive({
    input$pro
  })
  mar = reactive({
    input$mar
  })
  r2 = reactive({
    input$r2
  })
  r22 = reactive({
    input$r22
  })
  r7 = reactive({
    input$r7
  })
  mas = reactive({
    input$mas
  })
  not = reactive({
    input$not
  })
  age = reactive({
    input$age
  })
  rat = reactive({
    input$rat
  })
  KPI = reactive({
    input$KPI
  })
  sco = reactive({
    input$sco
  })
  output$Text_0 = renderPrint({
    odd = exp(-9.77-1.36*ana()+2.31*opr()+1.01*pro()+4*mar()+0.17*r2()+0.42*r22()+0.35*r7()+0.15*mas()-1.88*(not()-1)/9-0.96*(age()-20)/40+1.22*(rat()-1)/4+1.52*KPI()+9.37*(sco()-39)/60)
    p = odd/(odd+1)
    p
  })
  output$Text_1 = renderPrint({
    Compare
  })
  output$Text_2 = renderPrint({
    summary(model)
  })
  output$Text_3 = renderPrint({
    model_dl@model[["variable_importances"]]
  })

}

shinyApp(ui = ui, server = server)


h2o.shutdown()
