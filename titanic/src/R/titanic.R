library(rpart)

data <- read.csv("../../data/titanic.csv")

# Q1. Number of male & female
q1.sex <- table(data['Sex'])
q1.res <- q1.sex[c('male', 'female')]
cat(q1.res, file="titanic_q1.txt")

# Q2. Survived percent
q2.survived <- table(data['Survived'])
q2.percent <- q2.survived['1'] / nrow(data) * 100
cat(sprintf("%1.2f", q2.percent), file="titanic_q2.txt")

# Q3. First class percent
q3.pclass <- table(data['Pclass'])
q3.percent <- q3.pclass['1'] / nrow(data) * 100
cat(sprintf("%1.2f", q3.percent), file="titanic_q3.txt")

# Q4. Age average & median
q4.age <- data$Age
q4.mean <- mean(q4.age, na.rm = TRUE)
q4.median <- median(q4.age, na.rm = TRUE)
q4.list <- c(q4.mean, q4.median)
cat(sprintf("%1.2f", q4.list), file="titanic_q4.txt")

# Q5. SibSp & Parch correlation
q5.corr <- cor(data$SibSp, data$Parch)
cat(sprintf("%1.2f", q5.corr), file="titanic_q5.txt")

# Q6. Most popular female name
q6.female <- subset(data, Sex == "female", value=TRUE)
q6.female <- sub("[\\w '-]+, Mrs\\. \\w+ [\\w ]*\\((\\w+).*", "\\1", q6.female$Name, perl=TRUE)
q6.female <- sub("[\\w '-]+, (Mrs\\.|Miss\\.|Lady\\.|the Countess\\. of) \\((\\w+).*", "\\2", q6.female, perl=TRUE)
q6.female <- sub("[\\w '-]+, (Mrs|Miss|Mme|Ms|Mlle)\\. (\\w+).*", "\\2", q6.female, perl=TRUE)
q6.list <- data.frame(q6.female$Name, q6.female)
q6.stat <- table(grep("\\w+", q6.female, perl=TRUE, value=TRUE))
q6.name <- names(which.max(q6.stat))
cat(q6.result, file="titanic_q6.txt")

# Q7. Features importance
q7.data <- subset(data, select=c('Survived', 'Pclass', 'Age', 'Fare', 'Sex'))
q7.data <- na.omit(q7.data)
q7.data$Gender[q7.data$Sex == 'female'] <- 0
q7.data$Gender[q7.data$Sex == 'male'] <- 1
q7.data$Sex <- NULL
names(q7.data)[names(q7.data) == 'Gender'] <- 'Sex'

formula = Survived ~ Pclass + Age + Fare + Sex
set.seed(241)
fit <- rpart(formula, method = "class", data = q7.data)

summary <- summary(fit)$variable.importance

result <- sprintf("%s, %s", names(summary)[1], names(summary)[2])
cat(result, file="titanic_q7.txt")
