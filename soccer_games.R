library(tidyverse)
library(Metrics)
library(ggplot2)
library(ggthemr)
library(tidymodels)
library(parsnip)
library(poissonreg)
library(caret)

#read data
#https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017
data <- read_csv("/results.csv")

#list of competing countries
euro2020_teams <- c("Austria", "Germany", "Poland", "Netherlands", 
                    "Italy", "England", "Scotland", "Portugal", 
                    "France", "Belgium", "Czech Republic", "Slovakia",
                    "Hungary", "Wales", "Sweden", "Northern Ireland",
                    "Turkey", "Switzerland", "Russia", "Finland",
                    "Croatia", "Spain","Ukraine", "Denmark")


#https://www.kaggle.com/cashncarry/fifaworldranking?select=fifa_ranking-2021-05-27.csv
rankings_data <- read_csv("fifa_ranking-2021-05-27.csv")
rankings_data <- rankings_data %>% filter(rank_date ==  '2021-05-27') %>% select(country_full, total_points)
rankings_data$country_full[rankings_data$country_full == "Côte d'Ivoire"] <- 'Ivory Coast'
rankings_data$country_full[rankings_data$country_full == "USA"] <- 'United States'

#filtering out all matches before 2019 and with non-competing countries
data <- data %>% 
  filter(date > "2019-01-01") %>%
  filter(home_team %in% euro2020_teams | away_team %in% euro2020_teams) %>%
  filter(!is.na(home_score)) %>% 
  arrange(date, desc=TRUE)

#add rankings to data
data <- data %>% 
            left_join(rankings_data, by = c("home_team" = "country_full")) %>% rename(home_team_ranking=total_points) %>%
            left_join(rankings_data, by = c("away_team" = "country_full")) %>% rename(away_team_ranking=total_points)
  
#add fifa ranking difference variable
data <- data %>% mutate(ranking_diff = home_team_ranking-away_team_ranking)


#EDA -------
#home score
ggthemr('fresh')
ggplot(data, aes(home_score)) +
  geom_histogram(binwidth=1, color="white") + 
  ggtitle("Goals scored by the home team") +
  scale_x_continuous(name="Goals", breaks=0:10) +
  geom_vline(aes(xintercept = mean(home_score)), color="red") +
  geom_text(aes(label = round(mean(home_score),1), x= mean(home_score)+0.3, y=100), size=4, color="red")

#home score
ggplot(data, aes(away_score)) +
  geom_histogram(binwidth=1, color="white") + 
  ggtitle("Goals scored by the away team") +
  scale_x_continuous(name="Goals", breaks=0:7) + 
  geom_vline(aes(xintercept = mean(away_score)), color="red") +
  geom_text(aes(label = round(mean(away_score),1), x= mean(away_score)+0.3, y=140), size=4, color="red")


#all score
ggplot(data, aes(away_score+home_score)) +
  geom_histogram(binwidth=1, color="white") + 
  ggtitle("Goals scored per match") +
  scale_x_continuous(name="Goals", breaks=0:10) +
  geom_vline(aes(xintercept = mean(away_score+home_score)), color="red")
  

data %>% 
  ggplot(aes(ranking_diff, home_score)) + 
  geom_point() + 
  xlab("FIFA Ranking Difference between home and away team") + 
  ylab("Goals scored by home team") +
  ggtitle("FIFA Ranking Difference of teams vs. home team goals")

data %>% 
  ggplot(aes(ranking_diff, away_score)) + 
  geom_point() + 
  xlab("FIFA Ranking Difference between home and away team") + 
  ylab("Goals scored by away team") +
  ggtitle("FIFA Ranking Difference of teams vs. away team goals")
  

euro_data <- data %>% filter(tournament == "UEFA Euro")
data <- data %>% filter(tournament != "UEFA Euro")
  
#Machine Learning ------
train_test_split <- initial_split(data, prop = 0.8)
data_train <- training(train_test_split)
data_test <- testing(train_test_split)

#set up models
pois_mod <- poisson_reg() %>%  set_engine("glm")
lin_mod <- linear_reg() %>% set_engine("lm")
rf_mod <-  rand_forest(trees=100) %>% set_engine("randomForest") %>% set_mode("regression")

#set up cv
folds <- vfold_cv(data_train, v = 30)

#set up model formulas
hs_formula_single <- formula("home_score ~ ranking_diff")
as_formula_single <- formula("away_score ~ ranking_diff")

#poisson model home
pois_mod_hs_wf <- 
  workflow() %>%
  add_model(pois_mod) %>%
  add_formula(hs_formula_single)

#poisson model away
pois_mod_as_wf <- 
  workflow() %>%
  add_model(pois_mod) %>%
  add_formula(as_formula_single)

#lin model home
lin_mod_hs_wf <- 
  workflow() %>%
  add_model(lin_mod) %>%
  add_formula(hs_formula_single)

#lin model away
lin_mod_as_wf <- 
  workflow() %>%
  add_model(lin_mod) %>%
  add_formula(as_formula_single)

#rf model away
rf_mod_as_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(hs_formula_single)

#rf model away
rf_mod_hs_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(as_formula_single)


set.seed(23)

#perform cross validation for each model
pois_mod_hs_rs <- 
  pois_mod_hs_wf %>% 
  fit_resamples(folds)

pois_mod_as_rs <- 
  pois_mod_as_wf %>% 
  fit_resamples(folds)

lin_mod_hs_rs <- 
  lin_mod_hs_wf %>% 
  fit_resamples(folds)

lin_mod_as_rs <- 
  lin_mod_as_wf %>% 
  fit_resamples(folds)

rf_mod_as_rs <- 
  rf_mod_as_wf %>% 
  fit_resamples(folds)

rf_mod_hs_rs <- 
  rf_mod_hs_wf %>% 
  fit_resamples(folds)

#compare cv results ---
collect_metrics(pois_mod_hs_rs)
collect_metrics(pois_mod_as_rs)
collect_metrics(lin_mod_hs_rs)
collect_metrics(lin_mod_as_rs)
collect_metrics(rf_mod_hs_rs)
collect_metrics(rf_mod_as_rs)

#fit final models ----
rf_mod_hs <- rf_mod %>% fit(hs_formula_single, data = data)
rf_mod_as <- rf_mod %>% fit(as_formula_single, data = data)

lin_mod_hs <- lin_mod %>% fit(hs_formula_single, data = data)
lin_mod_as <- lin_mod %>% fit(as_formula_single, data = data)

pois_mod_hs <- pois_mod %>% fit(hs_formula_single, data = data)
pois_mod_as <- pois_mod %>% fit(as_formula_single, data = data)

#predict scores ----
pois_hs_res <- unname(unlist(predict(pois_mod_hs, data %>% select(ranking_diff))))
pois_as_res <- unname(unlist(predict(pois_mod_as, data %>% select(ranking_diff))))

lm_hs_res   <- unname(unlist(predict(lin_mod_hs, data %>% select(ranking_diff))))
rf_hs_res   <- unname(unlist(predict(rf_mod_hs, data %>% select(ranking_diff))))


#compare models visually
ggplot(data %>% select(ranking_diff, home_score), aes(ranking_diff,home_score)) +
  geom_point() +
  geom_line(aes(ranking_diff, pois_hs_res), color="red", size=1) + 
  geom_line(aes(ranking_diff, lm_hs_res), color="green", size=1) + 
  geom_line(aes(ranking_diff, rf_hs_res), color="blue", size=1) +
  xlab("FIFA Ranking Difference of teams vs. away team goals") + 
  ylab("Goals scored by home team") +
  ggtitle("Comparison of Models")
  

#add results column (reclassify match outcomes into win,lose,draw)
data <- data %>% mutate(result=case_when(home_score > away_score ~ 0, 
                                                     home_score < away_score ~ 1,
                                                     home_score == away_score ~ 2))



#reclassify predicted goals into win,lose,draw by rounding and comparison
predicted_result <- rep(0, length(pois_hs_res))

predicted_result[round(pois_hs_res) > round(pois_as_res)] <- 0
predicted_result[round(pois_hs_res) < round(pois_as_res)] <- 1
predicted_result[round(pois_hs_res) == round(pois_as_res)] <- 2

#analyse performance (accuracy and f1)
data = data %>% mutate(predicted_result = as.factor(predicted_result))
data = data %>% mutate(result = as.factor(result))

performance <- confusionMatrix(
                data=as.factor(predicted_result), 
                reference=as.factor(data$result))


cm <- conf_mat(data, result, predicted_result)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  ggtitle("Confusion Matrix of match outomes (with rounding)") +
  scale_x_discrete(labels=c("Home Team Win","Away Team Win","Draw")) + 
  scale_y_discrete(labels=c("Draw","Away Team Win","Home Team Win"))


#macro f1
sum(performance[["byClass"]][ , "F1"])/3
#weighted f1
sum(performance[["byClass"]][ , "F1"]*colSums(performance$table))/sum(colSums(performance$table))
#accuracy
performance$overall["Accuracy"]


#calculate f1 scores for different thresholds
diffs <- seq(0.01,2,0.01)
weighted_f1_scores <- rep(NA,length(diffs))


for(i in 1:length(diffs)){
  
  predicted_result_enh <- rep(NA, length(pois_hs_res))
  
  predicted_result_enh[pois_hs_res > pois_as_res+diffs[i]] <- 0
  predicted_result_enh[pois_hs_res < pois_as_res-diffs[i]] <- 1
  predicted_result_enh[!(pois_hs_res > pois_as_res+diffs[i]) & !(pois_hs_res < pois_as_res-diffs[i])] <- 2
  
  performance <- confusionMatrix(
    as.factor(predicted_result_enh), 
    as.factor(data$result))
  
  weighted_f1 <- sum(performance[["byClass"]][ , "F1"]*colSums(performance$table))/sum(colSums(performance$table))
  weighted_f1_scores[i] <- weighted_f1
}

ggplot(as_tibble(diffs, weighted_f1_scores), aes(diffs,weighted_f1_scores)) + 
  geom_point() + 
  geom_line() +
  geom_vline(aes(xintercept=diffs[which.max(weighted_f1_scores)]), color="red")+
  geom_text(aes(label = diffs[which.max(weighted_f1_scores)], x= diffs[which.max(weighted_f1_scores)]+0.1, y=0.7), size=4, color="red") + 
  ggtitle("F1-Scores for goal difference thesholds") +
  ylab("Weighted F1-Scores") +
  xlab("Goal Difference Thresholds")


predicted_result_enh <- rep(NA, length(pois_hs_res))

diff <- 0.41

predicted_result_enh[pois_hs_res > pois_as_res+diff] <- 0
predicted_result_enh[pois_hs_res < pois_as_res-diff] <- 1
predicted_result_enh[!(pois_hs_res > pois_as_res+diff) & !(pois_hs_res < pois_as_res-diff)] <- 2

data = data %>% mutate(predicted_result_enh = as.factor(predicted_result_enh))

cm <- conf_mat(data, result, predicted_result_enh)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  ggtitle("Confusion Matrix of match outomes (with difference threshold)") +
  scale_x_discrete(labels=c("Home Team Win","Away Team Win","Draw")) + 
  scale_y_discrete(labels=c("Draw","Away Team Win","Home Team Win"))


performance <- confusionMatrix(
  data=as.factor(predicted_result_enh), 
  reference=as.factor(data$result))

#weighted f1
sum(performance[["byClass"]][ , "F1"]*colSums(performance$table))/sum(colSums(performance$table))



#------------
#EXPORE EURO 2020 MATCHES
#-----------

euro_data <- euro_data %>% mutate(result=case_when(home_score > away_score ~ 0, 
                                                  home_score < away_score ~ 1,
                                                  home_score == away_score ~ 2))

euro_data_short <- euro_data %>% select(home_team, away_team, home_score, away_score, result)


#predict euro 2020
pois_hs_res_euro <- unname(unlist(predict(pois_mod_hs, euro_data %>% select(ranking_diff))))
pois_as_res_euro <- unname(unlist(predict(pois_mod_as, euro_data %>% select(ranking_diff))))
predicted_result_euro <- rep(0, length(pois_hs_res_euro))

#reclassify predicted goals into win,lose,draw by rounding
predicted_result_euro[round(pois_hs_res_euro) > round(pois_as_res_euro)] <- 0
predicted_result_euro[round(pois_hs_res_euro) < round(pois_as_res_euro)] <- 1
predicted_result_euro[round(pois_hs_res_euro) == round(pois_as_res_euro)] <- 2

#check performance
performance <- confusionMatrix(as.factor(predicted_result_euro), 
                               as.factor(euro_data_short$result))


sum(performance[["byClass"]][ , "F1"]*colSums(performance$table))/sum(colSums(performance$table))


#f1 with different threshold (from above)
predicted_result_enh <- rep(NA, length(pois_hs_res_euro))
diff <- 0.41

predicted_result_enh[pois_hs_res_euro > pois_as_res_euro+diff] <- 0
predicted_result_enh[pois_hs_res_euro < pois_as_res_euro-diff] <- 1
predicted_result_enh[!(pois_hs_res_euro > pois_as_res_euro+diff) & !(pois_hs_res_euro < pois_as_res_euro-diff)] <- 2

performance <- confusionMatrix(
  as.factor(predicted_result_enh), 
  as.factor(euro_data_short$result))

sum(performance[["byClass"]][ , "F1"]*colSums(performance$table))/sum(colSums(performance$table))


euro_data_short <-
euro_data_short %>%
  mutate(home_score_predicted = pois_hs_res_euro) %>%
  mutate(away_score_predicted = pois_as_res_euro) %>%
  mutate(result_predicted=predicted_result_enh)


write_csv(euro_data_short, "euro_res.csv")

  