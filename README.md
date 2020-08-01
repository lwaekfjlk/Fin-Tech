# Fin-Tech
Relevant codes of Fin-Tech taught by Xiaolin Zheng is preserved.

Current available codes include:

* Web-Crawler
  * tutorial code based on **Scrapy** Framework [which I am not familiar with]
  * trivial code based on **Requests** packages [which I prefer] to grasp basic stock information from xueqiu.com
  
* Data-Processing

  * Basic Operation including data preprocessing, data cleanning and cross feature engineering

* Clustering-Classification [unfinished]
  * Clustering code including KNN, Spectral Clustering
  * Binary classification including Perceptron, Logistic Regression, Linear Regression

* Text-Classification

  * Using Dataset from *tushare*
  * Multi-class text classification and vectorizing by means of **TF-IDF** and **word2vec**

* Investment-Portfolio

  * Using Dataset from *NYSE*

  * Include models of

    * EW (naive version)

      $W = 1/n$

    * MV (*Markowitz mean-variance*) 

      ​														$min \quad W^{T}CW + \alpha \mu^{T}W$

      ​																	$s.t. \sum W_i = 1$

    * EG (*1998 Mathematical Finance EG*)

      <img src="./Investment-Portfolio/EG.png" alt="截屏2020-07-24上午2.00.37" style="zoom:50%;" width = 300 />

    * ONS (*2006 ICML ONS*)

      <img src="./Investment-Portfolio/ONS.png" alt="截屏2020-07-24上午2.00.37" style="zoom:50%;" width = 300 />
  
* Risk-Control

  * Using Feature Engineering and Classification Model to classify applicants
  * Feature Engineering includes constructing and filtering polynomial features,  normarlizing and fulfilling
  * Classification Model includes lightgbm,GBDT,LR,XGBoost
  * Best results are created by lightgbm of **72.64%** AUC

* Potential-Customer-Mining

  * Feature Engineering only include basic normalizing and fulfilling
  * Classification Model uses 8 models in sklearn
  
* **Home Credit Default Risk**

  * finished a kaggle competition named "Home Credit Default Risk", Link is https://www.kaggle.com/c/home-credit-default-risk/overview

  * Classification Model using LightGBM

  * Feature Engineering includes data precessing of 6 Datasets

    * application_train/test.csv
    * bureau.csv / bureau_balance.csv
    * previous_application.csv
    * POS_CASH_balance.csv
    * installments_payments.csv
    * credit_card_balance.csv

  * Key Points when modifying the process in order to raise the score on Kaggle

    * Better Domain Knowledge constructed feature
    * Better model parameters

  * Best Score on test data

    ![截屏2020-08-01下午7.47.36](./Home_Credit_Default_Risk/result.png)
