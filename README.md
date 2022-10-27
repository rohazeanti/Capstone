![banner](https://media.git.generalassemb.ly/user/43213/files/03225bb2-7904-4643-ae05-420c6841e721)

##  Capstone: Fake News Detection using Traditional Machine Learning

#### _Rohazeanti Mohamad Jenpire_

### Problem Statement
Many fake news and data are disseminated online, particularly on social feeds and conversation groups. Fake news is information, stories or hoaxes made to misinform or mislead audiences. These types of narratives are made to influence people’s views, push the political agenda, create embarrassment, and get gains out of online publishers. The proliferation of news on social media and the Internet is deceiving people to an extent which needs to be stopped. 

This project aims to deploy a highly accurate model that classifies a given news article as either fake or true, allowing consumers to check for news reliability through their browsers efficiently. Stakeholders: Government, Journalists

As a data scientist engaged by the Government, I am tasked to create a fake news  detection technique, using Machine Learning model, for members of public to determine the authenticity of a given news. 


### Executive Summary
The ISOT Fake News Dataset contains over 40,000 articles between 2015-2018.  Text data were aggreggated into a single data frame and data was cleaned according to standard protocol (handling null values, confirming data types, and searching for any abnormal values). Because text data can contain many symbols and phrases that do not add value to a sentence, any symbols or tags were removed to the best of my abilities. Basic features were engineered to allow for examination of full text in the article, article length in characters and article length in words. Many had empty text but title remained. Hence, I combined both title and text to create a new feature called title_text. 

Once the data had been clean, text preprocessing was conducted to expand contractions in the text, and posts were lemmetized. In order to be compatible with the text vectorizers, the lemmetized words were joined as a string. At this point, the data frame was considered ready for exploratory data analysis and modeling. The data dictionary can be found below.

During EDA, distributions for all numeric features were explored, the most frequent word counts were generated per label. 

Following EDA, the data was prepared for modeling. Tfidf Vectorizer and Count Vectorizer were used in all the model. A 70/30 train test split was conducted, and a column transformer with the two vectorizers and top 20 high frequency words were extracted and added into stop_words library. 

During the modeling process, a pipeline was created. Pipelines contained the column transformer (Tfidf Vectorizer/Count Vectorizer) and a classification model. RandomSearchCV was used to find the best params. The models built were using tradition Machine Learning:

* LogisticRegression
* RandomForestClassifier
* DecisionTreeClassifier
* MultinomialNB
* KNeighborsClassifier
* AdaBoostClassifier
* GradientBoostingClassifier

Summary of Classification Results
|                                            |   Train score |   Test score |   Generalisation |   Accuracy |   Precision |   Recall |   Specificity |    F1 |   ROC AUC | Execution Time   |
|:-------------------------------------------|--------------:|-------------:|-----------------:|-----------:|------------:|---------:|--------------:|------:|----------:|:-----------------|
| LogisticRegression_TfidfVectorizer         |         0.982 |        0.975 |            0.713 |      0.975 |       0.976 |    0.969 |         0.98  | 0.972 |    0.9962 | 00:38:14         |
| RandomForestClassifier_TfidfVectorizer     |         1     |        0.965 |            3.5   |      0.965 |       0.974 |    0.949 |         0.978 | 0.961 |    0.9943 | 00:45:47         |
| DecisionTreeClassifier_TfidfVectorizer     |         0.973 |        0.913 |            6.166 |      0.913 |       0.903 |    0.907 |         0.917 | 0.905 |    0.9184 | 00:46:01         |
| MultinomialNB_TfidfVectorizer              |         0.924 |        0.923 |            0.108 |      0.923 |       0.912 |    0.922 |         0.925 | 0.917 |    0.9765 | 00:10:56         |
| KNeighborsClassifier_TfidfVectorizer       |         1     |        0.846 |           15.4   |      0.846 |       0.785 |    0.913 |         0.788 | 0.844 |    0.9275 | 01:00:05         |
| AdaBoostClassifier_TfidfVectorizer         |         0.99  |        0.98  |            1.01  |      0.98  |       0.982 |    0.975 |         0.985 | 0.978 |    0.9973 | 02:15:50         |
| GradientBoostingClassifier_TfidfVectorizer |         0.985 |        0.964 |            2.132 |      0.964 |       0.964 |    0.958 |         0.969 | 0.961 |    0.9917 | 01:11:21         |

|                                            |   Train score |   Test score |   Generalisation |   Accuracy |   Precision |   Recall |   Specificity |    F1 |   ROC AUC | Execution Time   |
|:-------------------------------------------|--------------:|-------------:|-----------------:|-----------:|------------:|---------:|--------------:|------:|----------:|:-----------------|
| LogisticRegression_CountVectorizer         |         1     |        0.976 |            2.4   |      0.976 |       0.977 |    0.971 |         0.981 | 0.974 |    0.9946 | 00:27:59         |
| RandomForestClassifier_CountVectorizer     |         1     |        0.967 |            3.3   |      0.967 |       0.975 |    0.953 |         0.98  | 0.964 |    0.995  | 00:59:59         |
| DecisionTreeClassifier_CountVectorizer     |         0.969 |        0.919 |            5.16  |      0.919 |       0.921 |    0.901 |         0.935 | 0.911 |    0.9226 | 00:49:16         |
| MultinomialNB_CountVectorizer              |         0.934 |        0.933 |            0.107 |      0.933 |       0.912 |    0.946 |         0.922 | 0.929 |    0.9698 | 00:15:08         |
| KNeighborsClassifier_CountVectorizer       |         1     |        0.788 |           21.2   |      0.788 |       0.749 |    0.809 |         0.771 | 0.778 |    0.8453 | 01:32:25         |
| AdaBoostClassifier_CountVectorizer         |         0.988 |        0.979 |            0.911 |      0.979 |       0.984 |    0.97  |         0.986 | 0.977 |    0.9973 | 01:34:10         |
| GradientBoostingClassifier_CountVectorizer |         0.981 |        0.966 |            1.529 |      0.966 |       0.969 |    0.956 |         0.974 | 0.962 |    0.9924 | 01:31:48         |


Metrics used when evaluating the performance of the model.
* Precision
* Recall
* Specificity
* F1
* ROC AUC
* Confusion Matrix
* Execution Time

* Confusion Matrix: Logistic Regression TFIDF
![logreg_cm](https://media.git.generalassemb.ly/user/43213/files/6d192e23-50de-4aac-a968-5fdcc4fe6cb3)


* Confusion Matrix: AdaBoost TFIDF
![tfidf_cm](https://media.git.generalassemb.ly/user/43213/files/78c0311b-694d-42e1-833d-b4fc03e1b3d1)


Based on the metrics listed, it appears that Logistic Regression with Tf-IDF is the best amongst others despite falling behind AdaBoost Classifier. It has a precision score 0f 0.976 with execution time of 38 minutes. Whereas for AdaBoost Classifier, it has a precision score of 0.984 with executing time of 2 hours 15 mins which is about 4-5 times longer! The execution time is the deal breaker for me when deciding the better model. Hence, I choose Logistic Regression over AdaBoost for its efficiency with minimal difference in performance. 

The model can be deployed in Streamlit. The source file can be found [here](fakenews.py)

### Data Sources
We will use the University of Victoria’s ISOT Fake News Dataset. It contains more than 12,600 real news articles from Reuters.com and more than 12,600 fake news articles that were flagged by Politifact (a fact-checking organization in the United States). The dataset contains articles relating to a variety of topics, though mostly political news and world news. More information on the dataset can be found here[here](ISOT_Fake_News_Dataset_ReadMe)

[Fake News Dataset](./News _dataset/Fake.csv)
[Real News Dataset](./News _dataset/Real.csv)

### Data Dictionary
There are 4 columns in each of the dataset.


|Feature|Data Type|Description|
|---|---|---|
|title|String|Source of data|
|text|String|Source of data|
|subject|String|Source of data|
|date|datetime|Source of data|


### Conclusion & Recommendations
Conclusion:

- Hardcopy newspapers that were earlier preferred are now being substituted with social media and  Internet. 
- To detect fake news manually by cross-checking multiple sources can be daunting, time consuming and may cause further confusion.
- With a fake news detection system, it speeds up the process of determining whether a piece of news is fake or real. However, it does not stop there. 
Government are now taking measures ensuring its citizens consume legitimate news. (POFMA, Education)
- Even though there are other models that performed better, I stick with Logistic Regression simply due to its efficiency.
- Collecting the data once isn’t going to cut it given how quickly information spreads in today’s connected world and the number of articles being churned out.


Recommendation:
- This system is suitable for use by members of public as this group of people consume news
- This system is also suitable for journalist as they are mainly responsible for disemminating news

Future Improvements
- Use Deep Learning such as Word2Vec and LSTM 
- Tweak parameters
- Add more data

Future Works
- Analyse sentiments of fake news
- Deteck fake news through video and images
- Predict using propagation of fake news
