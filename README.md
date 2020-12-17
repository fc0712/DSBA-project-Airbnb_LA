# Github Repo for Data Science Project of Analysis of Airbnb


## Notebook
If the notebook for some reason is not loading on github the following links should take you to jupyter NBviewer. 

Data Cleaning / Pre-processing : [Link to nbviewer notebook](https://nbviewer.jupyter.org/github/fc0712/DSBA-project-Airbnb_LA/blob/master/Data%20Cleaning%20Notebook.ipynb)

Analysis: [Link to nbviewer notebook](https://nbviewer.jupyter.org/github/fc0712/DSBA-project-Airbnb_LA/blob/master/Analysis.ipynb)

##  Data
The table gives a overview of the data avalible in the data folder

| Data                      | Description                                                           |
|---------------------------|-----------------------------------------------------------------------|
| Violent_LA.html           | HTML file containing the tables of violent crime on L.A.Times         |
| airbnb_cleaned.csv        | Cleaned CSV file of Airbnb data - Generated in preprocessing notebook |
| key_new.jpg               | Key for Word cloud masking                                            |
| missing_neigh_scraped.csv | CSV containing scraped data from AreaVibes                            |
| sentiment_score_ML.csv    | CSV containing average sentiment scores of all listings.              |
| study.pkl                 | Pickle file containing Optuna study object                            |

## Conclusion
In order to answer the presented research questions extensive preprocessing of the Airbnb dataset took place, due to
the high number of missing values, which resulted in the main dataset being reduced from approximately 30,000 to
15,000 observations.

Natural Language Processing included sentiment analysis and topic modelling of Airbnb reviews data, which revealed a
predominance of positive reviews. A decomposition of the identified reviews sentiment indicated that positive and
negative reviews involved a lot of overlapping topics, among which place, location, bed and host were the most
frequent. Based on the implementation of NLP in the project, one particularly interesting factor identified was the fact
hosts appears to have a tremendous influence on renters satisfaction, and thus indicating that hosts need to be more
engaged compared to more traditional housing options such as hotels.

In order to predict the optimal listing price, different Machine Learning models were introduced. Based on a five fold
cross validation, the best performing Machine Learning model, and therefore the optimal model for the purpose of the
project, was found to be the Cat Boost model. Extensive hyper parameter tuning was performed on the Cat Boost model,
which improved model performance marginally and ensured robustness of the model.
A breakdown of the model performance revealed a 𝑅2 value of 74.19 and a RMSE of 54.99, thereby indicating that the model is able to explain
74.19% of the variation in the listing price. Having the highest Shap values, the attributes bedrooms, price category
neighborhood, and bathrooms had the most significant impact the model performance, which makes sense, considering
the attributes bedrooms and bathrooms being indicative of the size of the listing. 

It can thereby be concluded that it is possible to predict an approximate optimal listing price for new Airbnb listings. Further optimization of the models
performance could however be achieved by implementing a Deep Learning model on listing photos, to capture the
overall attractiveness of the listings.
