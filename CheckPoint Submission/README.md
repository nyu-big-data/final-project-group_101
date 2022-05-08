# DSGA1004 - BIG DATA
## Final project-Check Point Submission
**Group Number:** 101


**Group Member**

- Cooper Zhao(jz5246)
- Vanessa Xu(zx657)
- Catherine Zheng(rz2432)


## Partition Data
We splited the `ratings.csv` into training, validation and test dataset.
We started by partition the user identities with ratio 0.2:0.4:0.4 into train, val, and test. 
Then we splited the user's interactions in val and test dataset based on each user's median timestamp.
The old half of user's interactionsthe in val and test data goes into training set. Then the ratio for train, val, and test becomes 0.6:0.2:0.2.

### Example:
This is a interactions table for 10 users, the value instead of being ratings for each movie will be the the timestamp they rating on this movie.(1: oldest, 4:newest)


`ratings.csv`
|         | user_1 | user_2 | user_3 | user_4 | user_5 | user_6 | user_7 | user_8 | user_9 | user_10 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| movie_1 | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1       |
| movie_2 | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2       |
| movie_3 | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3       |
| movie_4 | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4       |


**Training Set**
|         | user_1 | user_2 | user_3 | user_4 | user_5 | user_6 | user_7 | user_8 | user_9 | user_10 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| movie_1 | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1       |
| movie_2 | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2       |
| movie_3 | 3      | 3      |        |        |        |        |        |        |        |         |
| movie_4 | 4      | 4      |        |        |        |        |        |        |        |         |


**Validation Set**
|         | user_3 | user_4 | user_5 | user_6 |
|---------|--------|--------|--------|--------|
| movie_3 | 3      | 3      | 3      | 3      |
| movie_4 | 4      | 4      | 4      | 4      |


**Test Set**
|         | user_7 | user_8 | user_9 | user_10 |
|---------|--------|--------|--------|---------|
| movie_3 | 3      | 3      | 3      | 3       |
| movie_4 | 4      | 4      | 4      | 4       |


## Model Fitting & Evaluation
### Popularity Baseline Model
We applied popularity baseline model to both `ml-latest-small/ratings.csv` and `ml-latest/ratings.csv` with damping value = 101, and predicted the top 100 movies with highest average ratings score in training set.
And we used MAP@100 with package `pyspark.ml.evaluation.RankingEvaluator` to evaluate the model performence.


|            | ratings_small       | ratings_full         |
|------------|---------------------|----------------------|
| validation | 0.04200661766182382 | 0.025203392149632926 |
| test       | 0.11034154976775325 | 0.048323563592365296 |

### Alternating Least Squares Model
#### `model.transform(test)` 


#### `model.recommendForUserSubset(test, 100)`
Spark's alternating least squares (ALS) method to learn latent factor representations and apply to `ml-latest-small/ratings.csv` with hyperparameter `rank = 100`, `maxIter=10`, `regParam=0.1`, and `alpha = 10`. 
We use the function `model.recommendForUserSubset()` to get the top 100 personalized recommendations. 
And we used MAP@100 with package `pyspark.ml.evaluation.RankingEvaluator` to evaluate the model performence.

|            | ratings_small        | ratings_full          |
|------------|----------------------|-----------------------|
| validation | 0.007910911442319646 | 0.0003105726230967176 |
| test       | 0.12172311773663758  | 0.00460049048690818   |




## The data set

This project used the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The data is hosted in NYU's HPC environment under `/scratch/work/courses/DSGA1004-2021/movielens`.

Two versions of the dataset are provided: a small sample (`ml-latest-small`, 9000 movies and 600 users) and a larger sample (`ml-latest`, 58000 movies and 280000 users).
Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which you may consider as additional features beyond
the collaborative filter.
Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.


## Compile

1. Connect to Peel
```bash
ssh <NetId>@peel.hpc.nyu.edu
```


2. Git clone the repository
```bash
git clone git@github.com:nyu-big-data/final-project-group_101.git
cd CheckPoint Submission
```


3. To run Spark jobs on the Peel cluster,  run the following command:
```bash
source shell_setup.sh
```


4. Open and run the `PartitionData.ipynb` to partition the data.



5. Load data onto HDFS
```bash
hfs -put Data/ratings_small_train.csv
hfs -put Data/ratings_small_val.csv
hfs -put Data/ratings_small_test.csv

hfs -put Data/ratings_full_train.csv
hfs -put Data/ratings_full_val.csv
hfs -put Data/ratings_full_test.csv
```


6. Call spark-submit to run the code


**Popularity Baseline Model ratings_small**
```bash
spark-submit baseline_small.py
```


**Popularity Baseline Model ratings_full**
```bash
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true baseline_full_fitting.py
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true baseline_full_predicting.py
```


**ALS Model on ratings_small**
```bash
spark-submit ALS_small.py
```


7. Once you submit the job to Peel, Spark will continuously output a log until completion. In this log, you will receive a tracking URL to check the progress.
```bash
tracking URL: http://horton.hpc.nyu.edu:8088/proxy/application_1613664569968_2108
```


8. To see the output of the scripts, run the following command:

```bash
yarn logs -applicationId <your_application_id> -log_files stdout
yarn logs -applicationId <your_application_id> -log_files stderr
```

**Note:** The Application ID is the last part of url, or in the example above, application_1613664569968_2108. Your Application ID will be different every time you spark-submita job. 
