# DSGA1004 - BIG DATA
## Final project-Check Point Submission
**Group Number:** 101


**Group Member**

- Cooper Zhao(jz5246)
- Vanessa Xu(zx657)
- Catherine Zheng(rz2432)


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
4. Run the best ALS Model file to generate user and item vector parquet file

```bash
spark-submit annoy_als_small.py
spark-submit annoy_als_full.py
```
 
5.put the data from hsdf to local machine 

```bash
hfs -get hdfs:/user/{netID}/user_vec_small.parquet
hfs -get hdfs:/user/{netID}/item_vec_small.parquet

hfs -get hdfs:/user/{netID}/user_vec_full.parquet
hfs -get hdfs:/user/{netID}/item_vec_full.parquet
```

6. Run small_Fast_Search_vs_Brute_Search.ipynb
7. Run full_Fast_Search_vs_Brute_Search.ipynb
