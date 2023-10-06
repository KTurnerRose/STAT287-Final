# STAT/CS287-FP
STAT 287 Final Project with DMV and Census Data

_____________________________________
**The Team: Red Piranha**

Yoshi Bird
Ted Hadley
Erica Quallen
Kelly Turner
_____________________________________

**Directories**

data/
* ZipCodeSHP/
    * zip code shapefile (VT_ZIP_Code_Areas.shp) and dependent files (.cpg, .dbf, .prj, .sbn, .sbx, .shx, .xml)
* original data (DMV_2020_Data.csv) 
* cleaned VT data (dmv2020_vt.csv)
* fully cleaned VT and MPG data (dmv2020_vt_mpgclean.csv)
* dmv data merged with census data (dmv_merge_clean.csv)
* dmv merged data with EPA_VehicleClass binned (data_clean_withBins.csv)
* census data
      - mean commute time (commute_zca_2019.csv)
      - mean household size (hhsize_zca_2019.csv)
      - median household income (income_zca_2019.csv)
      - zip code area population (population_zca_2019.csv)
      - number of vehicles per household (vehicles_zca_2019.csv)


minutes/
* includes all meeting minutes throughout project
* minutes_REDPIRANHA_2021-12-06
* minutes_REDPIRANHA_2021-12-08
* minutes_REDPIRANHA_2021-12-10
* minutes_REDPIRANHA_2021-12-13
* minutes_REDPIRANHA_2021-12-15

figures/
* numerous figures developed during EDA and mapping

_____________________________________

**Files**

notes.txt
* notes and ideas produced along the way (pseudo-logbook)

FP_RedPiranha_filter_data.py
* Loads data
* Filters out data with missing values
* Stores cleaned data as new file
* Combines cleaned data with census data

FP_RedPiranha_EDA.py
* Exploratory Data Analysis
* Visualizations
      * scatter plots
      * histograms
      * barplots
      * variable distributions

FP_RedPiranha_model.py
* Model development
* Linear Regression Model
* Multinomial Logit Models
* Mapping with shapefile
