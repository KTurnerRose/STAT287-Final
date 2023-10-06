# -*- coding: utf-8 -*-
"""
STAT 287 Final Project Exploratory Data Analysis
Team Red Piranha
"""

########################################################################
#########                      IMPORTS                      ############
########################################################################

import csv
import pandas as pd
import os
import pickle
from collections import Counter
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from scipy.stats.kde import gaussian_kde
from scipy import stats

# Import objects from other files
from FP_RedPiranha_filter_data import dmv_zca_merge_clean as dmv_merge_clean

########################################################################
#########               FUNCTION DECLARATIONS               ############
########################################################################

### Copied from filter_data ###

def load_records(filepath):
    """ 
    
    Loads data from external file 
    
    :param filepath: filepath
    :return : pd.DataFrame
    
    """
    data = pd.read_csv(filepath, header = "infer")
    data = data.iloc[:, 1:]
    return data

def vermont_or_bust(data):
    """
    Removes all non-Vermont entries.
   :param data : pd.DataFrame
   :return: pd.DataFrame
    """
    df = data.copy()
    df = df.loc[df['DMV_State'] == 'VT']
    return df

def fix_zips(data):
    """
    Clean zipcode errors so they match the census database.
    :param data : pd.DataFrame
    :return: pd.DataFrame
    """
    df = data.copy()
    df['DMV_Zip'] = df['DMV_Zip'].str.extract('(\d{,4})')
    df['DMV_Zip'] = '0' + df['DMV_Zip']
    return df

def find_replace_missings(data: pd.DataFrame):
    """
    Replaces NaN datatype with placeholder value -999 for ease of 
    counting and replacing missing values.
    :param data : pd.DataFrame
    :return : pd.DataFrame
    """
    
    df = data.copy()
    no_nan_data = df.replace(to_replace= [np.NaN], value= [-999])
    return no_nan_data

def get_missings(data: pd.DataFrame):                                             
    """                                                                          
    Pass in the raw uploaded pandas dataframe and appends a column       
    vector containing the number of missing values per instance.                 
    :param data: pd.DataFrame                                                     
    :return: pd.DataFrame                                                          
    """                                                                                                                                 ##  |
    df = data.copy()
    missing_features = np.zeros((data.shape[0], 1))
    list_of_counters = []
    for index, row in enumerate(data.itertuples()):
        counter = Counter(row)
        if -999 not in counter.keys():
            missing_features[index] = 0
        else:
            missing_features[index] = counter[-999]
        list_of_counters.append(counter)
    df["MISSING COUNTS"] = missing_features
    return df

### Start EDA functions here ###

def dmv_single_var_stats(data, varlist):
    """
    Generates histograms of the distribution of each variable passed in a list 
    and generates/prints a dataframe of summary stats for all the variables in 
    the list.
    :param data: pd.DataFrame, list
    :return: pd.DataFrame       
    """
    df = data.copy()
    df = df[0:10000]
    for var in varlist:
        df = df.loc[df[var]!=-999]
        df.hist(var, bins=20)
    summary_stats = df.describe()
    print(summary_stats[varlist])
    return summary_stats

def dmv_multi_var_means (data, var1, var2):
    """
    Generates/prints a table of mean values for var2 grouped by var1 values.
    
    NOTE: Var2 canNOT be a categorical variable.
    :param data: pd.DataFrame, list
    :return: pd.DataFrame      
    """
    df = data.copy()
    df = df[0:10000]
    df = df.loc[df[var1]!=-999]
    df = df.loc[df[var2]!=-999]
    multi_var_means = df[[var1, var2]].groupby(var1).mean()
    print(multi_var_means)
    return(multi_var_means)

def vals_distr(data, var, xrotation):
    """
    Generates histograms of value distributions for categorical variables.
    :param data: pd.DataFrame, str, int
    :return: none
    """
    df = data.copy()
    df2 = df.replace(to_replace = -999, value = "N/A")
    counter = Counter(df2[var])
    var_counts = dict(counter)
    plt.bar(var_counts.keys(), var_counts.values())
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Values distribution for ' + var)
    ax = plt.gca()
    #make x axis labels more visible
    if len(ax.get_xticks()) > 200:
        ax.set_xticks(ax.get_xticks()[::5])
    elif len(ax.get_xticks()) > 1000:
        ax.set_xticks(ax.get_xticks()[::500])
    #styling x axis labels
    plt.xticks(rotation=xrotation)
    filename = 'figures/' + var + "_vals_hist.png"
    plt.savefig(filename)
    plt.show()
    
def vals_distr_top_x_keys(data, var, xrotation, xlabel, x):
    """
    Generates histograms of value distributions for categorical variables.
    :param data: pd.DataFrame, str, int
    :return: none
    """
    df = data.copy()
    df2 = df.replace(to_replace = -999, value = "N/A")
    counter = Counter(df2[var])
    keys = get_top_x_key_values(df2, var, x)
    values = []
    for key in keys:
        values.append(counter[key])
    plt.tight_layout()
    plt.bar(keys, values)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title('Values distribution for ' + var)
    plt.xticks(rotation=xrotation)
    filename = 'figures/' + var + "_vals_hist.png"
    plt.savefig(filename)
    plt.show()
    
def get_top_x_key_values(data, var, x):
    counter = Counter(data[var])
    keys = list(counter.keys())
    values = []
    for key in keys:
        values.append(counter[key])
    values = np.array(values)
    indices = np.flip(values.argsort())
    top_x_keys = []
    for i in indices[:x]:
        top_x_keys.append(keys[i])
    return top_x_keys
    
    
def cull_the_herd(data, var, thresh):
    df = data.copy()
    counter = Counter(df[var])
    above_thresh = []
    for key in counter.keys():
        if counter[key]>thresh:
            above_thresh.append(key)
    for index, value in df[var].items():
        if value not in above_thresh:
            df2 = df.drop(df.iloc[index])
    return df2
        

def correlation_heatmap (data):
    """
    Generates heatmap showing correlations between values of categorical variables.
    Returns correlation matrix of correlation scores.
    :param data: pd.DataFrame, list 
    :return: pd.DataFrame
    """
    df = data.copy()
    df = df
    df2 = pd.get_dummies(df)
    fig = plt.figure()
    df_hmap = fig.add_subplot()
    sns.heatmap(df2.corr(), cmap = 'Blues', ax = df_hmap)
    df_hmap.set_title('Correlation of fuel types and electric vehicle labels')
    filename = 'figures/fuel_elec_hmap'
    plt.savefig(filename)
    plt.show()
    return df2.corr()

def linear_func(x, slope, intercept):
    """
    Generates y values of a linear regression.
    """
    return slope * x + intercept


########################################################################
#########                    MAIN CODE                      ############
########################################################################


# load data
dmv2020_raw = load_records('data/DMV_2020_data.csv')
varnames = dmv2020_raw.columns
print(varnames)

# Cleans data (missingness)
no_nan_dmv2020 = find_replace_missings(dmv2020_raw)
dmv2020_misscol = get_missings(no_nan_dmv2020)

# Fixes zip codes to include leading zeros and classified as a string
dmv2020_stringzips = dmv2020_misscol.astype({'DMV_Zip': str}, copy = False)

dmv2020_vt = vermont_or_bust(dmv2020_stringzips)
dmv2020 = fix_zips(dmv2020_vt)

### EDA Begins here ###

### Plots histograms for all values of categorical variables with small amount of categories ###
vals_distr(dmv2020, "DMV_Vehicle_Type", 0)
vals_distr(dmv2020, "DMV_Fuel", 0)
vals_distr(dmv2020, 'EV_label_TRC', 0)
vals_distr(dmv2020, 'EPA_VehicleClass', 90)
vals_distr(dmv2020_stringzips, "DMV_State", 45)

### Plots histograms for values with counts in the top x of all categories (categorical variables) ###
vals_distr_top_x_keys(dmv2020, "DMV_Model", 45, 'Vehicle Model', 20)
vals_distr_top_x_keys(dmv2020, "DMV_Zip", 90, 'Zip Code', 30)
vals_distr_top_x_keys(dmv2020, "DMV_Make", 90, 'Vehicle Make', 30)

### Generates summary statistics for non-categorical variables
dmv_single_var_stats(dmv2020, ['MPGeComb_AllFuels_TRC', 'MPGeHwy_AllFuels_TRC', 'MPGeCity_AllFuels_TRC'])
test_means = dmv_multi_var_means(dmv2020, 'DMV_Zip', 'MPGeComb_AllFuels_TRC')
print('Burlington mean MPG: ', test_means.loc['05408'])

### Generates correlation heatmap for fuel vs EV label ###
dmv_heatmap_slice = dmv2020[['DMV_Fuel', 'EV_label_TRC']]
dmv_heatmap_test = correlation_heatmap(dmv_heatmap_slice)
print(dmv_heatmap_test)

############ EVEN MORE FIGURES using census data ############


dmv_merge_clean = dmv_merge_clean.replace("-", np.NaN)
data_clean = dmv_merge_clean.dropna()

# Clean
data_clean['MEAN_COMM_TIME'] = data_clean['MEAN_COMM_TIME'].astype(dtype = "float64")
data_clean['MED_INCOME'] = data_clean['MED_INCOME'].astype(dtype = "float64")
data_clean['HH_SIZE'] = data_clean['HH_SIZE'].astype(dtype = "float64")

### Generate Med Income to Fuel Efficiency plots

# Get Linear Regression relating median income to fuel efficiency
filtered_data = data_clean.filter(['ZIP_CODE','MED_INCOME', 'MPGeComb_AllFuels_TRC']).groupby('ZIP_CODE').mean()
slope, intercept, r, p, std_err = stats.linregress(filtered_data['MED_INCOME'], filtered_data['MPGeComb_AllFuels_TRC'])
lin_regress_values = []
for x in filtered_data['MED_INCOME']:
    lin_regress_values.append(linear_func(x, slope, intercept))

# Generate and save scatter Plot
plt.scatter(filtered_data['MED_INCOME'], filtered_data['MPGeComb_AllFuels_TRC'], color = 'mediumpurple')
plt.plot(filtered_data['MED_INCOME'], lin_regress_values, ls = '--', color = 'r')
plt.text(110000,21,'R = {}'.format(round(r,2)))
plt.ylabel('Average Fuel Efficiency (MPG)')
plt.xlabel('Median Income')
plt.title('Average Fuel Efficiency by Median Income of VT Zip Code')
plt.savefig('figures/fe_by_med_inc.png')
plt.show()

### Scatter plot: Fuel Efficiency by zip code population

# Get Linear Regression relating median income to fuel efficiency
filtered_data = data_clean.filter(['ZIP_CODE','TOTAL_POP', 'MPGeComb_AllFuels_TRC']).groupby('ZIP_CODE').mean()
slope, intercept, r, p, std_err = stats.linregress(filtered_data['TOTAL_POP'], filtered_data['MPGeComb_AllFuels_TRC'])
lin_regress_values = []
for x in filtered_data['TOTAL_POP']:
    lin_regress_values.append(linear_func(x, slope, intercept))

# Generate and save plot
plt.scatter(filtered_data['TOTAL_POP'], filtered_data['MPGeComb_AllFuels_TRC'], color = 'mediumpurple')
plt.plot(filtered_data['TOTAL_POP'], lin_regress_values, ls = '--', color = 'red')
plt.text(22500,21,'R = {}'.format(round(r,2)))
plt.ylabel('Average Fuel Efficiency (MPG)')
plt.xlabel('Population')
plt.title('Average Fuel Efficiency by Population of VT Zip Code')
plt.savefig('figures/fe_by_pop.png')
plt.show()

### Most popular vehicles by rural/urban divides
# Filter data into three separate data frames indicating 3 population ranges
filtered_data = data_clean.filter(['ZIP_CODE', 'TOTAL_POP', 'DMV_Make'])
filtered_data_low_pop = filtered_data[filtered_data['TOTAL_POP'] < 2500]
filtered_data_med_pop = filtered_data[filtered_data['TOTAL_POP'] >= 2500]
filtered_data_med_pop = filtered_data_med_pop[filtered_data_med_pop['TOTAL_POP'] < 10000]
filtered_data_high_pop = filtered_data[filtered_data['TOTAL_POP'] >= 10000]

## Make Side-by-Side Bar Plot

# Get top 10 makes overall
keys = get_top_x_key_values(filtered_data, 'DMV_Make', 10)

# Create counter objects for the three resulting data frames
make_counters = [Counter(filtered_data_low_pop['DMV_Make']),
                 Counter(filtered_data_med_pop['DMV_Make']),
                 Counter(filtered_data_high_pop['DMV_Make'])]

# Get counts of the top 10 overall makes when filtered by population ranges
make_counts = []
for make_counter in make_counters:
    counts = []
    for key in keys:
        counts.append(make_counter[key])
    make_counts.append(counts)

# Generate and save the side-by-side barplot

# Format bars in barplot
width = 0.25
fig, ax = plt.subplots()
rects_low = ax.bar(np.arange(len(keys)) - 1.5 * width, make_counts[0], width, label = 'less than 2500')
rects_med = ax.bar(np.arange(len(keys)) - 0.5 * width, make_counts[1], width, label = 'between 2500 and 10000')
rects_high = ax.bar(np.arange(len(keys)) + 0.5 * width, make_counts[2], width, label = 'greater than 10000')

# Formats figure labels and saves figure.
ax.set_ylabel('Counts')
ax.set_xlabel('Vehicle Make')
ax.set_xticks(np.arange(len(keys)))
ax.set_xticklabels(keys)
ax.legend()
plt.title('Top 10 Vehicle Makes by Zip Code Population Ranges')
plt.savefig('figures/top_10_makes_by_pop.png')
plt.show()

### Generalize the EPA_VehicleClass field into fewer categories
conditions = [
(data_clean['EPA_VehicleClass'].str.contains("Cars")),
(data_clean['EPA_VehicleClass'].str.contains("Minivan")),
(data_clean['EPA_VehicleClass'].str.contains("Station")),
(data_clean['EPA_VehicleClass'].str.contains("Utility")),
(data_clean['EPA_VehicleClass'].str.contains("Pickup"))
]
vehicle_class_values = ["Car","Minivan","SUV","Station Wagon","Truck"]
data_clean['EPA_VehicleClass_bins'] = np.select(conditions, vehicle_class_values, default = "Other")

# Get PMF of EPA Vehicle Class
counter_EPA = Counter(data_clean['EPA_VehicleClass_bins'])
EPA_proportions = []
for vehicle_class_count in counter_EPA.values():
    EPA_proportions.append(vehicle_class_count / sum(counter_EPA.values()))
pmf = pd.DataFrame({'Vehicle_Class':counter_EPA.keys(), 'Probability':EPA_proportions})

# Generate and save bar plot
plt.bar('Vehicle_Class', 'Probability', data = pmf.sort_values('Probability', ascending = False), color = 'mediumpurple')
plt.xlabel('Vehicle Class')
plt.ylabel('Probability')
plt.title('PMF of Vehicle Class in Vermont')
plt.savefig('figures/vehicleclasspmf.png')
plt.show()

#Get PMF of EV_Label
counter_EV = Counter(data_clean['EV_label_TRC'])
EV_proportions = []
for EV_label_count in counter_EV.values():
    EV_proportions.append(EV_label_count / sum(counter_EV.values()))    
pmf = pd.DataFrame({'EV_label':counter_EV.keys(), 'Probability':EV_proportions})

# Generate and save bar plot
plt.bar('EV_label', 'Probability', data = pmf.sort_values('Probability', ascending = False), color = 'mediumpurple')
plt.xlabel('EV_label')
plt.ylabel('Probability')
plt.title('PMF of EV Labels in Vermont')
plt.savefig('figures/EVlabelspmf.png')
plt.show()

