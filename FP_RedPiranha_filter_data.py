# -*- coding: utf-8 -*-
"""
STAT 287 Final Project Data Clean-up
Team Red Piranha

"""

########################################################################
#########                      IMPORTS                      ############
########################################################################

import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from string import punctuation
from string import ascii_letters

########################################################################
#########               FUNCTION DECLARATIONS               ############
########################################################################

### LOAD AND SAVE FUNCTIONS ###

def load_records(filepath, dtype):
    """

    Loads data from external file

    :param filepath: filepath
    :return : pd.DataFrame

    """
    data = pd.read_csv(filepath, dtype = dtype, header = "infer")
    return data

def save_data(dataframe, filename):
    """

    Saves pandas DataFrame to csv file

    :param dataframe: pd.DataFrame
    :param filename: Name of output csv

    """
    dataframe.to_csv(filename, header = True)
    output = print("Succesfully Saved!")

    return output



### MISSINGNESS FUNCTIONS ###

def get_missings(data: pd.DataFrame):
    """
    Pass in a pandas dataframe and appends a column containing the number of missing values 
    per instance.
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """                                                                                                                                 ##  |
    df = data.copy()
    df['MISSING_COUNT'] = df.isnull().sum(axis = 1)
    return df

def missings_distributions(data: pd.DataFrame):
    """
    Calculates distributions of missing values across variables of a DataFrame
    and returns them as a dictionary. Also generates histogram plot of
    distributions.

    Parameters
    ----------
    :param data : pd.DataFrame
    :return : dictionary (variable name: proportion missingness)

    """
    df = data.copy()
    missing_counts = dict(df.isnull().sum())
    for key, value in missing_counts.items():
        missing_counts[key] = value/len(df)

    plt.bar(missing_counts.keys(), missing_counts.values(), color='mediumpurple')
    plt.xlabel('Variable')
    plt.xticks(rotation = 'vertical')
    plt.ylabel('Proportion of missing values')
    plt.title('Missingness proportion by variable')
    plt.show()

    return missing_counts

### FUNCTIONS FOR CHECKING FOR VALUE ERRORS ###

def get_val_distr(data, var):
    df = data.copy()
    for column_name, contents in df.iteritems():
        counter = Counter(contents)
    return counter

def get_unique_vals(data: pd.DataFrame):
    df = data.copy()
    unique_vals_by_var = {}
    for column_name, contents in df.iteritems():
        unique_vals_by_var[column_name] = list(df[column_name].unique())
    return unique_vals_by_var

def check_var_errors(valslist, vartype, valids, questionchars):
    invalid_vals = []
    valid_vals = []
    questionable_vals = []
    for i in valslist:
        if i in valids:
            valid_vals.append(i)
        if type(i) != vartype:
            invalid_vals.append(i)
        elif valids != [] and i not in valids:
            invalid_vals.append(i)
        else:
            if i not in invalid_vals and i not in valid_vals:
                questionable_vals.append(i)
                # for char in str(i):
                #     if char in questionchars:
                #         questionable_vals.append(i)
    return valid_vals, invalid_vals, questionable_vals

### FUNCTIONS TO CLEAN AND PRE-PROCESS DATA
def vermont_or_bust(data):
    """
    Removes all non-Vermont entries.

   :param data : pd.DataFrame
   :return: pd.DataFrame

    """
    df = data.copy()
    df = df.loc[df['DMV_State'] == 'VT']
    return df



########################################################################
#########              DATA/PARAMETERS INFO                 ############
########################################################################

punct = set(punctuation)
letters = set(ascii_letters)

attributes = {'DMV_Make': {'Type': str,
                           'Flag characters': punct,
                           'Validated values':[]
                           },
              'DMV_Model': {'Type': str,
                            'Flag characters': punct,
                            'Validated values': []
                            },
              'DMV_Vehicle_Type': {'Type': str,
                                   'Flag characters': punct,
                                   'Validated values': ['A', 'B']
                                   },
              'DMV_Fuel': {'Type': str,
                           'Flag characters': punct,
                           'Validated values':['B', 'G', 'D', 'E', 'O', 'P'],
                           },
              'DMV_City': {'Type': str,
                           'Flag characters': punct.discard('.'),
                           'Validated values': []
                           },
              'DMV_State': {'Type': str,
                            'Flag characters': [],
                            'Validated values': ['AK', 'AL', 'AR', 'AZ', 'CA',
                                                 'CO', 'CT', 'DC', 'DE', 'FL',
                                                 'GA','HI', 'IA', 'ID', 'IL',
                                                 'IN', 'KS', 'KY', 'LA', 'MA',
                                                 'MD', 'ME', 'MI', 'MN', 'MO',
                                                 'MS', 'MT', 'NC', 'ND', 'NE',
                                                 'NH', 'NJ', 'NM', 'NV', 'NY',
                                                 'OH', 'OK', 'OR', 'PA', 'RI',
                                                 'SC', 'SD', 'TN', 'TX', 'UT',
                                                 'VA', 'VT', 'WA', 'WI', 'WV',
                                                 'WY']
                            },
              'DMV_Zip': {'Type': str,
                          'Flag characters': [letters, punct.discard('-')],
                          'Validated values':[]
                          },
              'EV_label_TRC': {'Type': str,
                               'Flag characters': punct,
                               'Validated values':['PHEV', 'BEV', 'HEV', 'ICE']
                               },
              'MPGeComb_AllFuels_TRC': {'Type': float,
                                        'Flag characters': letters,
                                        'Validated values': range(5, 55)
                                        },
              'MPGeCity_AllFuels_TRC': {'Type': float,
                                        'Flag characters': letters,
                                        'Validated values': range(5, 55)
                                        },
              'MPGeHwy_AllFuels_TRC': {'Type': float,
                                       'Flag characters': letters,
                                       'Validated values': range(5, 55),
                                       },
              'EPA_VehicleClass': {'Type': str,
                                   'Flag characters': [],
                                   'Validated values': []
                                   }
              }

########################################################################
#########                    MAIN CODE                      ############
########################################################################

""" ANALYSIS FOR MISSINGNESS AND ERRORS """
# load data
raw_dmv2020 = load_records('data/DMV_2020_data.csv', {'DMV_Zip':'str'})
varnames = raw_dmv2020.columns

# append a column vector of missing counts to the end of the dataframe
dmv2020 = get_missings(raw_dmv2020)

# get a list of the distributions of missing values across all variables
dmv2020_proportion_missing_by_variable = missings_distributions(dmv2020)

# get dictionary of unique values for each variable
dmv2020_vals_by_var = get_unique_vals(dmv2020)

# convert zip codes to strings before running values checks


# pulls list of unique values for each variable to analyze for invalid values
for i in varnames:
    try:
        attributes[i]['Unique values'] = dmv2020_vals_by_var[i]
        attributes[i]['Valid values'], attributes[i]['Invalid values'], attributes[i]['Suspect values'] = check_var_errors(attributes[i]['Unique values'], attributes[i]['Type'], attributes[i]['Validated values'], attributes[i]['Flag characters'])
    except:
        print(i)

# You can now print information on the valid, invalid, and suspect values,
# which you can use to clean data below

print(attributes['DMV_Zip']['Invalid values'])



"""CLEANING AND PREPROCESSING DATA"""

# STEP ONE: Fix state information and filter for VT addresses only #

#any entry for DMV_State that is not NaN or a state code has been found
#to be a VT Town --> change any non 2-character entry to "VT"

for state in dmv2020['DMV_State']:
    try:
        if len(state) > 2:
            state = "VT"
    except:
        continue

#filter for VT only addresses and cleanup zip codes to be valid 5-digit codes
dmv2020_vt = vermont_or_bust(dmv2020)



#check missingness on VT only records
post_zip_MD = missings_distributions(dmv2020_vt)

#save VT records
save_data(dmv2020_vt, "data/dmv2020_vt.csv")


#STEP TWO: Remove rows with missing MPG fields
dmv2020_vt_mpgclean = dmv2020_vt.dropna()

#identify most common vehicle makes and models in Vermont
print(get_val_distr(dmv2020_vt_mpgclean, 'DMV_Make').most_common(10))
dmv2020_vt_mpgclean['Make_Model'] = dmv2020_vt_mpgclean['DMV_Make'] + dmv2020_vt_mpgclean['DMV_Model']
print(get_val_distr(dmv2020_vt_mpgclean,'Make_Model').most_common(20))
dmv2020_vt_mpgclean['Make_Engine'] = dmv2020_vt_mpgclean['DMV_Make'] + dmv2020_vt_mpgclean['EV_label_TRC']
print(get_val_distr(dmv2020_vt_mpgclean,'Make_Engine').most_common(20))
print(get_val_distr(dmv2020_vt_mpgclean, 'DMV_Fuel').most_common(5))

#save records with no missing values
save_data(dmv2020_vt_mpgclean, "data/dmv2020_vt_mpgclean.csv")


#MERGE RECORDS WITH CENSUS DATA

#change name of DMV_Zip to ZIP_CODE for merge
dmv2020_vt.columns = dmv2020_vt.columns.str.replace('DMV_Zip', 'ZIP_CODE')

#load census and zip code area data
census_income = load_records('data/income_zca_2019.csv', dtype = {"ZCTA": "str"})
census_pop = load_records('data/population_zca_2019.csv', dtype = {"ZCTA": "str"})
census_hhsize = load_records('data/hhsize_zca_2019.csv', dtype = {"ZCTA": "str"})
census_vehicles = load_records('data/vehicles_zca_2019.csv', dtype = {"ZCTA": "str"})
census_commute = load_records('data/commute_zca_2019.csv', dtype = {"ZCTA": "str"})

#remove "ZCTA5 " from ZCTA field
census_income[['ZCTA', 'ZIP_CODE']] = census_income['ZCTA'].str.split(" ", expand = True)
census_pop[['ZCTA', 'ZIP_CODE']] = census_pop['ZCTA'].str.split(" ", expand = True)
census_hhsize[['ZCTA', 'ZIP_CODE']] = census_hhsize['ZCTA'].str.split(" ", expand = True)
census_vehicles[['ZCTA', 'ZIP_CODE']] = census_vehicles['ZCTA'].str.split(" ", expand = True)
census_commute[['ZCTA', 'ZIP_CODE']] = census_commute['ZCTA'].str.split(" ", expand = True)


#calculate average number of vehicles per household
census_vehicles['AVG_NUM_VEH'] = 0

for index, row in census_vehicles.iterrows():
    if row['NO_VEH'] != 0 and row['1_VEH'] != 0 and row['2_VEH'] != 0 and row['3_VEH'] != 0 and row['4_VEH'] != 0:
        a = row['1_VEH'] + row['2_VEH']*2 + row['3_VEH']*3 + row['4_VEH']*4
        b = row['NO_VEH'] + row['2_VEH'] + row['3_VEH'] + row['4_VEH']
        avg = a/b
        census_vehicles.loc[index, ['AVG_NUM_VEH']] = avg
    else:
        census_vehicles.loc[index, ['AVG_NUM_VEH']] = 0


#merge census and dmv data on zip code
dmv_zca_merge = dmv2020_vt.merge(census_income, on = "ZIP_CODE", how = "left").merge(census_pop, on = "ZIP_CODE", how = "left").merge(census_hhsize, on ="ZIP_CODE", how = "left")
dmv_zca_merge = dmv_zca_merge.merge(census_vehicles, on = "ZIP_CODE", how = "left").merge(census_commute, on = "ZIP_CODE", how = "left")

dmv_zca_merge_clean = dmv_zca_merge[['ID', 'DMV_Make', 'DMV_Fuel', 'ZIP_CODE', 'EV_label_TRC', 'MPGeComb_AllFuels_TRC', 'EPA_VehicleClass', 'MED_INCOME', 'TOTAL_POP', 'HH_SIZE', 'AVG_NUM_VEH', 'MEAN_COMM_TIME']]

#save cleaned data
save_data(dmv_zca_merge_clean, "data/dmv_merge_clean.csv")
