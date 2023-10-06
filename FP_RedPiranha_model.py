# -*- coding: utf-8 -*-
"""
STAT 287 Final Project Main Code
Team Red Piranha

"""

########################################################################
#########                      IMPORTS                      ############
########################################################################

import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import geopandas as gpd


#imports from other project files
from FP_RedPiranha_filter_data import dmv_zca_merge_clean as dmv_merge_clean
from FP_RedPiranha_filter_data import save_data

    
########################################################################
#########                    MAIN CODE                      ############
########################################################################

dmv_merge_clean = dmv_merge_clean.replace("-", np.NaN)
data_clean = dmv_merge_clean.dropna()

data_clean['MEAN_COMM_TIME'] = data_clean['MEAN_COMM_TIME'].astype(dtype = "float64")
data_clean['MED_INCOME'] = data_clean['MED_INCOME'].astype(dtype = "float64")
data_clean['HH_SIZE'] = data_clean['HH_SIZE'].astype(dtype = "float64")



#LINEAR REGRESSION
#dependent variable - MPGe_Comb_AllFuels_TRC
#independent variables - MED_INCOME, HH_SIZE,MEAN_COMM_TIME

lm1 = smf.ols(formula = "MPGeComb_AllFuels_TRC ~ MED_INCOME + HH_SIZE + MEAN_COMM_TIME", data = data_clean).fit()
print(lm1.params)
print(lm1.summary())


#MULTINOMIAL LOGIT MODEL 1 
#dependent variable - EPA_VehicleClass
#independent variables - MED_INCOME, HH_SIZE, COMM_TIME

#assign bins to EPA_VehicleClass_bins based on EPA_VehicleClass
conditions = [
     (data_clean['EPA_VehicleClass'].str.contains("Cars")),
     (data_clean['EPA_VehicleClass'].str.contains("Minivan")),
     (data_clean['EPA_VehicleClass'].str.contains("Station")),
     (data_clean['EPA_VehicleClass'].str.contains("Utility")),
     (data_clean['EPA_VehicleClass'].str.contains("Pickup"))
     ]
values = ["Car","Minivan","SUV","Station Wagon","Truck"]

data_clean['EPA_VehicleClass_bins'] = np.select(conditions, values, default = "Other")

#build the model
x1 = data_clean[['MED_INCOME', 'HH_SIZE', 'MEAN_COMM_TIME']]
y1 = data_clean[['EPA_VehicleClass_bins']]

#test the model
logit_mod1 = sm.MNLogit(y1, sm.add_constant(x1)).fit()
print(logit_mod1.summary())



#MULTINOMIAL LOGIT MODEL 2
#dependent variable - EV_label_TRC
#independent variables - MED_INCOME, HH_SIZE, MEAN_COMM_TIME

#build the model
x2 = data_clean[['MED_INCOME', 'HH_SIZE', 'MEAN_COMM_TIME']]
y2 = data_clean[['EV_label_TRC']]

#test the model
logit_mod2 = sm.MNLogit(y2, sm.add_constant(x2)).fit()
print(logit_mod2.summary())



#save new file with binned VehicleClass
save_data(data_clean, "data/data_clean_withBins.csv")


#MAPPING VISUALIZATIONS

#load shapefile of VT zip code areas
zca_shp = gpd.read_file('data/ZipCodeSHP/VT_Zip_Code_Areas.shp')

zca_shp.columns = zca_shp.columns.str.replace('ZCTA', 'ZIP_CODE')

#merge dmv_merge_clean and zca_shp
zca_merge = dmv_merge_clean.merge(zca_shp, on = "ZIP_CODE", how = "left")
zca_merge = zca_merge.dropna()


#create subsets
zca_merge_make = zca_merge[['DMV_Make', 'ZIP_CODE', 'geometry']]
zca_merge_mpg = zca_merge[['ZIP_CODE', 'MPGeComb_AllFuels_TRC', 'geometry']]
zca_merge_ev = zca_merge[['ZIP_CODE', 'EV_label_TRC', 'geometry']]
zca_merge_vehclass = zca_merge[['ZIP_CODE', 'EPA_VehicleClass', 'geometry']]
zca_merge_income = zca_merge[['ZIP_CODE', 'MED_INCOME', 'geometry']]
zca_merge_commute = zca_merge[['ZIP_CODE', 'MEAN_COMM_TIME', 'geometry']]

#aggregate data
zca_mpg = zca_merge_mpg.groupby(by = ["ZIP_CODE"], dropna = True).mean()
zca_mpg = zca_mpg.merge(zca_shp, on = "ZIP_CODE", how = "left")

zca_make = zca_merge_make.groupby("ZIP_CODE").agg(lambda x:x.value_counts().index[0])
zca_make = zca_make.merge(zca_shp, on = "ZIP_CODE", how = "left")

zca_merge_income['MED_INCOME'] = zca_merge_income['MED_INCOME'].astype("float64")
zca_income = zca_merge_income.groupby("ZIP_CODE", dropna = True).mean()
zca_income = zca_income.merge(zca_shp, on = "ZIP_CODE", how = "left")

zca_merge_commute['MEAN_COMM_TIME'] = zca_merge_commute['MEAN_COMM_TIME'].astype(dtype = "float64")
zca_commute = zca_merge_commute.groupby("ZIP_CODE", dropna = True).mean()
zca_commute = zca_commute.merge(zca_shp, on = "ZIP_CODE", how = "left")

#convert to geodataframe
zca_mpg_geo= gpd.GeoDataFrame(zca_mpg)
zca_make_geo = gpd.GeoDataFrame(zca_make)
zca_income_geo = gpd.GeoDataFrame(zca_income)
zca_commute_geo = gpd.GeoDataFrame(zca_commute)

#VISUALIZATIONS
#fuel efficiency
fig, ax = plt.subplots(1,1)
zca_mpg_geo.plot(column = "MPGeComb_AllFuels_TRC", ax = ax, legend = True)
ax.axis('off')
ax.set_title("Average Fuel Efficiency (mpg) by Zip Code Area")
plt.savefig("figures/mpg_by_zca.png")

#most common make
fig, ax = plt.subplots(1,1)
zca_make_geo.plot(column = "DMV_Make", ax = ax, legend = True)
ax.axis('off')
ax.set_title("Most Common Vehicle Make by Zip Code Area")
leg = ax.get_legend()
leg.set_bbox_to_anchor((1.15,0.5))
plt.savefig("figures/make_by_zca.png")

#median income
fig, ax = plt.subplots(1,1)
zca_income_geo.plot(column = "MED_INCOME", ax = ax, legend = True)
ax.axis('off')
ax.set_title("Median Income by Zip Code Area")
plt.savefig("figures/income_by_zca.png")

#mean commute time
fig, ax = plt.subplots(1,1)
zca_commute_geo.plot(column = "MEAN_COMM_TIME", ax = ax, legend = True)
ax.axis('off')
ax.set_title("Average Commute Time (minutes) by Zip Code Area") 
plt.savefig("figures/commute_by_zca.png")