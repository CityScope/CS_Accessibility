#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:36:09 2020

@author: doorleyr
"""

import pandas as pd
import pyproj
import json
import numpy as np

def approx_shape_centroid(geometry):
    if geometry['type']=='Polygon':
        centroid=list(np.mean(geometry['coordinates'][0], axis=0))
        return centroid
    elif geometry['type']=='MultiPolygon':
        centroid=list(np.mean(geometry['coordinates'][0][0], axis=0))
        return centroid
    else:
        print('Unknown geometry type')

# change to aalto/abm directory
od=pd.read_csv('spatial_data/T06_tma_e_TOL2008_point/od.csv')



od['h_lon'], od['h_lat']=pyproj.transform(pyproj.Proj(init='epsg:3047'),
                              pyproj.Proj(init='epsg:4326')
                              ,list(od['ax']),list(od['ay']))

od['w_lon'], od['w_lat']=pyproj.transform(pyproj.Proj(init='epsg:3047'),
                              pyproj.Proj(init='epsg:4326')
                              ,list(od['tx']),list(od['ty']))

axyind_to_lon_lat={}
for od_ind, od_row in od.iterrows():
    if od_row['axyind'] not in axyind_to_lon_lat:
        axyind_to_lon_lat[od_row['axyind']]=[od_row['h_lon'], od_row['h_lat']]

od_by_home=od.groupby('axyind')['yht'].sum()

parking=json.load(open('spatial_data/parking.geojson'))
    
point_capacity_output={'type': 'FeatureCollection',
                       'features':[]}
point_capacity_features=[]

for p in parking['features']:
    new_feature=p.copy()       
    p['properties']={'parking':p['properties']['Capacity'] }
    p['properties']['centroid']=approx_shape_centroid(p['geometry'])
    point_capacity_features.append(p)
    
for axyind, axy_row in od_by_home.iteritems():
    homes=axy_row
    centroid=axyind_to_lon_lat[axyind]
    geometry={'coordinates': centroid, 'type': 'Point'}
    properties={'centroid': centroid, 'housing': homes}
    new_feature={"type": "Feature",
                 "geometry": geometry,
                 "properties": properties}
    point_capacity_features.append(new_feature)

# change directory to CS_Accessibility 
    
point_capacity_output['features']=  point_capacity_features
  
json.dump(point_capacity_output, open('python/Aalto/data/model_area.geojson', 'w'))
    
    