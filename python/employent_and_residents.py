#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:50:01 2019

@author: doorleyr
"""

import json
import numpy as np
import pandas as pd


def approx_shape_centroid(geometry):
    if geometry['type']=='Polygon':
        centroid=list(np.mean(geometry['coordinates'][0], axis=0))
        return centroid
    elif geometry['type']=='MultiPolygon':
        centroid=list(np.mean(geometry['coordinates'][0][0], axis=0))
        return centroid
    else:
        print('Unknown geometry type')


# =============================================================================
# Hamburg
# =============================================================================
city='Hamburg'
stat_areas=json.load(open('python/Hamburg/data/stat_areas.geojson'))
for f in stat_areas['features']:
    f['properties']['centroid']=approx_shape_centroid(f['geometry'])
    f['properties']['housing']=int(f['properties']['BevGes'])
json.dump(stat_areas, open('python/'+city+'/data/model_area.geojson', 'w'))

# =============================================================================
# Detroit
# =============================================================================

city='Detroit'

ZONES_PATH='python/'+city+'/data/model_area.geojson'
OD_PATH='python/'+city+'/data/LODES/mi_od_main_JT00_2017.csv.gz'

od=pd.read_csv(OD_PATH)
zones = json.load(open(ZONES_PATH))

geoid_order=[f['properties']['GEO_ID'].split('US')[1] for f in zones['features']]
for i, f in enumerate(zones['features']):
    centroid=approx_shape_centroid(f['geometry'])
    f['properties']['centroid']=centroid

block_groups_df=pd.DataFrame(index=geoid_order, columns=['residents', 'workers', 'x', 'y'])
od['h_bg']=od.apply(lambda row: str(row['h_geocode'])[:12], axis=1)
od['w_bg']=od.apply(lambda row: str(row['w_geocode'])[:12], axis=1)
od.head()
workers_by_resi_bg=od.groupby('h_bg')['S000'].agg('sum')
workers_by_pow_bg=od.groupby('w_bg')['S000'].agg('sum')

for f in zones['features']:
    geoid=f['properties']['GEO_ID'].split('US')[1]
    if geoid in workers_by_pow_bg.index:
        f['properties']['employment']=int(workers_by_pow_bg.loc[geoid])
    else:
        f['properties']['employment']=0
        print(str(geoid)+' has no employees')
    if geoid in workers_by_resi_bg.index:    
        f['properties']['housing']=int(workers_by_resi_bg.loc[geoid])
    else:
        f['properties']['housing']=0
        print(str(geoid)+' has no residents')

json.dump(zones, open(ZONES_PATH, 'w'))