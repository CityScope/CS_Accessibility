#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:59:06 2019

@author: doorleyr
"""

import pandana as pdna
import json
import urllib
import requests
import os,sys
from scipy import spatial
import numpy as np
from time import sleep

# add the parent directory to path
sys.path.insert(0,os.path.abspath(os.path.join('./', os.pardir)))
from grid_geojson.grid_geojson import *

city='Hamburg'

table_name_map={'Boston':"mocho",
                'Hamburg':"grasbrook",
                'Detroit': 'corktown'}
host='https://cityio.media.mit.edu/'


lu_types_to_amenities={0: 'education', 1: 'groceries', 2:'food', 3: 'nightlife',
                       4: 'food', 5: 'food'}
MAX_SEARCH=20
NUM_POIS=2

PDNA_HDF_PATH='./python/Hamburg/data/comb_network.hdf5'
CITYIO_SAMPLE_PATH='./python/Hamburg/data/sample_cityio_data.json'
NODES_PATH='./python/Hamburg/data/comb_network_nodes.csv'
EDGES_PATH='./python/Hamburg/data/comb_network_edges.csv'
EDGES_ACCESS_AREA_PATH='./python/Hamburg/data/access_area_edges.csv'
NODES_ACCESS_AREA_PATH='./python/Hamburg/data/access_area_nodes.csv'
cityIO_grid_url=host+'api/table/'+table_name_map[city]
cityIO_output_path=host+'api/table/update/'+table_name_map[city]+'/'
access_output_path=cityIO_output_path+'access'

# =============================================================================
# Functions
def get_osm_amenies(bounds, tags):
    str_bounds=str(bounds[0])+','+str(bounds[1])+','+str(bounds[2])+','+str(bounds[3])
    osm_url_bbox=OSM_URL_ROOT+str_bounds
    with urllib.request.urlopen(osm_url_bbox) as url:
        data=json.loads(url.read().decode())
    amenities={t:{'x': [], 'y': []} for t in tags}
    for a in range(len(data['elements'])):
        for t in tags:
            for recordTag in list(data['elements'][a]['tags'].items()):
                if recordTag[0] +'_'+recordTag[1] in tags[t]:
                    amenities[t]['x'].append(data['elements'][a]['lon'])
                    amenities[t]['y'].append(data['elements'][a]['lat'])
    return amenities

def get_node_accessibility(amenities, net, max_search, num_pois):
    dist_to_second={}
    for t in amenities:
        net.set_pois(category=t, 
                     x_col=amenities[t]['x'], 
                     y_col=amenities[t]['y'])
        dist_to_second[t]=net.nearest_pois(max_search, t, 
                      num_pois=num_pois).loc[:,2]
    return dist_to_second
            
def get_grid_accessibility(amenities, net, max_search, num_pois, nearest_nodes, xs, ys):
    dist_to_second=get_node_accessibility(amenities, net, max_search, num_pois)
    grids={}
    for t in tags:
        grids[t]= [max_search for i in range(len(xs))]
        for i, v in nearest_nodes.items():
            grids[t][i]=dist_to_second[t].loc[v]
    return grids
        
def create_access_geojson(xs, ys, grids): 
    scaler=20      
    output_geojson={
     "type": "FeatureCollection",
     "features": []
    }    
    for i in range(len(xs)):
        geom={"type": "Point","coordinates": [xs[i],ys[i]]}
        props={t: 1- grids[t][i]/scaler for t in tags}
        feat={
         "type": "Feature",
         "properties": props,
         "geometry": geom
        }
        output_geojson["features"].append(feat) 
    return output_geojson
# =============================================================================

# =============================================================================
# Get cityIO data
try:
    with urllib.request.urlopen(cityIO_grid_url+'/header/spatial') as url:
    #get the latest grid data
        cityIO_spatial_data=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))
    cityIO_spatial_data=cityIO_data['header']['spatial']
    
grid=Grid(cityIO_spatial_data['longitude'], cityIO_spatial_data['latitude'], 
          cityIO_spatial_data['rotation'],  cityIO_spatial_data['cellSize'], 
          cityIO_spatial_data['nrows'], cityIO_spatial_data['ncols'])

grid_points=grid.all_cells_top_left
grid_points_ll = [[g['lon'] , g['lat']] for g in grid_points]
# =============================================================================

# =============================================================================
# Get pandana network
net=pdna.Network.from_hdf5(PDNA_HDF_PATH)
# =============================================================================

# =============================================================================
# get all amenities in within bounding box of study area
OSM_URL_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];node[~"^(amenity|leisure|shop)$"~"."];out;&bbox='

tags={
      'food': ['amenity_restaurant', 'amenity_cafe' 'amenity_fast_food', 'amenity_pub'],
      'nightlife': ['amenity_bar' , 'amenity_pub' , 'amenity_nightclub', 'amenity_biergarten'],  #(according to OSM, pubs may provide food, bars dont)
      'groceries': ['shop_convenience', 'shop_grocer', 'shop_greengrocer', 'shop_food', 'shop_supermarket'], 
      'education': ['amenity_school', 'amenity_university', 'amenity_college']
      }
# To get all amenity data
bounds_all=9.965, 53.509, 10.05, 53.55
base_amenities=get_osm_amenies(bounds_all, tags)
# initialise the accessibility scores
net.init_pois(len(base_amenities), MAX_SEARCH, NUM_POIS)
# =============================================================================

# =============================================================================
# Prepare the grid points for the output accessibility results
nodes_x=net.nodes_df['x'].values
nodes_y=net.nodes_df['y'].values
points=[[nodes_x[i], nodes_y[i]] for i in range(len(nodes_y))]

kdtree_nodes=spatial.KDTree(np.column_stack((nodes_x, nodes_y)))

min_x, min_y= min(nodes_x), min(nodes_y)
max_x, max_y= max(nodes_x), max(nodes_y)
x = (np.linspace(min_x,max_x,50))
y =  (np.linspace(min_y,max_y,50))
xs,ys = np.meshgrid(x,y)
xs=xs.reshape(xs.shape[0]*xs.shape[1])
ys=ys.reshape(ys.shape[0]*ys.shape[1])

nearest_nodes=net.get_node_ids(xs, ys, mapping_distance=0.001)
# =============================================================================

# =============================================================================
# Accessibility Analysis
lastId=0
while True:
#check if grid data changed
    try:
        with urllib.request.urlopen(cityIO_grid_url+'/meta/hashes/grid') as url:
            hash_id=json.loads(url.read().decode())
    except:
        print('Cant access cityIO')
        hash_id=1
    if hash_id==lastId:
        sleep(1)
    else:
        try:
            with urllib.request.urlopen(cityIO_grid_url+'/grid') as url:
                cityIO_grid_data=json.loads(url.read().decode())
        except:
            print('Using static cityIO grid file')
            cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))  
            cityIO_grid_data=cityIO_data['grid']
        lastId=hash_id
        amenities=base_amenities.copy()
# =============================================================================
# Fake the locations of new amenities until we have this input 
        for gi, usage in enumerate(cityIO_grid_data):
            amenities[lu_types_to_amenities[usage[0]]]['x'].append(
                    grid_points_ll[gi][0])
            amenities[lu_types_to_amenities[usage[0]]]['y'].append(
                    grid_points_ll[gi][1])
# =============================================================================
        access_grids=get_grid_accessibility(amenities, net, MAX_SEARCH, 
                                            NUM_POIS, nearest_nodes, xs, ys)
        grid_geojson=create_access_geojson(xs, ys, access_grids)
        r = requests.post(access_output_path, data = json.dumps(grid_geojson))
        print(r)
        sleep(1)     
# =============================================================================
