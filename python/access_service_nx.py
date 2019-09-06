#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:04:07 2019

@author: doorleyr
"""

import pandas as pd
import networkx as nx
from time import sleep
import json
import urllib
import requests
import numpy as np
import os,sys
from scipy import spatial
# add the parent directory to path
sys.path.insert(0,os.path.abspath(os.path.join('./', os.pardir)))
from grid_geojson.grid_geojson import *

city='Hamburg'
lng_min, lat_min, lng_max, lat_max = 9.965, 53.509, 10.05, 53.55


table_name_map={'Boston':"mocho",
                'Hamburg':"grasbrook",
                'Detroit': 'corktown'}
host='https://cityio.media.mit.edu/'

lu_types_to_amenities={0: 'education', 1: 'groceries', 2:'food', 3: 'nightlife',
                       4: 'food', 5: 'food'}
RADIUS=15

CITYIO_SAMPLE_PATH='./python/Hamburg/data/sample_cityio_data.json'
NODES_PATH='./python/Hamburg/data/comb_network_nodes.csv'
EDGES_PATH='./python/Hamburg/data/comb_network_edges.csv'
#EDGES_ACCESS_AREA_PATH='./python/Hamburg/data/access_area_edges.csv'
#NODES_ACCESS_AREA_PATH='./python/Hamburg/data/access_area_nodes.csv'
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
        
def create_access_geojson(xs, ys, grids): 
    scalers={t: max([grids[i][t] for i in range(len(grids))]) for t in base_amenities}   
    output_geojson={
     "type": "FeatureCollection",
     "features": []
    }    
    for i in range(len(xs)):
        geom={"type": "Point","coordinates": [xs[i],ys[i]]}
        props={t: grids[i][t]/scalers[t] for t in base_amenities}
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
# =============================================================================

# =============================================================================
# Get network
edges=pd.read_csv(EDGES_PATH)
nodes=pd.read_csv(NODES_PATH)
graph=nx.DiGraph()
for i, row in edges.iterrows():
    graph.add_edge(row['from_int'], row['to_int'], 
                     attr_dict={'weight_minutes':row['weight']})
rev_graph=graph.reverse()
amenities_at_nodes={n: {t:0 for t in base_amenities} for n in graph.nodes}   
# =============================================================================

# =============================================================================
# Prepare the grid points for the output accessibility results
nodes_x=nodes['x'].values
nodes_y=nodes['y'].values
points=[[nodes_x[i], nodes_y[i]] for i in range(len(nodes_y))]

kdtree_nodes=spatial.KDTree(np.column_stack((nodes_x, nodes_y)))

lng_min, lat_min, lng_max, lat_max
x = (np.linspace(lng_min,lng_max,50))
y =  (np.linspace(lat_min,lat_max,50))
xs,ys = np.meshgrid(x,y)
xs=xs.reshape(xs.shape[0]*xs.shape[1])
ys=ys.reshape(ys.shape[0]*ys.shape[1])

sample_nodes_id_int=[nodes.iloc[kdtree_nodes.query(
        [xs[i], ys[i]])[1]]['id_int'] for i in range(len(xs))]
# =============================================================================

# =============================================================================
# Baseline Accessibility
# =============================================================================
for tag in base_amenities:
    for ai in range(len(base_amenities[tag]['x'])):
        nearest_node=nodes.iloc[kdtree_nodes.query(
                [base_amenities[tag]['x'][ai],
                base_amenities[tag]['y'][ai]])[1]]['id_int']
        amenities_at_nodes[nearest_node][tag]+=1


sample_nodes_acc_base={n: {t:0 for t in base_amenities} for n in set(
        sample_nodes_id_int)}
for n in sample_nodes_acc_base:
    isochrone_graph=nx.ego_graph(graph, n, radius=RADIUS, center=True, 
                                 undirected=False, distance='weight')
    for tag in base_amenities:
        sample_nodes_acc_base[n][tag]=sum([amenities_at_nodes[n][tag] for n in isochrone_graph.nodes])
    
access_grids=[sample_nodes_acc_base[n] for n in sample_nodes_id_int]
grid_geojson=create_access_geojson(xs, ys, access_grids)
r = requests.post(access_output_path, data = json.dumps(grid_geojson))

# =============================================================================
# Interactive Accessibility Analysis
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
        sample_nodes_acc=sample_nodes_acc_base.copy()
# =============================================================================
# Fake the locations of new amenities until we have this input 
        for gi, usage in enumerate(cityIO_grid_data):
            a_tag=lu_types_to_amenities[usage[0]]
            a_node=nodes.iloc[kdtree_nodes.query([grid_points_ll[gi][0],
                                                  grid_points_ll[gi][1]])[1]]['id_int']
            affected_nodes=nx.ego_graph(rev_graph, a_node, radius=RADIUS, center=True, 
                                 undirected=False, distance='weight').nodes
            for n in affected_nodes:
                if n in sample_nodes_acc:
                    sample_nodes_acc[n][a_tag]+=1  
        access_grids=[sample_nodes_acc_base[n] for n in sample_nodes_id_int]
        grid_geojson=create_access_geojson(xs, ys, access_grids)
        r = requests.post(access_output_path, data = json.dumps(grid_geojson))
        print(r)
        sleep(1) 
# =============================================================================
# =============================================================================
