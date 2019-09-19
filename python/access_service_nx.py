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
from scipy import spatial
import pyproj

city='Hamburg'
lng_min, lat_min, lng_max, lat_max = 9.965, 53.509, 10.05, 53.55


table_name_map={'Boston':"mocho",
                'Hamburg':"grasbrook",
                'Detroit': 'corktown'}
host='https://cityio.media.mit.edu/'

lu_types_to_amenities={0: 'education', 1: 'groceries', 2:'food', 3: 'nightlife',
                       4: 'food'}
RADIUS=20

CITYIO_SAMPLE_PATH='./python/Hamburg/data/sample_cityio_data.json'
NODES_PATH='./python/Hamburg/data/comb_network_nodes.csv'
EDGES_PATH='./python/Hamburg/data/comb_network_edges.csv'
GRID_INT_SAMPLE_PATH='./python/Hamburg/data/grid_interactive.geojson'
GRID_FULL_SAMPLE_PATH='./python/Hamburg/data/grid_full.geojson'

local_epsg = '31468'
projection=pyproj.Proj("+init=EPSG:"+local_epsg)
wgs=pyproj.Proj("+init=EPSG:4326")

cityIO_grid_url=host+'api/table/'+table_name_map[city]
cityIO_output_path=host+'api/table/update/'+table_name_map[city]+'/'
access_output_path=cityIO_output_path+'access'

walk_speed_met_min=5*1000/60


# =============================================================================
# Functions
def get_osm_amenies(bounds, tags):
    """
    takes a list representing the bounds of the area of interest and
    a dictionary defining tag categories and the Oassociated OSM tags 
    Returns a list of amenities with their tag categories
    """
    str_bounds=str(bounds[0])+','+str(bounds[1])+','+str(bounds[2])+','+str(bounds[3])
    osm_url_bbox=OSM_URL_ROOT+str_bounds
    with urllib.request.urlopen(osm_url_bbox) as url:
        data=json.loads(url.read().decode())
    amenities={t:{'lon': [], 'lat': [], 'x':[], 'y': []} for t in tags}
    for a in range(len(data['elements'])):
        for t in tags:
            for recordTag in list(data['elements'][a]['tags'].items()):
                if recordTag[0] +'_'+recordTag[1] in tags[t]:
                    lon, lat=data['elements'][a]['lon'], data['elements'][a]['lat']
                    x,y=pyproj.transform(wgs, projection,lon, lat)
                    amenities[t]['lon'].append(lon)
                    amenities[t]['lat'].append(lat)
                    amenities[t]['x'].append(x)
                    amenities[t]['y'].append(y)
    return amenities

def create_sample_points(grid_x, grid_y, col_margin_left, row_margin_top, 
                         cell_width, cell_height,stride):
    """
    X denotes a coodrinate [x,y]
    dXdRow denotes the change in the coordinates when the row index increases by 1
    """
    dXdCol=np.array([grid_x[1]-grid_x[0], grid_y[1]-grid_y[0]])
    dXdRow=np.array([dXdCol[1], -dXdCol[0]]) # rotate the vector 90 degrees
    grid_origin=np.array([grid_x[0], grid_y[0]])
    sample_points_origin=grid_origin-row_margin_top*dXdRow-col_margin_left*dXdCol
    sample_points=np.array([sample_points_origin+stride*j*dXdCol+stride*i*dXdRow for i in range(
            int(cell_height/stride)) for j in range(int(cell_width/stride))])
    return list(sample_points[:,0]), list(sample_points[:,1])
    
        
def create_access_geojson(xs, ys, grids, scalers):
    """
    takes lists of x and y coordinates and a list containing the accessibility 
    score for each point and tag category
    """
       
    output_geojson={
     "type": "FeatureCollection",
     "features": []
    }    
    for i in range(len(xs)):
        geom={"type": "Point","coordinates": [xs[i],ys[i]]}
        props={t: np.power(grids[i][t]/scalers[t], 1) for t in base_amenities}
        feat={
         "type": "Feature",
         "properties": props,
         "geometry": geom
        }
        output_geojson["features"].append(feat) 
    return output_geojson

def createGridGraphs(grid_coords_xy, graph, nrows, ncols, cell_size, kd_tree_nodes):
    """
    returns new networks including roads around the cells
    """
#    create graph internal to the grid
    graph.add_nodes_from('g'+str(n) for n in range(len(grid_coords_xy)))
    for c in range(ncols):
        for r in range(nrows):
            # if not at the end of a row, add h link
            if not c==ncols-1:
                graph.add_edge('g'+str(r*ncols+c), 'g'+str(r*ncols+c+1), 
                      attr_dict={'weight_minutes':cell_size/walk_speed_met_min})
                graph.add_edge('g'+str(r*ncols+c+1), 'g'+str(r*ncols+c), 
                      attr_dict={'weight_minutes':cell_size/walk_speed_met_min})
            # if not at the end of a column, add v link
            if not r==nrows-1:
                graph.add_edge('g'+str(r*ncols+c), 'g'+str((r+1)*ncols+c), 
                      attr_dict={'weight_minutes':cell_size/walk_speed_met_min})
                graph.add_edge('g'+str((r+1)*ncols+c), 'g'+str(r*ncols+c), 
                      attr_dict={'weight_minutes':cell_size/walk_speed_met_min})
    # create links between the 4 corners of the grid and the road network
    for n in [0, ncols-1, (nrows-1)*ncols, (nrows*ncols)-1]: 
        dist_to_closest, closest_ind=kd_tree_nodes.query(grid_coords_xy[n], k=1)
        closest_node_id=nodes.iloc[closest_ind]['id_int']
        graph.add_edge('g'+str(n), closest_node_id, attr_dict={ 
                   'weight_minutes':dist_to_closest/walk_speed_met_min})
        graph.add_edge(closest_node_id, 'g'+str(n), attr_dict={
                   'weight_minutes':dist_to_closest/walk_speed_met_min})
    return graph 
# =============================================================================

# =============================================================================
# Get the grid data
# Interactive grid parameters
try:
    with urllib.request.urlopen(cityIO_grid_url+'/header/spatial') as url:
    #get the latest grid data
        cityIO_spatial_data=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))
    cityIO_spatial_data=cityIO_data['header']['spatial']

# Interactive grid geojson    
try:
    with urllib.request.urlopen(cityIO_grid_url+'/grid_interactive_area') as url:
    #get the latest grid data
        grid_interactive=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    grid_interactive=json.load(open(GRID_INT_SAMPLE_PATH))
    
# Full table grid geojson      
try:
    with urllib.request.urlopen(cityIO_grid_url+'/grid_full_table') as url:
    #get the latest grid data
        grid_full_table=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    grid_full_table=json.load(open(GRID_FULL_SAMPLE_PATH))
    

grid_points_ll=[f['geometry']['coordinates'][0][0] for f in grid_interactive['features']]
grid_points_x, grid_points_y=pyproj.transform(wgs, projection,
                                              [grid_points_ll[p][0] for p in range(len(grid_points_ll))], 
                                              [grid_points_ll[p][1] for p in range(len(grid_points_ll))])
grid_points_xy=[[grid_points_x[i], grid_points_y[i]] for i in range(len(grid_points_x))]
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
# Create the transport network
# Baseline network from urbanaccess results
edges=pd.read_csv(EDGES_PATH)
nodes=pd.read_csv(NODES_PATH)

nodes_lon=nodes['x'].values
nodes_lat=nodes['y'].values
nodes_x, nodes_y= pyproj.transform(wgs, projection,nodes_lon, nodes_lat)
kdtree_base_nodes=spatial.KDTree(np.column_stack((nodes_x, nodes_y)))


graph=nx.DiGraph()
for i, row in edges.iterrows():
    graph.add_edge(row['from_int'], row['to_int'], 
                     attr_dict={'weight_minutes':row['weight']})
    
amenities_at_base_nodes={n: {t:0 for t in base_amenities} for n in graph.nodes}               
# associate each amenity with its closest node in the base network
for tag in base_amenities:
    for ai in range(len(base_amenities[tag]['x'])):
        nearest_node=nodes.iloc[kdtree_base_nodes.query(
                [base_amenities[tag]['x'][ai],
                base_amenities[tag]['y'][ai]])[1]]['id_int']
        amenities_at_base_nodes[nearest_node][tag]+=1
        

# Add links for the new network defined by the interactive area    
graph=createGridGraphs(grid_points_xy, graph, cityIO_spatial_data['nrows'], 
                       cityIO_spatial_data['ncols'], cityIO_spatial_data['cellSize'], 
                       kdtree_base_nodes)

# =============================================================================

# =============================================================================
# Prepare the sample grid points for the output accessibility results
# Sampling points should correspond to points on the full_table grid but extend 
# further in the surrounding city
full_grid_lons=[f['geometry']['coordinates'][0][0][0] for f in grid_full_table['features']]
full_grid_lats=[f['geometry']['coordinates'][0][0][1] for f in grid_full_table['features']]
full_grid_x, full_grid_y= pyproj.transform(wgs, projection,full_grid_lons, full_grid_lats)
col_margin_left=70
row_margin_top=50
cell_width=250
cell_height=250
stride=5
sample_x, sample_y= create_sample_points(full_grid_x, full_grid_y, col_margin_left, 
                                         row_margin_top, cell_width, 
                                         cell_height,stride)
sample_lons, sample_lats=pyproj.transform(projection,wgs, sample_x, sample_y)

import matplotlib.pyplot as plt
plt.scatter(full_grid_x, full_grid_y, color='blue')
plt.scatter(sample_x, sample_y, color='red', alpha=0.5)
#plt.scatter([n[0] for n in all_nodes_xy], [n[1] for n in all_nodes_xy], color='green', alpha=0.1)
# =============================================================================

# =============================================================================
# Baseline Accessibility
# add virtual links joining each sample point to its closest nodes within a tolerance
# include both baseline links and new links

# first create new kdTree including the baseline nodes and the new gird nodes
all_nodes_ids, all_nodes_xy=[], []
for ind_node in range(len(nodes_x)):
    all_nodes_ids.append(nodes.iloc[ind_node]['id_int'])
    all_nodes_xy.append([nodes_x[ind_node], nodes_y[ind_node]])
for ind_grid_cell in range(len(grid_points_xy)):
    all_nodes_ids.append('g'+str(ind_grid_cell))
    all_nodes_xy.append(grid_points_xy[ind_grid_cell])

kdtree_all_nodes=spatial.KDTree(np.array(all_nodes_xy))

all_sample_node_ids=[]
for p in range(len(sample_x)):
    all_sample_node_ids.append('s'+str(p))
    graph.add_node('s'+str(p))
    distance_to_closest, closest_nodes=kdtree_all_nodes.query([sample_x[p], sample_y[p]], 5)
    for candidate in zip(distance_to_closest, closest_nodes):
        if candidate[0]<30:
            close_node_id=all_nodes_ids[candidate[1]]
            graph.add_edge('s'+str(p), close_node_id, 
                     attr_dict={'weight_minutes':candidate[0]/walk_speed_met_min})

           
# for each sample node, create an isochrone and count the amenities of each type        
sample_nodes_acc_base={n: {t:0 for t in base_amenities} for n in range(len(sample_x))} 
for sn in sample_nodes_acc_base:
    isochrone_graph=nx.ego_graph(graph, 's'+str(sn), radius=RADIUS, center=True, 
                                 undirected=False, distance='weight')
    reachable_real_nodes=[n for n in isochrone_graph.nodes if n in amenities_at_base_nodes]
    for tag in base_amenities:
        sample_nodes_acc_base[sn][tag]=sum([amenities_at_base_nodes[reachable_node][tag] 
                                            for reachable_node in reachable_real_nodes])    

# build the acessiility grid
scalers_base={t: 1*max([sample_nodes_acc_base[i][t] for i in range(
        len(sample_nodes_acc_base))]) for t in base_amenities}
grid_geojson=create_access_geojson(sample_lons, sample_lats, 
                                   sample_nodes_acc_base, scalers_base)
r = requests.post(access_output_path, data = json.dumps(grid_geojson))
print(r)
# =============================================================================

# =============================================================================
# Interactive Accessibility Analysis
# instead of recomputing the isochrone for every sample point, we will reverse 
# the graph and compute the isochrone around each new amenity
rev_graph=graph.reverse()
first_pass=True 
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
        sample_nodes_acc={n: {t:sample_nodes_acc_base[n][t] for t in base_amenities
                              } for n in range(len(sample_x))}
# =============================================================================
# Fake the locations of new amenities until we have this input 
        for gi, usage in enumerate(cityIO_grid_data):
            if usage[0] in lu_types_to_amenities:
                a_tag=lu_types_to_amenities[usage[0]]
            else:
                a_tag='food'
            a_node='g'+str(gi)
            affected_nodes=nx.ego_graph(rev_graph, a_node, radius=RADIUS, center=True, 
                                 undirected=False, distance='weight').nodes
            for n in affected_nodes:
                if n in all_sample_node_ids:
                    sample_nodes_acc[int(n.split('s')[1])][a_tag]+=1 
        if first_pass:
            scalers={t: 1.1*max([sample_nodes_acc[i][t] for i in range(
                    len(sample_nodes_acc_base))]) for t in base_amenities}
        first_pass=False
        grid_geojson=create_access_geojson(sample_lons, sample_lats, 
                                           sample_nodes_acc, scalers)
        r = requests.post(access_output_path, data = json.dumps(grid_geojson))
        print(r)
        sleep(1) 
# =============================================================================
# =============================================================================
