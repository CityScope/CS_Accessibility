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
import random
import requests
import numpy as np
from scipy import spatial
import pyproj
import sys

table_name=sys.argv[1]

# TODO: below is temporary solution to work for both MIT and HCU versions
if 'grasbrook' in table_name: 
    city='Hamburg'
    if table_name=='grasbrook': 
        META_GRID_SAMPLE_PATH='./python/'+city+'/data/meta_grid_sample.geojson'
        META_GRID_HEADER_SAMPLE_PATH='./python/'+city+'/data/meta_header_sample.geojson'
    else:
        META_GRID_SAMPLE_PATH='./python/'+city+'/data/meta_grid_sample_gb_comp.geojson'
        META_GRID_HEADER_SAMPLE_PATH='./python/'+city+'/data/meta_header_sample_gb_comp.geojson'
elif 'corktown' in table_name: 
    city='Detroit'
    META_GRID_SAMPLE_PATH='./python/'+city+'/data/meta_grid_sample_gb_comp.geojson'
    META_GRID_HEADER_SAMPLE_PATH='./python/'+city+'/data/meta_header_sample.geojson'
else:
    'Table name not recognised'

configs=json.load(open('./python/configs.json'))
city_configs=configs[city]

host='https://cityio.media.mit.edu/'

simple_pois_per_lu={'street': {},
                  'housing': {'housing': 1000},
                  'housing2': {'housing': 2000},
                  'working': {'employment': 1000, 'groceries': 1/5},
                  'working_2': {'employment': 2000, 'food': 1},
                  'green': {'green_space': 1}}
RADIUS=20

add_grid_roads=city_configs['add_grid_roads']

CITYIO_SAMPLE_PATH='./python/'+city+'/data/sample_cityio_data.json'

GRID_MAPPING_SAMPLE_PATH='./python/'+city+'/data/grid_mapping.json'
TYPE_MAPPING_SAMPLE_PATH='./python/'+city+'/data/type_mapping_sample.json'
UA_NODES_PATH='./python/'+city+'/data/access_network_nodes.csv'
UA_EDGES_PATH='./python/'+city+'/data/access_network_edges.csv'
ZONES_PATH='python/'+city+'/data/model_area.geojson'

local_epsg = city_configs['local_epsg']
projection=pyproj.Proj("+init=EPSG:"+local_epsg)
wgs=pyproj.Proj("+init=EPSG:4326")

cityIO_get_url=host+'api/table/'+table_name
cityIO_post_url=host+'api/table/update/'+table_name+'/'
access_post_url=cityIO_post_url+'access'
indicator_post_url=cityIO_post_url+'ind_access'

dummy_link_speed_met_min=2*1000/60


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
        props={t: np.power(grids[i][t]/scalers[t], 1) for t in all_poi_types}
        feat={
         "type": "Feature",
         "properties": props,
         "geometry": geom
        }
        output_geojson["features"].append(feat) 
    return output_geojson

def createGridGraphs(meta_grid_xy, interactive_meta_cells, graph, nrows, ncols, 
                     cell_size, kd_tree_nodes, dist_thresh):
    """
    returns new networks including roads around the cells
    """
#    create graph internal to the grid
#    graph.add_nodes_from('g'+str(n) for n in range(len(grid_coords_xy)))
    n_links_to_real_net=0
    for c in range(ncols):
        for r in range(nrows):
            cell_num=r*ncols+c
            if cell_num in interactive_meta_cells: # if this is an interactive cell
                # if close to any real nodes, make a link
                dist_to_closest, closest_ind=kd_tree_nodes.query(meta_grid_xy[cell_num], k=1)
                if dist_to_closest<dist_thresh:
                    n_links_to_real_net+=1
                    closest_node_id=nodes.iloc[closest_ind]['id_int']
                    graph.add_edge('g'+str(cell_num), closest_node_id, attr_dict={ 
                            'weight_minutes':dist_to_closest/dummy_link_speed_met_min})
                    graph.add_edge(closest_node_id, 'g'+str(cell_num), attr_dict={
                            'weight_minutes':dist_to_closest/dummy_link_speed_met_min})                    
                # if not at the end of a row, add h link
                if not c==ncols-1:
                    graph.add_edge('g'+str(r*ncols+c), 'g'+str(r*ncols+c+1), 
                          attr_dict={'weight_minutes':cell_size/dummy_link_speed_met_min})
                    graph.add_edge('g'+str(r*ncols+c+1), 'g'+str(r*ncols+c), 
                          attr_dict={'weight_minutes':cell_size/dummy_link_speed_met_min})
                # if not at the end of a column, add v link
                if not r==nrows-1:
                    graph.add_edge('g'+str(r*ncols+c), 'g'+str((r+1)*ncols+c), 
                          attr_dict={'weight_minutes':cell_size/dummy_link_speed_met_min})
                    graph.add_edge('g'+str((r+1)*ncols+c), 'g'+str(r*ncols+c), 
                          attr_dict={'weight_minutes':cell_size/dummy_link_speed_met_min})
    print(n_links_to_real_net)
    return graph 


def create_lu_to_pois_map(lu_descriptions, all_poi_types):
    pois_per_lu={}
    for lu_code, description in enumerate(lu_descriptions):
        if isinstance(description, str):
            pois_per_lu[lu_code]=simple_pois_per_lu[description]
        elif isinstance(description, dict):
            print('Complex mapping')
            this_poi_mapping={poi:0 for poi in all_poi_types}
            if description['type']=='building':
                # Ground Floor Use
                if description["bld_useGround"]=="commercial":
                    this_poi_mapping['food']= 1/2 
                    this_poi_mapping['groceries']=1/4
                    this_poi_mapping['nightlife']=1/10
                    this_poi_mapping['shopping']=1/5
                elif description["bld_useGround"]=="residential":
                    this_poi_mapping['housing']+=50
                elif description["bld_useGround"]=="office":
                    this_poi_mapping['jobs']+=50
                # Upper Floor Use
                if description["bld_useUpper"]=="residential":
                    this_poi_mapping['housing']+=50*description['bld_numLevels']
                elif description["bld_useGround"]=="office":
                    this_poi_mapping['jobs']+=50*description['bld_numLevels']
            elif description['type']=='open_space':
                if description['os_type']=='green_space':
                    this_poi_mapping['green_space']+=1
            elif description['type']=='empty':
                pass
            pois_per_lu[lu_code]=this_poi_mapping
        else:
            print('Unknown Mapping Type')
    return pois_per_lu
                 
# =============================================================================

# =============================================================================
# Get the grid data
# Interactive grid parameters
try:
    with urllib.request.urlopen(cityIO_get_url+'/header/spatial') as url:
    #get the latest grid data
        cityIO_spatial_data=json.loads(url.read().decode())
    with urllib.request.urlopen(cityIO_get_url+'/header/mapping/type') as url:
    #get the latest grid data
        type_mapping=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file for spatial parameters')
    cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))
    cityIO_spatial_data=cityIO_data['header']['spatial']
    type_mapping=json.load(open(TYPE_MAPPING_SAMPLE_PATH))
n_cells=cityIO_spatial_data['ncols']*cityIO_spatial_data['nrows']



# Full meta grid geojson      
try:
    with urllib.request.urlopen(cityIO_get_url+'/meta_grid') as url:
    #get the meta_grid from cityI/O
        meta_grid=json.loads(url.read().decode())
    with urllib.request.urlopen(cityIO_get_url+'/meta_grid_header') as url:
        meta_grid_header=json.loads(url.read().decode())
except:
    print('Using static meta_grid file for initialisation')
    meta_grid=json.load(open(META_GRID_SAMPLE_PATH))
    meta_grid_header=json.load(open(META_GRID_HEADER_SAMPLE_PATH))
    
# Interactive cell to meta_grid geojson      
int_to_meta_grid={}
for fi, f in enumerate(meta_grid['features']):
    if f['properties']['interactive']:
        int_to_meta_grid[int(f['properties']['interactive_id'])]=fi 
      
meta_grid_ll=[meta_grid['features'][i][
        'geometry']['coordinates'][0][0
        ] for i in range(len(meta_grid['features']))]

interactive_meta_cells=[v for k,v in int_to_meta_grid.items()]

meta_grid_x, meta_grid_y=pyproj.transform(wgs, projection,
                                              [meta_grid_ll[p][0] for p in range(len(meta_grid_ll))], 
                                              [meta_grid_ll[p][1] for p in range(len(meta_grid_ll))])

meta_grid_xy=[[meta_grid_x[i], meta_grid_y[i]] for i in range(len(meta_grid_x))]

grid_points_xy=[meta_grid_xy[int_to_meta_grid[int_grid_cell]]
                 for int_grid_cell in range(n_cells)]
# =============================================================================

# =============================================================================
# get all amenities in within bounding box of study area
print('Getting OSM data')
OSM_URL_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];node[~"^(amenity|leisure|shop)$"~"."];out;&bbox='
#

tags=city_configs['osm_pois']
# To get all amenity data
bounds_all=city_configs['bboxes']['amenities']
base_amenities=get_osm_amenies(bounds_all, tags)
# =============================================================================

# =============================================================================
# Get the numbers of jobs and residents in every zone in the study area
if city_configs['zonal_pois']:
    zones = json.load(open(ZONES_PATH))
# =============================================================================

# =============================================================================
# Create the transport network
# Baseline network from urbanaccess results
print('Building the base transport network')
edges=pd.read_csv(UA_EDGES_PATH)
nodes=pd.read_csv(UA_NODES_PATH)   

nodes_lon=nodes['x'].values
nodes_lat=nodes['y'].values
nodes_x, nodes_y= pyproj.transform(wgs, projection,nodes_lon, nodes_lat)
kdtree_base_nodes=spatial.KDTree(np.column_stack((nodes_x, nodes_y)))


graph=nx.DiGraph()
for i, row in edges.iterrows():
    graph.add_edge(row['from_int'], row['to_int'], 
                     attr_dict={'weight_minutes':row['weight']})
  
all_poi_types=[tag for tag in base_amenities]+city_configs['zonal_pois']
pois_at_base_nodes={n: {t:0 for t in all_poi_types} for n in graph.nodes} 

print('Finding closest node to each base POI')            
# associate each amenity with its closest node in the base network
for tag in base_amenities:
    for ai in range(len(base_amenities[tag]['x'])):
        nearest_node=nodes.iloc[kdtree_base_nodes.query(
                [base_amenities[tag]['x'][ai],
                base_amenities[tag]['y'][ai]])[1]]['id_int']
        pois_at_base_nodes[nearest_node][tag]+=1
if city_configs['zonal_pois']:
    for f in zones['features']:
        centroid_xy=pyproj.transform(wgs, projection,f['properties']['centroid'][0], 
                                     f['properties']['centroid'][1])
        distance, nearest_node_ind=kdtree_base_nodes.query(centroid_xy)
        nearest_node=nodes.iloc[nearest_node_ind]['id_int']
        if distance<1000: #(because some zones are outside the network area)
            for poi_type in city_configs['zonal_pois']:
                pois_at_base_nodes[nearest_node][poi_type]+=f['properties'][poi_type]

# Add links for the new network defined by the interactive area  
print('Adding dummy links for the grid network') 
graph=createGridGraphs(meta_grid_xy, interactive_meta_cells, graph, meta_grid_header['nrows'], 
                       meta_grid_header['ncols'], cityIO_spatial_data['cellSize'], 
                       kdtree_base_nodes, 100)

# =============================================================================

# =============================================================================
# Prepare the sample grid points for the output accessibility results
col_margin_left=city_configs['sampling_grid']['col_margin_left']
row_margin_top=city_configs['sampling_grid']['row_margin_top']
cell_width=city_configs['sampling_grid']['cell_width']
cell_height=city_configs['sampling_grid']['cell_height']
stride=city_configs['sampling_grid']['stride']
sample_x, sample_y= create_sample_points(meta_grid_x, meta_grid_y, col_margin_left, 
                                         row_margin_top, cell_width, 
                                         cell_height,stride)
sample_lons, sample_lats=pyproj.transform(projection,wgs, sample_x, sample_y)

# =============================================================================

# =============================================================================
# Baseline Accessibility
# add virtual links joining each sample point to its closest nodes within a tolerance
# include both baseline links and new links

# first create new kdTree including the baseline nodes and the new grid nodes

print('Baseline Accessibility') 

all_nodes_ids, all_nodes_xy=[], []
for ind_node in range(len(nodes_x)):
    all_nodes_ids.append(nodes.iloc[ind_node]['id_int'])
    all_nodes_xy.append([nodes_x[ind_node], nodes_y[ind_node]])
for ind_grid_cell in range(len(grid_points_xy)):
    meta_grid_id=int_to_meta_grid[ind_grid_cell]
    all_nodes_ids.append('g'+str(meta_grid_id))
    all_nodes_xy.append(meta_grid_xy[meta_grid_id])

kdtree_all_nodes=spatial.KDTree(np.array(all_nodes_xy))

# add the virtual links between sample points and closest nodes
MAX_DIST_VIRTUAL=30
all_sample_node_ids=[]
for p in range(len(sample_x)):
    all_sample_node_ids.append('s'+str(p))
    graph.add_node('s'+str(p))
    distance_to_closest, closest_nodes=kdtree_all_nodes.query([sample_x[p], sample_y[p]], 5)
    for candidate in zip(distance_to_closest, closest_nodes):
        if candidate[0]<MAX_DIST_VIRTUAL:
            close_node_id=all_nodes_ids[candidate[1]]
            graph.add_edge('s'+str(p), close_node_id, 
                     attr_dict={'weight_minutes':candidate[0]/(dummy_link_speed_met_min)})


# for each sample node, create an isochrone and count the amenities of each type        
sample_nodes_acc_base={n: {poi_type:0 for poi_type in all_poi_types} for n in range(len(sample_x))} 
for sn in sample_nodes_acc_base:
    isochrone_graph=nx.ego_graph(graph, 's'+str(sn), radius=RADIUS, center=True, 
                                 undirected=False, distance='weight')
    reachable_real_nodes=[n for n in isochrone_graph.nodes if n in pois_at_base_nodes]
    for poi_type in all_poi_types:
        sample_nodes_acc_base[sn][poi_type]=sum([pois_at_base_nodes[reachable_node][poi_type] 
                                            for reachable_node in reachable_real_nodes])    

# build the acessibility grid
scalers_base={poi_type: 1*max([sample_nodes_acc_base[i][poi_type] for i in range(
        len(sample_nodes_acc_base))]) for poi_type in all_poi_types}
grid_geojson=create_access_geojson(sample_lons, sample_lats, 
                                   sample_nodes_acc_base, scalers_base)
avg_access={t: np.mean([sample_nodes_acc_base[g][t
                        ] for g in sample_nodes_acc_base]
            ) for t in sample_nodes_acc_base[0]}
r = requests.post(access_post_url, data = json.dumps(grid_geojson))
print('Base geojson: {}'.format(r))
r = requests.post(indicator_post_url, data = json.dumps(avg_access))
print('Base indicators: {}'.format(r))

# =============================================================================

# =============================================================================
# Interactive Accessibility Analysis
# instead of recomputing the isochrone for every sample point, we will reverse 
# the graph and compute the isochrone around each new amenity
print('Preparing for interactve updates') 
rev_graph=graph.reverse()
# find the sample nodes affected by each interactive grid cell
affected_sample_nodes={}
for gi in range(len(grid_points_xy)):
    # TODO: use the meta_grid index
    a_node='g'+str(int_to_meta_grid[gi])
    affected_nodes=nx.ego_graph(rev_graph, a_node, radius=RADIUS, center=True, 
     undirected=False, distance='weight').nodes
    affected_sample_nodes[gi]=[n for n in affected_nodes if n in all_sample_node_ids]

first_pass=True 
lastId=0
last_header_id=0
mapping_data={}
pois_per_lu=create_lu_to_pois_map(mapping_data, all_poi_types)
print('Listening for grid updates') 
while True:
# =============================================================================
#     check if type mapping changed
# =============================================================================
    try:
        with urllib.request.urlopen(cityIO_get_url+'/meta/hashes/header') as url:
            header_hash_id=json.loads(url.read().decode())
    except:
        print('Cant access city_IO header hash')
    if not header_hash_id==last_header_id:
        print('Getting new header data')
        try:
            with urllib.request.urlopen(cityIO_get_url+'/header/mapping/type') as url:
                mapping_data=json.loads(url.read().decode())
                pois_per_lu=create_lu_to_pois_map(mapping_data, all_poi_types)
            last_header_id=header_hash_id
        except:
            print('Cant access city_IO header data')    
# =============================================================================
#     check if grid data changed
# =============================================================================
    try:
        with urllib.request.urlopen(cityIO_get_url+'/meta/hashes/grid') as url:
            hash_id=json.loads(url.read().decode())
    except:
        print('Cant access city_IO grid hash')
        hash_id=1
    if hash_id==lastId:
        sleep(0.2)
    else:
        try:
            with urllib.request.urlopen(cityIO_get_url+'/grid') as url:
                cityIO_grid_data=json.loads(url.read().decode())
        except:
            print('Cant access city_IO grid data')
        lastId=hash_id
        sample_nodes_acc={n: {t:sample_nodes_acc_base[n][t] for t in all_poi_types
                              } for n in range(len(sample_x))}
        for gi, usage in enumerate(cityIO_grid_data):
            if not type(usage)==list:
                print('Usage value is not a list: '+str(usage))
                usage=[-1,-1]
            this_grid_lu=int(usage[0])
            if this_grid_lu in pois_per_lu:
                sample_nodes_to_update=affected_sample_nodes[gi]
                for poi in pois_per_lu[this_grid_lu]:
                    if poi in all_poi_types:
                        n_to_add=pois_per_lu[this_grid_lu][poi]
                        if n_to_add<1:
                            if random.uniform(0,1)<=n_to_add:
                                n_to_add=1
                            else:
                                n_to_add=0
                        for n in sample_nodes_to_update:
                            sample_nodes_acc[int(n.split('s')[1])][poi]+=n_to_add
        if first_pass:
            scalers={t: 1.1*max([sample_nodes_acc[i][t] for i in range(
                    len(sample_nodes_acc_base))]) for t in all_poi_types}
        first_pass=False
        avg_access={t: np.mean([sample_nodes_acc[g][t
                        ] for g in sample_nodes_acc]
            ) for t in sample_nodes_acc[0]}
        grid_geojson=create_access_geojson(sample_lons, sample_lats, 
                                           sample_nodes_acc, scalers)
        r = requests.post(access_post_url, data = json.dumps(grid_geojson))
        print('Geojson: {}'.format(r))
        r = requests.post(indicator_post_url, data = json.dumps(avg_access))
        print('Indicators: {}'.format(r))
        sleep(0.5) 

# =============================================================================
