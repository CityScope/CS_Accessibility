var map
mapboxgl.accessToken = MAPBOX_TOKEN;

map = new mapboxgl.Map({
      container: 'map', // container id
      style: 'mapbox://styles/mapbox/dark-v9',
      center: [ 10, 53.5], // starting position [lng, lat]
      // center: [0,0],
      zoom: 8 ,// starting zoom
      pitch: 0
  }); 

map.on('load', function(){ 
  // d3.json("./python/Boston/output_geojson.geojson", function(data) {
    d3.json("https://cityio.media.mit.edu/api/table/grasbrook/access", function(data) {
    console.log(data);
    makeMap(data);
  });
});

function makeMap(input_points_geojson) {  
  console.log(input_points_geojson.features[0])	
  for (var key in input_points_geojson.features[0].properties) {
  	console.log(key)
    if (typeof map.getSource(key) == "undefined"){
      console.log('Creating '+key+' for first time')
      map.addSource(key, { type: 'geojson', data: input_points_geojson });
      map.addLayer({
                    "id": key,
                    "type": "circle",
                    "source": key,
                      'paint': {
                      'circle-color': ["case", 
                      ['<',['number',['get', key]],0.2],['rgb', 200,0,0],
                      ['<',['number',['get', key]],0.4],['rgb', 200,100,0],
                      ['<',['number',['get', key]],0.6],['rgb', 200,200,0],
                      ['<',['number',['get', key]],0.8],['rgb', 100,200,0],
                      ['rgb', 0,200,0]],
                      'circle-radius':5,
                      'circle-opacity': 0.3
                      }           
          });
      ////////Toggle capability//////
      addToLayerControl(key);
      ////////Toggle capability//////
    }
    else{
      //update data only
      map.getSource(key).setData(input_points_geojson); 
      console.log('updating '+key) 
    }
    
  }

}

  