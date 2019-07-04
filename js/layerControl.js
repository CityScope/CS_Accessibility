function addToLayerControl(id){
	var link = document.createElement('a');
	link.href = '#';
	link.className = 'active';
	link.textContent = id;

	link.onclick = function (e) {
	  var clickedLayer = this.textContent;
	  e.preventDefault();
	  e.stopPropagation();

	  var visibility = map.getLayoutProperty(clickedLayer, 'visibility');

	  if (visibility === 'visible') {
	      map.setLayoutProperty(clickedLayer, 'visibility', 'none');
	      this.className = '';
	  } else {
	      this.className = 'active';
	      map.setLayoutProperty(clickedLayer, 'visibility', 'visible');
	  }
	};

	var layers = document.getElementById('menu');
	layers.appendChild(link);
}