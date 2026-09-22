Requires: Python - see [https://www.python.org](https://www.python.org)

## GW2MapCompTSP
Uses a modified TSP (Travelling Salesman Problem) algorithm to create a semi-optimized route across GW2 Maps from an image. It does not take in to account terrain (caves, mountains, underground, wunderwater).

The image file must be named dots.png, have a black background and must include:
* Red dot for the start location.
* Green dot for the end location.
* White dots for waypoints to visit.
* Yellow dots for all other locations to visit (POI, Vista, Mastery, Hero Point).

The yellow dots can also be placed to guide through or around some terrain.

The Output file will be route.png.

### Usage

* Take a screenshot of the map or use a map image from the wiki.
* Create a new layer and place the dots.
* Copy layer to a new image with a black background, save as dots.png in the project folder.
* Use tsp.bat or python tsp.py
* Load route.png and then add it as a new layer over the original map image.
