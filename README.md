# unbalancedGerryMandering

This package is related to the Gerrymendering project that allows different population weights.

## Contents
- `unbalancedGerry/` directory containing the generation algorithm, calculation of some map statistics, and the map drawing methods.
- `data/` the data related to Wisconsin elections and shapefile
- `requriements.txt` environment were our experiments are based on 
- `demonstration.ipynb`  an example of the map generation algorithm
- `exampleOutput/`  a repo created for storing the temporary example in demonstration.ipynb

## Algorithm details
The algorithm here does two things.
1. Generate an initial map by randomly merging a unit to its closest adjacent unit at a time until the number of units equals the number of districts wanted.
2. Keep exchanging units at the border of two districts until the population ratio is satisfied.

The way to call the main routine is the following (it could be found in `demonstration.ipynb`):
<pre>
result = generateMap(adjList = adjList,
                     districtNum = DISTRICT_NUM,
                     population = population,
                     position = position,
                     popWeights = [2,2,2,1,1],
                     popBound = 0.05,
                     iterations=10000)`
</pre>


1. adjList : list that stores the adjacency information for each pair of units
2. districtNum : district number we want in our final map
3. population : population of each unit
4. popWeights : final portion of population for each district in an array
5. popBound : tolerance for population weights
6. iterations : iteration limit for the second step in the algorithm.