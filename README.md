# Genetic Algorithm for Traveling Salesman Problem (TSP)  
**Optimizing routes between Polish cities using adaptive genetic algorithms with geospatial data integration**  

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  
[![Dependencies](https://img.shields.io/badge/dependencies-pandas%20geopandas%20matplotlib%20numpy-orange)](requirements.txt)  

## Table of Contents  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Usage](#usage)  
- [Algorithm Parameters](#algorithm-parameters)  
- [Technical Implementation](#technical-implementation)  
- [Visualization](#visualization)  
- [Customization](#customization)  

---  

## Key Features  
**Real-World Geospatial Processing**  
- Utilizes coordinates of 50 largest Polish cities  
- Automated distance matrix generation from ESRI Shapefiles  
- Integrated spatial operations with GeoPandas  

**Advanced Genetic Operators**  
- **Adaptive Mutation**: Probability increases at 33%, 66%, and 75% of max iterations  
- **Repairable Crossover**: OX-like crossover with duplicate resolution  
- **Elitist Succession**: Preserves top performers between generations  

**Dual Selection Strategies**  
- **Ranking Selection**: Strict fitness-based selection  
- **Roulette Wheel Selection**: Diversity-preserving probabilistic selection  

**Optimization Tracking**  
- Tracks best/median/75th percentile fitness across generations (in kilometers)  
- Saves intermediate routes at 25%, 50%, and 100% iterations  

---  

## Installation  
1. Clone repository:  
```bash  
git clone https://github.com/marekk13/travelling_salesman_problem.git  
cd travelling_salesman_problem  
```  

2. Install dependencies:  
```bash
pip install pandas geopandas matplotlib numpy 
```  
  

---  

## Data Preparation  
**Required Files**  
- `cities.shp`: Shapefile with coordinates of Polish cities.  
- `distances_matrix.csv`: CSV file with distances between 50 biggest Polish cities.  
- `Miejscowosci.zip`: Required only if generating new distances_matrix.csv file is needed.

**Generate the distance matrix**
```bash  
python -c "from tsp_ga import DataHandler; dh = DataHandler(); dh.generate_distance_matrix(dh.cities)"  
```  

---  

## Usage  
Initialize and run optimization:  
  
```python
from tsp_ga import DataHandler, TSPOptimizer  
dh = DataHandler()  

optimizer = TSPOptimizer(  
    dh,  
    max_iter=400,  
    n_cities=50,  # Set to 10-50 to optimize fewer cities  
    n_pop=150,  
    satisfying_result=3800  # Threshold in kilometers  
)  
optimizer.optimize() 
``` 
  

---  

## Algorithm Parameters  
| Parameter | Type | Description | Default |  
|-----------|------|-------------|---------|  
| `max_iter` | int | Maximum generations | 400 |  
| `n_pop` | int | Population size | 100 |  
| `n_cities` | int | Cities to optimize (max 50) | 50 |  
| `selection_method` | str | `ranking` or `roulette` | `ranking` |  
| `satisfying_result` | int | Early stopping threshold (km) | 4000 |  

---  

## Technical Implementation  
**Core Classes**  
- `DataHandler`: Manages geospatial data loading and distance matrix generation.  
- `TSPOptimizer`: Implements genetic algorithm with adaptive operators.  

**Key Methods**  
  
```python
# DataHandler  
def generate_distance_matrix()  # Creates CSV from SHP  
def load_geospatial_data()      # Loads processed data  

# TSPOptimizer  

def optimize()                  # Main evolutionary loop  
def _genetic_operators()        # Adaptive crossover/mutation  
def _visualize_results()        # Dual-panel matplotlib output  
```
  

---  

## Visualization  
**Output Includes**  
1. **Geospatial View**  
   - Map of Poland with optimized routes at different iterations (25%, 50%, 100%).   

2. **Convergence Analysis**  
   - Fitness evolution plot with annotations for minimum distance.  

<img width="1250" alt="wykres" src="https://github.com/user-attachments/assets/9de48843-3fda-4aa9-975e-49d6e1af0fc0" />

---  

## Customization  
**Modify City List**  
To optimize a subset of cities:  
```python
# In your script:  
dh = DataHandler()  
dh.cities = ["Warszawa", "Kraków", "Gdańsk"]  # Custom city list 
``` 


**Hybrid Optimization Example**  
```python
def hybrid_optimization():  
    optimizer = TSPOptimizer(dh, max_iter=100)  
    for _ in range(3):  
        optimizer.optimize()  
        optimizer.params["mutation_prob"] *= 0.5  
```
