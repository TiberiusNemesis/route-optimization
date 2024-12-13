import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class RouteOptimizer:
    """
    Initial implementation of a traveling salesman problem solver using the Nearest Neighbor algorithm.
    """
    def __init__(self, coordinates: List[Tuple[float, float]]):
        self.coordinates = coordinates
        self.n_locations = len(coordinates)
        self.distances = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Computed once during initialization to avoid repeated calculations.
        """
        distances = np.zeros((self.n_locations, self.n_locations))
        
        for i in range(self.n_locations):
            for j in range(i + 1, self.n_locations):
                distance = np.sqrt(
                    (self.coordinates[i][0] - self.coordinates[j][0]) ** 2 +
                    (self.coordinates[i][1] - self.coordinates[j][1]) ** 2
                )
                distances[i][j] = distance
                distances[j][i] = distance
                
        return distances
    
    def find_route(self) -> Tuple[List[int], float]:
        """
        This is simple and fast algorithm, but it may not always find the optimal route.
        At first, it visits the closest unvisited location. 
        Then it visits the next closest unvisited location.
        After that, ya guessed it: it visits the next closest unvisited location.
        This repeats on and on until all locations are visited.
        """
        current = 0 
        unvisited = set(range(1, self.n_locations))
        route = [current]
        total_distance = 0
        
        while unvisited:
            next_location = min(
                unvisited,
                key=lambda x: self.distances[current][x]
            )
            
            total_distance += self.distances[current][next_location]
            route.append(next_location)
            unvisited.remove(next_location)
            current = next_location
        
        total_distance += self.distances[route[-1]][0]
        route.append(0)
        
        return route, total_distance
    
    def get_route_coordinates(self, route: List[int]) -> List[Tuple[float, float]]:
        """Converts the route indices to actual coordinates."""
        return [self.coordinates[i] for i in route]

def read_coordinates(filename: str) -> List[Tuple[float, float]]:
    coordinates = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
    return coordinates

def visualize_route(coordinates: List[Tuple[float, float]], route: List[int]):
    """
    A visual representation of the route through matplotlib.
    """

    x_coords = [coordinates[i][0] for i in route]
    y_coords = [coordinates[i][1] for i in route]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', label='Route')
    
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Optimized Route Visualization')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    coordinates = read_coordinates('delivery_points.txt')
    optimizer = RouteOptimizer(coordinates)
    route, distance = optimizer.find_route()
    print(f"ROUTE\n{route}\n\nDISTANCE\n{distance}")
    visualize_route(coordinates, route)