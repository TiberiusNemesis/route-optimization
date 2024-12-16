from abc import ABC, abstractmethod
from decimal import Decimal
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Location:
    """Represents a service location with coordinates and time constraints."""
    x: float
    y: float

class RouteStrategy(ABC):
    """
    An abstract base class for different routing strategies.
    This allows us to easily swap between different algorithms.
    """
    @abstractmethod
    def find_route(
        self,
        locations: List[Location],
        distances: np.ndarray,
    ) -> Tuple[List[int], float]:
        """Find a route through all locations."""
        pass
    
class NearestNeighborStrategy(RouteStrategy):
    """
    Implementation of the nearest neighbor strategy.
    Builds a route by always choosing the closest unvisited location.
    """
    def find_route(
        self,
        locations: List[Location],
        distances: np.ndarray
    ) -> Tuple[List[int], float]:
        n_locations = len(locations)
        current = 0
        unvisited = set(range(1, n_locations))
        route = [current]
        total_distance = Decimal('0')
        
        while unvisited:
            best_next = None
            best_distance = Decimal('infinity')
            
            for next_loc in unvisited:
                distance = Decimal(str(distances[current][next_loc]))
                
                if distance < best_distance:
                    best_distance = distance
                    best_next = next_loc
            
            total_distance += best_distance
            route.append(best_next)
            
            current = best_next
            unvisited.remove(best_next)
        
        # Assuming we return to start, maybe not necessary
        total_distance += Decimal(str(distances[route[-1]][0]))
        route.append(0)
        
        return route, float(total_distance)  

class TwoOptStrategy(RouteStrategy):
    """
    Implements 2-opt improvement strategy.
    Looks for crossing paths in the route and untangles them to create a shorter path.
    """
    def __init__(self, base_strategy: RouteStrategy):
        self.base_strategy = base_strategy
    
    def find_route(
        self,
        locations: List[Location],
        distances: np.ndarray,
    ) -> Tuple[List[int], float]:
        route, total_distance = self.base_strategy.find_route(
            locations, distances
        )
        
        improved = True
        total_distance = Decimal(str(total_distance))
        
        while improved:
            improved = False
            
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = self._try_two_opt_swap(route, i, j)
                    new_distance = self._calculate_total_distance(
                        new_route,
                        distances
                    )
                    
                    if new_distance < total_distance:
                        route = new_route
                        total_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return route, float(total_distance) 
    
    def _try_two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        """
        Create a new route by reversing the segment between i and j.
        This is the core operation of the 2-opt algorithm.
        """
        new_route = route[:i]
        new_route.extend(reversed(route[i:j + 1]))
        new_route.extend(route[j + 1:])
        return new_route
    
    def _calculate_total_distance(
        self,
        route: List[int],
        distances: np.ndarray
    ) -> Decimal:
        """Calculate the total distance of a route using Decimal for precision."""
        return sum(
            Decimal(str(distances[route[i]][route[i+1]]))
            for i in range(len(route)-1)
        )
class RouteValidationError(Exception):
    """Custom exception for route validation errors."""
    pass

class RouteOptimizer:
    """
    Initial implementation of a traveling salesman problem solver using the Nearest Neighbor algorithm.
    Can take into account time windows and service durations.
    """
    def __init__(
        self,
        locations: List[Location],
        strategy: Optional[RouteStrategy] = None
    ):
        if not locations:
            raise ValueError("Must provide at least one location")
            
        self.locations = locations
        self.distances = self._calculate_distance_matrix()
        
        self.strategy = strategy or TwoOptStrategy(NearestNeighborStrategy())
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Creates a matrix of distances between all locations.
        Computed once during initialization to avoid repeated calculations.
        """
        n = len(self.locations)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.sqrt(
                    (self.locations[i].x - self.locations[j].x) ** 2 +
                    (self.locations[i].y - self.locations[j].y) ** 2
                )
                distances[i][j] = distance
                distances[j][i] = distance
                
        return distances

    def find_route(self) -> Tuple[List[int], float]:
        """Find the best route using the current strategy."""
        return self.strategy.find_route(
            self.locations,
            self.distances
        )
    
    def set_strategy(self, strategy: RouteStrategy):
        """Change the routing strategy at runtime."""
        self.strategy = strategy

def read_locations(filename: str) -> List[Location]:
    locations = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    raise ValueError(f"Invalid x,y coordinates")
                    
                x, y = map(float, parts[:2])
                
                locations.append(Location(
                    x=x,
                    y=y
                ))
            except ValueError as e:
                raise ValueError(f"Error reading line {i}: {str(e)}")
    return locations

def visualize_route(locations: List[Location], route: List[int]):
    """
    A visual representation of the route through matplotlib.
    """

    x_coords = [locations[i].x for i in route]
    y_coords = [locations[i].y for i in route]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', label='Route')
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(i), fontsize=11, ha='right')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Optimized Route Visualization')
    plt.grid(True)
    plt.legend()
    plt.show()