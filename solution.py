from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Location:
    """Represents a service location with coordinates and time constraints."""
    x: float
    y: float
    service_time_minutes: int = 30
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    priority: int = 1  # higher number == more urgent
    
    def is_available_at(self, time: datetime) -> bool:
        """Check if location can be serviced at given time."""
        if self.time_window_start is None or self.time_window_end is None:
            return True
        return self.time_window_start <= time <= self.time_window_end
    
    def service_duration(self) -> timedelta:
        """Get service duration as timedelta."""
        return timedelta(minutes=self.service_time_minutes)

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
        start_time: datetime,
        max_duration: timedelta
    ) -> Tuple[List[int], List[datetime], float]:
        """Find a route through all locations."""
        pass
    
    def _is_route_valid(
        self,
        route: List[int],
        times: List[datetime],
        locations: List[Location],
        max_duration: timedelta
    ) -> bool:
        """Check if a route satisfies all the constraints."""
        if times[-1] - times[0] > max_duration:
            return False
            
        for location_index, arrival_time in zip(route, times):
            if not locations[location_index].is_available_at(arrival_time):
                return False
        
        return True
    
class NearestNeighborStrategy(RouteStrategy):
    """
    Implementation of the nearest neighbor strategy.
    Now modified to work with a strategy pattern.
    """
    def find_route(
        self,
        locations: List[Location],
        distances: np.ndarray,
        start_time: datetime,
        max_duration: timedelta
    ) -> Tuple[List[int], List[datetime], float]:
        n_locations = len(locations)
        current = 0
        unvisited = set(range(1, n_locations))
        route = [current]
        arrival_times = [start_time]
        total_distance = 0
        
        current_time = start_time + locations[0].service_duration()
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            best_arrival_time = None
            
            for next_loc in unvisited:
                distance = distances[current][next_loc]
                travel_time = self._estimate_travel_time(distance)
                arrival_time = current_time + travel_time
                
                if not locations[next_loc].is_available_at(arrival_time):
                    continue
                
                score = distance / locations[next_loc].priority
                
                if score < best_score:
                    best_score = score
                    best_next = next_loc
                    best_arrival_time = arrival_time
            
            if best_next is None:
                raise RouteValidationError("Cannot find valid route with time constraints")
            
            total_distance += distances[current][best_next]
            route.append(best_next)
            arrival_times.append(best_arrival_time)
            
            current_time = best_arrival_time + locations[best_next].service_duration()
            current = best_next
            unvisited.remove(best_next)
        
        # Assuming we return to start, maybe not necessary
        total_distance += distances[route[-1]][0]
        route.append(0)
        final_arrival = current_time + self._estimate_travel_time(
            distances[route[-2]][0]
        )
        arrival_times.append(final_arrival)

        if not self._is_route_valid(route, arrival_times, locations, max_duration):
            # TODO: We are not at all handling invalid routes for this strategy.
            # Ideally, we return the best one we can find.
            raise RouteValidationError("Cannot find valid route with time constraints")
        
        return route, arrival_times, total_distance
    
    def _estimate_travel_time(self, distance: float) -> timedelta:
        """Estimate travel time based on distance."""
        hours = distance / 30.0
        return timedelta(hours=hours)

class TwoOptStrategy(RouteStrategy):
    """
    Implements 2-opt, which is an improvement strategy.
    It looks for crossing paths in the route, then untangles them to create a shorter path.
    """
    def __init__(self, base_strategy: RouteStrategy):
        self.base_strategy = base_strategy
    
    def find_route(
        self,
        locations: List[Location],
        distances: np.ndarray,
        start_time: datetime,
        max_duration: timedelta
    ) -> Tuple[List[int], List[datetime], float]:
        route, arrival_times, total_distance = self.base_strategy.find_route(
            locations, distances, start_time, max_duration
        )
        
        improved = True
        while improved:
            improved = False
            
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = self._try_two_opt_swap(route, i, j)
                    new_times = self._calculate_arrival_times(
                        new_route,
                        locations,
                        distances,
                        start_time
                    )
                    
                    if new_times and self._is_route_valid(
                        new_route,
                        new_times,
                        locations,
                        max_duration
                    ):
                        new_distance = self._calculate_total_distance(
                            new_route,
                            distances
                        )
                        
                        # This "-1e-10" is to account for floating point errors.
                        # The other option is to use the decimal library. 
                        # It's likely more precise, and better in the long-term.
                        if new_distance < total_distance - 1e-10: 
                            route = new_route
                            arrival_times = new_times
                            total_distance = new_distance
                            improved = True
                            break
                
                if improved:
                    break
        
        return route, arrival_times, total_distance
    
    def _try_two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        """
        Create a new route by reversing the segment between i and j, which is the core operation of the 2-opt algorithm.
        """
        new_route = route[:i]
        new_route.extend(reversed(route[i:j + 1]))
        new_route.extend(route[j + 1:])
        return new_route
    
    def _calculate_arrival_times(
        self,
        route: List[int],
        locations: List[Location],
        distances: np.ndarray,
        start_time: datetime
    ) -> Optional[List[datetime]]:
        """
        Calculate the arrival times for a route.
        Returns None if our time windows are violated.
        """
        times = [start_time]
        current_time = start_time + locations[route[0]].service_duration()
        
        for i in range(1, len(route)):
            travel_time = self._estimate_travel_time(
                distances[route[i-1]][route[i]]
            )
            arrival_time = current_time + travel_time
            
            if not locations[route[i]].is_available_at(arrival_time):
                return None
            
            times.append(arrival_time)
            current_time = arrival_time + locations[route[i]].service_duration()
        
        return times
    
    def _calculate_total_distance(
        self,
        route: List[int],
        distances: np.ndarray
    ) -> float:
        """Calculates the total distance of a route."""
        return sum(
            distances[route[i]][route[i+1]]
            for i in range(len(route)-1)
        )
    
    def _estimate_travel_time(self, distance: float) -> timedelta:
        """Estimates travel time based on distance."""
        hours = distance / 30.0 
        # this means 30km/h as an avg speed for the property engineer. 
        # I honestly have no idea how to better estimate this but this sounded reasonable
        return timedelta(hours=hours)
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
        start_time: datetime,
        max_route_duration: timedelta = timedelta(hours=8),
        strategy: Optional[RouteStrategy] = None
    ):
        if not locations:
            raise ValueError("Must provide at least one location")
            
        self.locations = locations
        self.start_time = start_time
        self.max_route_duration = max_route_duration
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

    def find_route(self) -> Tuple[List[int], List[datetime], float]:
        """Find the best route using the current strategy."""
        return self.strategy.find_route(
            self.locations,
            self.distances,
            self.start_time,
            self.max_route_duration
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
                
                service_time = int(parts[2]) if len(parts) > 2 else 30
                priority = int(parts[3]) if len(parts) > 3 else 1
                
                locations.append(Location(
                    x=x,
                    y=y,
                    service_time_minutes=service_time,
                    priority=priority
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

if __name__ == '__main__':
    locations = read_locations('delivery_points.txt')
    optimizer = RouteOptimizer(locations, datetime(2024, 12, 16, 9, 0, 0))
    route, arrival_times, distance = optimizer.find_route()
    arrival_times = [time.strftime('%Y-%m-%d %H:%M:%S') for time in arrival_times]
    print(f"ROUTE\n{route}\n\nARRIVAL TIMES\n{arrival_times}\n\nDISTANCE\n{distance:.2f}")
    visualize_route(locations, route)