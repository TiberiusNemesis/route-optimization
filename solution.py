from typing_extensions import deprecated
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

class RouteValidationError(Exception):
    """Custom exception for route validation errors."""
    pass

class RouteOptimizer:
    """
    Initial implementation of a traveling salesman problem solver using the Nearest Neighbor algorithm.
    Enhanced with time windows and service durations.
    """
    def __init__(
        self,
        locations: List[Location],
        start_time: Optional[datetime] = None,
        max_route_duration: Optional[timedelta] = None
    ):
        if not locations:
            raise ValueError("At least one location is required")
            
        self.locations = locations
        self.n_locations = len(locations)
        self.start_time = start_time or datetime.now()
        self.max_route_duration = max_route_duration or timedelta(hours=10)
        self.distances = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Computed once during initialization to avoid repeated calculations.
        """
        distances = np.zeros((self.n_locations, self.n_locations))
        
        for i in range(self.n_locations):
            for j in range(i + 1, self.n_locations):
                print(type(self.locations[i]), type(self.locations[j]))
                print(str(self.locations[i]) + " " + str(self.locations[j]))
                distance = np.sqrt(
                    (self.locations[i].x - self.locations[j].x) ** 2 +
                    (self.locations[i].y - self.locations[j].y) ** 2
                )
                distances[i][j] = distance
                distances[j][i] = distance
                
        return distances

    def _estimate_travel_time(self, distance: float) -> timedelta:
        """Estimate travel time based on distance."""
        hours = distance / 30.0
        return timedelta(hours=hours)
    
    def _validate_route(
        self,
        route: List[int],
        times: List[datetime]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a route against the time window constraints.
        """
        route_duration = times[-1] - times[0]
        if route_duration > self.max_route_duration:
            return False, f"Route duration {route_duration} exceeds maximum {self.max_route_duration}"
            
        for i, (location_idx, arrival_time) in enumerate(zip(route, times)):
            location = self.locations[location_idx]
            if not location.is_available_at(arrival_time):
                return False, f"Location {location_idx} not available at {arrival_time}"
                
        return True, None
    
    def find_route(self) -> Tuple[List[int], List[datetime], float]:
        """
        This is simple and fast algorithm, but it may not always find the optimal route.
        At first, it visits the closest unvisited location. 
        Then it visits the next closest unvisited location.
        After that, ya guessed it: it visits the next closest unvisited location.
        This repeats on and on until all locations are visited.
        Now enhanced with time windows and service duration considerations.
        """
        current = 0 
        unvisited = set(range(1, self.n_locations))
        route = [current]
        arrival_times = [self.start_time]
        total_distance = 0
        
        current_time = self.start_time + self.locations[0].service_duration()
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            best_arrival_time = None
            
            for next_loc in unvisited:
                distance = self.distances[current][next_loc]
                travel_time = self._estimate_travel_time(distance)
                arrival_time = current_time + travel_time
                
                if not self.locations[next_loc].is_available_at(arrival_time):
                    print(f"Location {next_loc} not available at {arrival_time}")
                    continue
                
                score = distance / self.locations[next_loc].priority
                
                if score < best_score:
                    best_score = score
                    best_next = next_loc
                    best_arrival_time = arrival_time
            
            if best_next is None:
                raise RouteValidationError("No valid route found")
            
            total_distance += self.distances[current][best_next]
            route.append(best_next)
            arrival_times.append(best_arrival_time)
            current_time = best_arrival_time + self.locations[best_next].service_duration()
            current = best_next
            unvisited.remove(best_next)
        
        final_arrival = current_time + self._estimate_travel_time(
            self.distances[route[-2]][0]
        )
        arrival_times.append(final_arrival)
        total_distance += self.distances[route[-1]][0]
        route.append(0)

        is_valid, error = self._validate_route(route, arrival_times)
        if not is_valid:
            raise RouteValidationError(error)
        
        return route, arrival_times, total_distance
    
    def get_route_coordinates(self, route: List[int]) -> List[Tuple[float, float]]:
        """Converts the route indices to actual coordinates."""
        return [self.locations[i] for i in route]

def read_locations(filename: str) -> List[Location]:
    locations = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    raise ValueError(f"Line {i}: Invalid x,y coordinates")
                    
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

@deprecated("Use Location class instead")
def read_coordinates(filename: str) -> List[Tuple[float, float]]:
    coordinates = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))
    return coordinates

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
    # coordinates = read_coordinates('delivery_points.txt')
    locations = read_locations('delivery_points.txt')
    optimizer = RouteOptimizer(locations)
    route, arrival_times, distance = optimizer.find_route()
    arrival_times = [time.strftime('%Y-%m-%d %H:%M:%S') for time in arrival_times]
    print(f"ROUTE\n{route}\n\nARRIVAL TIMES\n{arrival_times}\n\nDISTANCE\n{distance:.2f}")
    visualize_route(locations, route)